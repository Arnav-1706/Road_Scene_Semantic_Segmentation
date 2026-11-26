from pathlib import Path
from threading import Event, Lock, Thread
from time import perf_counter
from uuid import uuid4

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, url_for
from werkzeug.utils import secure_filename

from inference_engine import (
    CLASS_NAMES,
    InferenceManager,
    annotate_frame_status,
    colorize_mask_to_bgr,
    draw_detections,
)


# --------------------------------------------------
# 1. Flask & Pipeline Configuration
# --------------------------------------------------

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

UPLOAD_DIR = BASE_DIR / "static" / "uploads"
RESULT_DIR = BASE_DIR / "static" / "results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov"}

VIDEO_MODES = {
    "high": {"label": "High Quality", "stride": 1, "description": "Analyze every frame"},
    "balanced": {"label": "Balanced", "stride": 3, "description": "Analyze every 3rd frame"},
    "fast": {"label": "Fast", "stride": 5, "description": "Analyze every 5th frame"},
}

app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

# Lazy, GPU-first inference manager (ONNX Runtime CUDA + PyTorch CUDA with CPU fallback)
engine = InferenceManager(MODEL_DIR)

# Thread-safe in-memory job store for real-time video progress and cancellation
VIDEO_JOBS: dict[str, dict] = {}
VIDEO_JOBS_LOCK = Lock()
MAX_STORED_JOBS = 20


# --------------------------------------------------
# 2. Helpers
# --------------------------------------------------

def allowed_image_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


def allowed_video_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


def _prune_old_jobs():
    if len(VIDEO_JOBS) <= MAX_STORED_JOBS:
        return
    removable = [
        jid for jid, job in VIDEO_JOBS.items()
        if job["status"] in {"completed", "cancelled", "error"}
    ]
    for jid in removable[:max(0, len(VIDEO_JOBS) - MAX_STORED_JOBS)]:
        VIDEO_JOBS.pop(jid, None)


def _update_job(job_id: str, **updates):
    with VIDEO_JOBS_LOCK:
        job = VIDEO_JOBS.get(job_id)
        if job is not None:
            job.update(updates)


def _compute_class_distribution(pixel_counts: np.ndarray) -> list[dict]:
    total_pixels = int(pixel_counts.sum())
    if total_pixels <= 0:
        return []
    distribution = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        count = int(pixel_counts[class_id])
        if count > 0:
            distribution.append({
                "class": class_name,
                "pixels": count,
                "percentage": round((count / total_pixels) * 100.0, 2),
            })
    return distribution


class VideoCancelledError(Exception):
    """Raised when the user cancels an active video analysis job."""


def process_video_pipeline(
    input_path: Path,
    output_path: Path,
    output_filename: str,
    model_name: str,
    use_yolo: bool,
    mode: str,
    cancel_event: Event | None = None,
    job_id: str | None = None,
) -> dict:
    start_time = perf_counter()
    mode_info = VIDEO_MODES[mode]
    stride = mode_info["stride"]
    mode_label = mode_info["label"]

    cap = None
    writer = None

    try:
        dev_info = engine.get_active_device_info(model_name, use_yolo)
        if job_id:
            _update_job(
                job_id,
                status="running",
                device=dev_info["device"],
                device_name=dev_info["device_name"],
                seg_backend=dev_info["seg_backend"],
            )

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError("Unable to open uploaded video.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            fps = 25.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if width <= 0 or height <= 0:
            raise ValueError("Invalid video dimensions.")

        if job_id:
            _update_job(job_id, total_frames=max(total_frames, 0))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("Unable to create output video.")

        frame_idx = 0
        analyzed_count = 0
        reused_count = 0
        detection_counts: dict[str, int] = {}
        analyzed_pixel_counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)

        latest_colored_bgr = None
        latest_dets: list[dict] = []

        # Batch YOLOv8n across multiple analyzed frames while keeping progress updates responsive
        yolo_chunk_target = engine.yolo_batch_size if use_yolo else 4

        while True:
            if cancel_event and cancel_event.is_set():
                raise VideoCancelledError("Video processing cancelled by user.")

            chunk_items = []
            analyzed_frames_in_chunk = []

            while len(analyzed_frames_in_chunk) < yolo_chunk_target:
                if cancel_event and cancel_event.is_set():
                    raise VideoCancelledError("Video processing cancelled by user.")
                success, frame = cap.read()
                if not success:
                    break
                is_analyzed = (frame_idx % stride == 0)
                chunk_items.append((frame_idx, frame, is_analyzed))
                if is_analyzed:
                    analyzed_frames_in_chunk.append(frame)
                frame_idx += 1

            if not chunk_items:
                break

            # Run batched YOLOv8n across all analyzed frames in this chunk
            parsed_yolo_batch = []
            if use_yolo and analyzed_frames_in_chunk:
                if cancel_event and cancel_event.is_set():
                    raise VideoCancelledError("Video processing cancelled by user.")
                _, parsed_yolo_batch = engine.predict_yolo_batch(analyzed_frames_in_chunk)

            analyzed_ptr = 0

            for _, frame, is_analyzed in chunk_items:
                if cancel_event and cancel_event.is_set():
                    raise VideoCancelledError("Video processing cancelled by user.")

                if is_analyzed:
                    mask_512 = engine.predict_mask_batch(model_name, [frame])[0]
                    analyzed_pixel_counts += np.bincount(mask_512.ravel(), minlength=len(CLASS_NAMES))
                    latest_colored_bgr = colorize_mask_to_bgr(mask_512, width, height)

                    if use_yolo:
                        latest_dets = parsed_yolo_batch[analyzed_ptr]
                        for det in latest_dets:
                            lbl = det["label"]
                            detection_counts[lbl] = detection_counts.get(lbl, 0) + 1
                    else:
                        latest_dets = []

                    analyzed_ptr += 1
                    analyzed_count += 1
                else:
                    reused_count += 1

                out_frame = cv2.addWeighted(frame, 0.55, latest_colored_bgr, 0.45, 0)
                if use_yolo and latest_dets:
                    draw_detections(out_frame, latest_dets)
                annotate_frame_status(out_frame, is_analyzed, mode_label)
                writer.write(out_frame)

                if job_id:
                    processed = analyzed_count + reused_count
                    elapsed = max(perf_counter() - start_time, 1e-4)
                    cur_fps = processed / elapsed
                    if total_frames > 0:
                        pct = min(99.5, (processed / total_frames) * 100.0)
                        eta = max(0.0, (total_frames - processed) / max(cur_fps, 1e-4))
                    else:
                        pct = 0.0
                        eta = None

                    _update_job(
                        job_id,
                        processed_frames=processed,
                        analyzed_frames=analyzed_count,
                        reused_frames=reused_count,
                        percent=round(pct, 1),
                        current_fps=round(cur_fps, 2),
                        elapsed_seconds=round(elapsed, 2),
                        eta_seconds=round(eta, 1) if eta is not None else None,
                    )

        cap.release()
        cap = None
        writer.release()
        writer = None

        total_written = analyzed_count + reused_count
        if total_written == 0:
            raise ValueError("No readable frames found.")

        processing_time = round(perf_counter() - start_time, 2)
        total_detections = sum(detection_counts.values())
        avg_per_analyzed = round(total_detections / analyzed_count, 2) if analyzed_count > 0 else 0.0
        class_distribution = _compute_class_distribution(analyzed_pixel_counts)

        return {
            "success": True,
            "video": f"/static/results/{output_filename}",
            "model": model_name,
            "mode": mode,
            "mode_label": mode_label,
            "stride": stride,
            "device": dev_info["device"],
            "device_name": dev_info["device_name"],
            "seg_backend": dev_info["seg_backend"],
            "frames_processed": total_written,
            "analyzed_frames": analyzed_count,
            "reused_frames": reused_count,
            "original_fps": round(fps, 2),
            "duration_seconds": round(total_written / fps, 2),
            "processing_time": processing_time,
            "processing_fps": round(total_written / max(processing_time, 1e-4), 2),
            "total_detections": total_detections,
            "avg_detections_per_analyzed_frame": avg_per_analyzed,
            "detection_counts": detection_counts,
            "class_distribution": class_distribution,
        }

    except VideoCancelledError:
        if cap is not None:
            cap.release()
            cap = None
        if writer is not None:
            writer.release()
            writer = None
        for p in (output_path, input_path):
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass
        raise

    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()


# --------------------------------------------------
# 3. Main Dashboard & System Info
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/system-info", methods=["GET"])
def system_info():
    info = engine.get_system_info()
    info["video_modes"] = VIDEO_MODES
    return jsonify(info)


# --------------------------------------------------
# 4. Image Analysis API
# --------------------------------------------------

@app.route("/api/analyze", methods=["POST"])
def analyze_image():
    start_time = perf_counter()

    try:
        if "file" not in request.files:
            return jsonify({"error": "No image uploaded."}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected."}), 400

        if not allowed_image_file(file.filename):
            return jsonify({"error": "Only JPG, JPEG and PNG images are supported currently."}), 400

        model_name = request.form.get("model", "unet").lower()
        use_yolo = request.form.get("use_yolo", "true").lower() == "true"

        if model_name not in {"unet", "deeplab"}:
            return jsonify({"error": "Invalid segmentation model."}), 400

        extension = secure_filename(file.filename).rsplit(".", 1)[1].lower()
        unique_name = f"{uuid4().hex}.{extension}"
        upload_path = UPLOAD_DIR / unique_name
        file.save(upload_path)

        original = cv2.imread(str(upload_path))
        if original is None:
            return jsonify({"error": "Unable to read uploaded image."}), 400

        original_height, original_width = original.shape[:2]

        dev_info = engine.get_active_device_info(model_name, use_yolo)
        mask_512 = engine.predict_mask_batch(model_name, [original])[0]

        mask = cv2.resize(
            mask_512,
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST,
        )

        colored_mask_bgr = colorize_mask_to_bgr(mask_512, original_width, original_height)
        overlay = cv2.addWeighted(original, 0.5, colored_mask_bgr, 0.5, 0)

        segmentation_name = f"seg_{unique_name.rsplit('.', 1)[0]}.png"
        segmentation_path = RESULT_DIR / segmentation_name
        cv2.imwrite(str(segmentation_path), overlay)

        pixel_counts = np.bincount(mask.flatten(), minlength=len(CLASS_NAMES))
        class_distribution = _compute_class_distribution(pixel_counts)

        object_counts = {}
        detection_url = None
        total_objects = 0

        if use_yolo:
            raw_results, _ = engine.predict_yolo_batch([original])
            result = raw_results[0]

            for _, name in result.names.items():
                object_counts[name] = 0

            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    class_name = result.names[class_id]
                    object_counts[class_name] = object_counts.get(class_name, 0) + 1

            total_objects = sum(object_counts.values())

            detection_name = f"det_{unique_name.rsplit('.', 1)[0]}.jpg"
            detection_path = RESULT_DIR / detection_name
            cv2.imwrite(str(detection_path), result.plot())
            detection_url = url_for("static", filename=f"results/{detection_name}")

        processing_time = round(perf_counter() - start_time, 3)

        return jsonify({
            "success": True,
            "model": model_name,
            "device": dev_info["device"],
            "device_name": dev_info["device_name"],
            "seg_backend": dev_info["seg_backend"],
            "original": url_for("static", filename=f"uploads/{unique_name}"),
            "segmentation": url_for("static", filename=f"results/{segmentation_name}"),
            "detection": detection_url,
            "total_objects": total_objects,
            "object_counts": object_counts,
            "class_distribution": class_distribution,
            "processing_time": processing_time,
            "image_width": original_width,
            "image_height": original_height,
        })

    except Exception as error:
        app.logger.exception("Image analysis failed")
        return jsonify({"error": str(error)}), 500


# --------------------------------------------------
# 5. Video Analysis APIs (Sync + Live Progress/Cancel)
# --------------------------------------------------

def _validate_video_request():
    if "file" not in request.files:
        return None, (jsonify({"error": "No video uploaded."}), 400)

    file = request.files["file"]
    if file.filename == "":
        return None, (jsonify({"error": "No video selected."}), 400)

    if not allowed_video_file(file.filename):
        return None, (jsonify({"error": "Only MP4, AVI and MOV videos are supported."}), 400)

    model_name = request.form.get("model", "unet").lower()
    use_yolo = request.form.get("use_yolo", "true").lower() == "true"
    mode = request.form.get("mode", "balanced").lower()

    if model_name not in {"unet", "deeplab"}:
        return None, (jsonify({"error": "Invalid segmentation model."}), 400)

    if mode not in VIDEO_MODES:
        return None, (jsonify({"error": "Invalid video processing mode."}), 400)

    extension = secure_filename(file.filename).rsplit(".", 1)[1].lower()
    unique_id = uuid4().hex
    input_name = f"{unique_id}.{extension}"
    output_name = f"processed_{unique_id}.mp4"
    input_path = UPLOAD_DIR / input_name
    output_path = RESULT_DIR / output_name
    file.save(input_path)

    return {
        "input_path": input_path,
        "output_path": output_path,
        "output_name": output_name,
        "model_name": model_name,
        "use_yolo": use_yolo,
        "mode": mode,
    }, None


@app.route("/api/analyze-video", methods=["POST"])
def analyze_video():
    params, err_resp = _validate_video_request()
    if err_resp is not None:
        return err_resp

    try:
        result = process_video_pipeline(
            input_path=params["input_path"],
            output_path=params["output_path"],
            output_filename=params["output_name"],
            model_name=params["model_name"],
            use_yolo=params["use_yolo"],
            mode=params["mode"],
        )
        return jsonify(result)
    except ValueError as err:
        return jsonify({"error": str(err)}), 400
    except Exception as error:
        app.logger.exception("Video analysis failed")
        return jsonify({"error": str(error)}), 500


@app.route("/api/analyze-video/start", methods=["POST"])
def start_video_analysis():
    params, err_resp = _validate_video_request()
    if err_resp is not None:
        return err_resp

    job_id = uuid4().hex
    cancel_event = Event()
    sys_info = engine.get_system_info()
    mode_info = VIDEO_MODES[params["mode"]]

    with VIDEO_JOBS_LOCK:
        _prune_old_jobs()
        VIDEO_JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "model": params["model_name"],
            "mode": params["mode"],
            "mode_label": mode_info["label"],
            "stride": mode_info["stride"],
            "device": sys_info["device"],
            "device_name": sys_info["device_name"],
            "total_frames": 0,
            "processed_frames": 0,
            "analyzed_frames": 0,
            "reused_frames": 0,
            "percent": 0.0,
            "current_fps": 0.0,
            "elapsed_seconds": 0.0,
            "eta_seconds": None,
            "error": None,
            "result": None,
            "_cancel_event": cancel_event,
        }

    def _worker():
        try:
            result = process_video_pipeline(
                input_path=params["input_path"],
                output_path=params["output_path"],
                output_filename=params["output_name"],
                model_name=params["model_name"],
                use_yolo=params["use_yolo"],
                mode=params["mode"],
                cancel_event=cancel_event,
                job_id=job_id,
            )
            _update_job(
                job_id,
                status="completed",
                percent=100.0,
                eta_seconds=0.0,
                elapsed_seconds=result["processing_time"],
                current_fps=result["processing_fps"],
                processed_frames=result["frames_processed"],
                analyzed_frames=result["analyzed_frames"],
                reused_frames=result["reused_frames"],
                total_frames=result["frames_processed"],
                result=result,
            )
        except VideoCancelledError:
            _update_job(job_id, status="cancelled", eta_seconds=None)
        except Exception as err:
            app.logger.exception("Async video analysis failed")
            _update_job(job_id, status="error", error=str(err), eta_seconds=None)

    Thread(target=_worker, daemon=True).start()
    return jsonify({"job_id": job_id, "status": "queued"}), 202


@app.route("/api/video-status/<job_id>", methods=["GET"])
def video_status(job_id: str):
    with VIDEO_JOBS_LOCK:
        job = VIDEO_JOBS.get(job_id)
        if job is None:
            return jsonify({"error": "Job not found."}), 404
        payload = {k: v for k, v in job.items() if not k.startswith("_")}
        return jsonify(payload)


@app.route("/api/video-cancel/<job_id>", methods=["POST"])
def cancel_video_analysis(job_id: str):
    with VIDEO_JOBS_LOCK:
        job = VIDEO_JOBS.get(job_id)
        if job is None:
            return jsonify({"error": "Job not found."}), 404
        cancel_event: Event = job["_cancel_event"]
        cancel_event.set()
        if job["status"] in {"queued", "running"}:
            job["status"] = "cancelling"
        return jsonify({"success": True, "job_id": job_id, "status": job["status"]})


# --------------------------------------------------
# 6. Run Flask
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        debug=True,
        use_reloader=False,
    )
