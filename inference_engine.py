import gc
import os
import threading
from pathlib import Path

import cv2
import numpy as np

# Load ONNX Runtime and preload bundled CUDA 13 / cuDNN 9 DLLs BEFORE importing torch
# so CUDAExecutionProvider binds cleanly to its expected runtime libraries on Windows.
import onnxruntime as ort

if hasattr(ort, "preload_dlls"):
    try:
        ort.preload_dlls()
    except Exception:
        pass

import torch
from ultralytics import YOLO


CLASS_NAMES = [
    "Road", "Sidewalk", "Building", "Wall", "Fence",
    "Pole", "Traffic Light", "Traffic Sign", "Vegetation",
    "Terrain", "Sky", "Person", "Rider", "Car",
    "Truck", "Bus", "Train", "Motorcycle", "Bicycle",
]

CITYSCAPES_COLORS_RGB = np.array([
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
], dtype=np.uint8)

# Precomputed BGR palette eliminates per-frame RGB->BGR conversion overhead
CITYSCAPES_COLORS_BGR = CITYSCAPES_COLORS_RGB[:, ::-1].copy()

INPUT_HEIGHT = 512
INPUT_WIDTH = 1024

# Exact float32 equivalent of tf.keras.applications.mobilenet_v2.preprocess_input: (x / 127.5) - 1.0
PREPROCESS_LUT = (np.arange(256, dtype=np.float32) / 127.5) - 1.0

# Deterministic Ultralytics-style palette for fast OpenCV bounding box drawing
_BOX_PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
    (147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
    (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255),
]


def preprocess_frame(bgr_image: np.ndarray) -> np.ndarray:
    """Resize BGR frame to 512x1024, convert to RGB, and apply MobileNetV2 normalization."""
    resized = cv2.resize(bgr_image, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return PREPROCESS_LUT[rgb]


def preprocess_batch(bgr_frames: list[np.ndarray]) -> np.ndarray:
    """Preprocess a list of BGR frames into a contiguous (N, 512, 1024, 3) float32 batch."""
    batch = np.empty((len(bgr_frames), INPUT_HEIGHT, INPUT_WIDTH, 3), dtype=np.float32)
    for idx, frame in enumerate(bgr_frames):
        batch[idx] = preprocess_frame(frame)
    return batch


def colorize_mask_to_bgr(mask_512x1024: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Colorize at model resolution (512x1024) first, then resize cleanly with INTER_NEAREST."""
    colored_small = CITYSCAPES_COLORS_BGR[mask_512x1024]
    if target_width == INPUT_WIDTH and target_height == INPUT_HEIGHT:
        return colored_small
    return cv2.resize(colored_small, (target_width, target_height), interpolation=cv2.INTER_NEAREST)


def annotate_frame_status(frame_bgr: np.ndarray, is_analyzed: bool, mode_label: str) -> np.ndarray:
    """Draw a compact, unobtrusive badge distinguishing analyzed frames from reused predictions."""
    tag = f"{mode_label} | {'Analyzed' if is_analyzed else 'Reused mask'}"
    (tw, th), baseline = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    pad_x, pad_y = 8, 5
    box_w = tw + pad_x * 2
    box_h = th + baseline + pad_y * 2
    x1, y1 = 10, 10
    roi = frame_bgr[y1:y1 + box_h, x1:x1 + box_w]
    if roi.size > 0:
        bg_color = np.array([34, 85, 34] if is_analyzed else [38, 64, 110], dtype=np.uint8)
        overlay = np.empty_like(roi)
        overlay[:] = bg_color
        cv2.addWeighted(overlay, 0.72, roi, 0.28, 0, dst=roi)
    text_color = (220, 255, 220) if is_analyzed else (235, 240, 255)
    cv2.putText(
        frame_bgr,
        tag,
        (x1 + pad_x, y1 + pad_y + th),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        text_color,
        1,
        cv2.LINE_AA,
    )
    return frame_bgr


def draw_detections(frame_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw crisp bounding boxes and class labels directly on a frame without full-frame copies."""
    if not detections:
        return frame_bgr

    out = frame_bgr
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cls_id = det["cls_id"]
        label = f"{det['label']} {det['conf']:.2f}"
        color = _BOX_PALETTE[cls_id % len(_BOX_PALETTE)]

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        top = max(y1 - th - baseline - 6, 0)
        cv2.rectangle(out, (x1, top), (x1 + tw + 8, top + th + baseline + 6), color, -1)
        cv2.putText(
            out,
            label,
            (x1 + 4, top + th + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


class InferenceManager:
    """
    Manages lazy, non-duplicated model loading across RTX 4060 GPU (ONNX Runtime CUDA + PyTorch CUDA)
    with explicit, verified CPU fallback.
    """

    def __init__(self, model_dir: Path, force_cpu: bool = False):
        self.model_dir = Path(model_dir)
        self.force_cpu = force_cpu or (os.environ.get("ROADAI_FORCE_CPU", "").lower() in {"1", "true", "yes"})
        self._lock = threading.Lock()
        self._infer_lock = threading.Lock()

        # Segmentation runtime state (only one segmentation model kept resident to save VRAM)
        self._active_seg_name: str | None = None
        self._seg_backend: str | None = None  # "onnx_cuda" | "onnx_cpu" | "tf_cpu"
        self._ort_session = None
        self._tf_unet = None
        self._tf_deeplab_infer = None

        # Object detection state
        self._yolo_model = None
        self._yolo_device = "cpu"

        # Hardware capability probe
        self.cuda_torch_available = (not self.force_cpu) and torch.cuda.is_available()
        self.gpu_name = torch.cuda.get_device_name(0) if self.cuda_torch_available else None
        self.ort_cuda_available = self._probe_ort_cuda()

        # Empirically selected optimal batch sizes for RTX 4060 (8 GB VRAM):
        # Segmentation batch=1 minimizes VRAM (528 MB U-Net / 1.4 GB DeepLab) & gives peak FPS;
        # YOLOv8n batch=8 boosts throughput 4.8x using only 138 MB VRAM.
        self.seg_batch_size = 1
        self.yolo_batch_size = 8 if self.cuda_torch_available else 4

    def _probe_ort_cuda(self) -> bool:
        if self.force_cpu:
            return False
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            return False
        # Verify an ONNX session actually initializes CUDAExecutionProvider without silent fallback
        probe_model = self.model_dir / "unet_mask.onnx"
        if not probe_model.exists():
            probe_model = self.model_dir / "deeplabv3plus_mask.onnx"
        if not probe_model.exists():
            return False
        try:
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            sess = ort.InferenceSession(
                str(probe_model),
                sess_options=opts,
                providers=[("CUDAExecutionProvider", {"use_tf32": "0", "arena_extend_strategy": "kSameAsRequested"})],
            )
            ok = sess.get_providers()[0] == "CUDAExecutionProvider"
            del sess
            gc.collect()
            return ok
        except Exception as err:
            print(f"[InferenceManager] CUDAExecutionProvider probe failed, using CPU fallback: {err}")
            return False

    def get_system_info(self) -> dict:
        gpu_active = self.ort_cuda_available and self.cuda_torch_available
        if self.ort_cuda_available:
            # Measured on RTX 4060 (incl. preprocessing): U-Net ~41.9 ms/frame vs DeepLabV3+ ~54.0 ms/frame
            recommended_model = "unet"
            recommended_reason = "Fastest on RTX 4060 GPU (~42 ms/frame vs ~54 ms/frame for DeepLabV3+) with ~1 GB lower VRAM"
        else:
            # Measured on ONNX CPU: U-Net ~0.55 s/frame vs DeepLabV3+ ~1.10 s/frame
            has_onnx = (self.model_dir / "unet_mask.onnx").exists()
            recommended_model = "unet" if has_onnx else "deeplab"
            recommended_reason = (
                "Fastest on ONNX CPU (~0.55 s/frame vs ~1.10 s/frame)"
                if has_onnx
                else "Fastest on TensorFlow CPU (~0.84 s/frame vs ~30.5 s/frame)"
            )

        return {
            "device": "GPU" if (self.ort_cuda_available or self.cuda_torch_available) else "CPU",
            "device_name": self.gpu_name if gpu_active else ("CPU Fallback (CUDA partially unavailable)" if self.ort_cuda_available or self.cuda_torch_available else "CPU Fallback"),
            "seg_provider": "CUDAExecutionProvider (RTX 4060)" if self.ort_cuda_available else "CPUExecutionProvider",
            "yolo_device": f"cuda:0 ({self.gpu_name})" if self.cuda_torch_available else "cpu",
            "recommended_video_model": recommended_model,
            "recommended_reason": recommended_reason,
            "seg_batch_size": self.seg_batch_size,
            "yolo_batch_size": self.yolo_batch_size,
        }

    def _unload_segmentation(self):
        self._ort_session = None
        self._tf_unet = None
        self._tf_deeplab_infer = None
        self._active_seg_name = None
        self._seg_backend = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _ensure_segmentation_loaded(self, model_name: str):
        with self._lock:
            if self._active_seg_name == model_name and (
                self._ort_session is not None or self._tf_unet is not None or self._tf_deeplab_infer is not None
            ):
                return

            # Release any previously loaded segmentation model so VRAM is never duplicated
            if self._active_seg_name is not None:
                self._unload_segmentation()

            onnx_filename = "unet_mask.onnx" if model_name == "unet" else "deeplabv3plus_mask.onnx"
            onnx_path = self.model_dir / onnx_filename

            if onnx_path.exists():
                opts = ort.SessionOptions()
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

                if self.ort_cuda_available:
                    try:
                        sess = ort.InferenceSession(
                            str(onnx_path),
                            sess_options=opts,
                            providers=[("CUDAExecutionProvider", {"use_tf32": "0", "arena_extend_strategy": "kSameAsRequested"})],
                        )
                        if sess.get_providers()[0] == "CUDAExecutionProvider":
                            self._ort_session = sess
                            self._seg_backend = "onnx_cuda"
                            self._active_seg_name = model_name
                            print(f"[InferenceManager] Loaded {onnx_filename} on CUDAExecutionProvider (RTX 4060)")
                            return
                    except Exception as err:
                        print(f"[InferenceManager] Failed CUDAExecutionProvider for {onnx_filename}, falling back to CPU: {err}")
                        self.ort_cuda_available = False

                # Explicit CPU fallback via ONNX Runtime
                sess = ort.InferenceSession(
                    str(onnx_path),
                    sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
                self._ort_session = sess
                self._seg_backend = "onnx_cpu"
                self._active_seg_name = model_name
                print(f"[InferenceManager] Loaded {onnx_filename} on CPUExecutionProvider")
                return

            # Last-resort fallback to original TensorFlow checkpoints if ONNX files are absent
            import tensorflow as tf
            if model_name == "unet":
                self._tf_unet = tf.keras.models.load_model(self.model_dir / "best_unet_model.keras", compile=False)
                self._seg_backend = "tf_cpu"
            else:
                loaded = tf.saved_model.load(str(self.model_dir / "deeplabv3plus_savedmodel"))
                self._tf_deeplab_infer = loaded.signatures["serve"]
                self._seg_backend = "tf_cpu"
            self._active_seg_name = model_name
            print(f"[InferenceManager] Loaded TensorFlow fallback for {model_name}")

    def _ensure_yolo_loaded(self):
        with self._lock:
            if self._yolo_model is not None:
                return
            yolo = YOLO(str(self.model_dir / "yolov8n.pt"))
            if self.cuda_torch_available:
                try:
                    yolo.to("cuda:0")
                    self._yolo_device = 0
                    print(f"[InferenceManager] Loaded YOLOv8n on GPU ({self.gpu_name})")
                except Exception as err:
                    print(f"[InferenceManager] YOLO GPU move failed, using CPU: {err}")
                    self._yolo_device = "cpu"
            else:
                self._yolo_device = "cpu"
                print("[InferenceManager] Loaded YOLOv8n on CPU")
            self._yolo_model = yolo

    def get_active_device_info(self, model_name: str, use_yolo: bool) -> dict:
        self._ensure_segmentation_loaded(model_name)
        if use_yolo:
            self._ensure_yolo_loaded()

        seg_on_gpu = self._seg_backend == "onnx_cuda"
        yolo_on_gpu = (not use_yolo) or (self._yolo_device == 0)
        if seg_on_gpu and yolo_on_gpu:
            device_label = "GPU"
        elif seg_on_gpu or (use_yolo and self._yolo_device == 0):
            device_label = "GPU/CPU"
        else:
            device_label = "CPU"

        return {
            "device": device_label,
            "device_name": self.gpu_name if seg_on_gpu else "CPU",
            "seg_backend": self._seg_backend,
            "yolo_device": f"cuda:0" if self._yolo_device == 0 else "cpu",
        }

    def predict_mask_batch(self, model_name: str, bgr_frames: list[np.ndarray], batch_size: int | None = None) -> np.ndarray:
        """
        Run batched semantic segmentation in memory-safe sub-batches and return (N, 512, 1024) uint8 masks.
        """
        if not bgr_frames:
            return np.empty((0, INPUT_HEIGHT, INPUT_WIDTH), dtype=np.uint8)

        chunk_size = max(1, batch_size or self.seg_batch_size)
        outputs = []

        with self._infer_lock:
            self._ensure_segmentation_loaded(model_name)
            for start in range(0, len(bgr_frames), chunk_size):
                sub_frames = bgr_frames[start:start + chunk_size]
                batch_input = preprocess_batch(sub_frames)
                if self._ort_session is not None:
                    masks = self._ort_session.run(["mask"], {"input": batch_input})[0]
                    outputs.append(masks.astype(np.uint8, copy=False))
                else:
                    import tensorflow as tf
                    if model_name == "unet":
                        probs = self._tf_unet(batch_input, training=False).numpy()
                    else:
                        probs = self._tf_deeplab_infer(input_layer_1=tf.convert_to_tensor(batch_input))["output_0"].numpy()
                    outputs.append(np.argmax(probs, axis=-1).astype(np.uint8))

        return outputs[0] if len(outputs) == 1 else np.concatenate(outputs, axis=0)

    def predict_yolo_batch(self, bgr_frames: list[np.ndarray], batch_size: int | None = None):
        """
        Run batched YOLOv8n inference in memory-safe chunks and return (raw_results, parsed_detections_per_frame).
        """
        if not bgr_frames:
            return [], []

        chunk_size = max(1, batch_size or self.yolo_batch_size)
        all_results = []
        parsed_all = []

        with self._infer_lock:
            self._ensure_yolo_loaded()
            for start in range(0, len(bgr_frames), chunk_size):
                sub_frames = bgr_frames[start:start + chunk_size]
                results = self._yolo_model.predict(
                    source=sub_frames,
                    device=self._yolo_device,
                    conf=0.25,
                    iou=0.45,
                    verbose=False,
                )
                for res in results:
                    all_results.append(res)
                    frame_dets = []
                    if res.boxes is not None and len(res.boxes) > 0:
                        xyxy = res.boxes.xyxy.cpu().numpy().astype(np.int32)
                        confs = res.boxes.conf.cpu().numpy()
                        clss = res.boxes.cls.cpu().numpy().astype(np.int32)
                        for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, clss):
                            frame_dets.append({
                                "box": (int(x1), int(y1), int(x2), int(y2)),
                                "conf": float(conf),
                                "cls_id": int(cls_id),
                                "label": res.names[int(cls_id)],
                            })
                    parsed_all.append(frame_dets)

        return all_results, parsed_all
