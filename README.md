# Road Scene Analysis (GPU-Accelerated)

A local Windows web application for road scene semantic segmentation and object detection across dashcam images and videos. It pairs 19-class Cityscapes semantic segmentation (**U-Net** and **DeepLabV3+** via ONNX Runtime CUDA) with **YOLOv8n** object detection (PyTorch CUDA) behind a Flask backend and a coffee-and-cream web interface.

---

## 1. Project Structure

```text
Road_Scene_Semantic_Segmentation-main/
├── app.py                     # Flask backend, async video job manager, and API routes
├── inference_engine.py        # GPU-first lazy model loader (ONNX Runtime CUDA + PyTorch CUDA + CPU fallback)
├── requirements.txt           # Core Python dependencies
├── README.md                  # Functionalities and local usage guide
├── benchmarks/
│   ├── BENCHMARK_RESULTS.md   # Presentation-ready before/after & batch-size benchmark report
│   ├── benchmark_results.csv  # Raw CSV metrics for charts/slides
│   ├── baseline_metrics.json  # Stage 1 baseline measurements
│   └── *_pip_freeze.txt       # Recorded package environments
├── models/
│   ├── best_unet_model.keras          # Original U-Net model (preserved)
│   ├── best_deeplabv3plus_model.keras # Original DeepLabV3+ checkpoint (preserved)
│   ├── deeplabv3plus_savedmodel/      # Original DeepLabV3+ SavedModel (preserved)
│   ├── unet.onnx / unet_mask.onnx     # ONNX + GPU-ArgMax U-Net models
│   ├── deeplabv3plus.onnx / deeplabv3plus_mask.onnx # ONNX + GPU-ArgMax DeepLabV3+ models
│   └── yolov8n.pt                     # Ultralytics YOLOv8n weights
├── templates/
│   └── index.html             # Coffee & cream web interface
└── static/
    ├── css/style.css          # Coffee & cream stylesheet
    ├── js/script.js           # Upload, live progress polling, cancellation, and Chart.js rendering
    ├── uploads/               # Saved user uploads
    └── results/               # Generated segmentation, detection, and video outputs
```

---

## 2. How to Run Locally (RTX 4060 GPU Environment)

Use the project's GPU-enabled `.venv` Python environment:

```powershell
.\.venv\Scripts\python.exe app.py
```

Then open:

```text
http://127.0.0.1:5000
```

> **Note**: Because TensorFlow `>= 2.11` does not support GPU on native Windows, segmentation runs through **ONNX Runtime (`CUDAExecutionProvider` with CUDA 13 / cuDNN 9)** and YOLOv8n runs through **PyTorch `2.14.1+cu130`**, automatically falling back to CPU (`ROADAI_FORCE_CPU=1` or when CUDA is unavailable).

---

## 3. Core Functionalities

### GPU-Accelerated Inference (`inference_engine.py`)
- **Lazy, Single-Model Loading**: Loads only the requested segmentation model (`unet` or `deeplab`) and releases the previous one on switch so VRAM is never duplicated. Loads YOLOv8n only when enabled.
- **GPU ArgMax Optimization**: `unet_mask.onnx` and `deeplabv3plus_mask.onnx` compute `ArgMax` on the GPU and transfer `uint8` masks (`0.5 MB/frame`) instead of 19-channel `float32` logits (`39.8 MB/frame`), preserving **99.996%–99.999%** pixel agreement with the original TensorFlow models.
- **Batched YOLOv8n + Fast Post-Processing**: Batches YOLO inference (`batch=8` on GPU) and uses a vectorized MobileNetV2 normalization LUT, precomputed BGR palette, and direct OpenCV box rendering.

### Video Processing Modes
- **High Quality (`stride=1`)**: Analyzes every frame.
- **Balanced (`stride=3`, Default)**: Analyzes every 3rd frame and blends the latest segmentation mask and bounding boxes over intermediate frames while preserving original video FPS and duration.
- **Fast (`stride=5`)**: Analyzes every 5th frame for maximum throughput.
- **Honest Statistics**: Object detections and class percentages are computed **strictly from analyzed frames** (never inflated by frame reuse), and each output frame is labeled `Analyzed` or `Reused mask`.

### Live UX & Cancellation
- Displays active hardware (`GPU` vs `CPU Fallback`) and the benchmark-verified recommended model (`U-Net` on GPU/ONNX).
- Live video progress bar showing percentage completed, `processed / total` and `analyzed` frame counts, processing FPS, elapsed time, and ETA.
- **Cancel Processing** button aborts mid-run, releases video handles, and cleans up partial files.

---

## 4. API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | Serves the web UI (`templates/index.html`). |
| `/api/system-info` | `GET` | Returns active GPU/CPU status, recommended model, and video modes. |
| `/api/analyze` | `POST` | Runs semantic segmentation + optional YOLOv8n on a single image (`.jpg`, `.jpeg`, `.png`). |
| `/api/analyze-video` | `POST` | Synchronous video analysis with `mode` (`high` \| `balanced` \| `fast`). |
| `/api/analyze-video/start` | `POST` | Starts async video processing and returns `{ "job_id": "..." }`. |
| `/api/video-status/<job_id>` | `GET` | Polls live progress (`percent`, `eta_seconds`, `current_fps`, `analyzed_frames`, `result`). |
| `/api/video-cancel/<job_id>` | `POST` | Signals cooperative cancellation and cleans up resources. |
