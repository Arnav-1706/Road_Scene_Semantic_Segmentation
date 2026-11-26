# Road Scene Analysis

A web application for road scene semantic segmentation and object detection across dashcam images and videos. It pairs 19-class Cityscapes semantic segmentation (**U-Net** and **DeepLabV3+**) with **YOLOv8n** object detection behind a Flask backend and a minimal coffee-and-cream web interface.

---

## 1. Project Structure

```text
Road_Scene_Semantic_Segmentation-main/
├── app.py                     # Flask backend, model loading, and /api/analyze endpoints
├── requirements.txt           # Python dependencies
├── README.md                  # Functionalities, usage, and deployment guide
├── models/
│   ├── best_unet_model.keras          # Keras U-Net segmentation model (512x1024 -> 19 classes)
│   ├── best_deeplabv3plus_model.keras # Keras DeepLabV3+ checkpoint (reference)
│   ├── deeplabv3plus_savedmodel/      # TensorFlow SavedModel used by app.py ("serve" signature)
│   └── yolov8n.pt                     # Ultralytics YOLOv8n object detection weights
├── templates/
│   └── index.html             # Single-page web UI
└── static/
    ├── css/style.css          # Coffee & cream responsive stylesheet
    ├── js/script.js           # Upload, API calls, video handling, and Chart.js rendering
    ├── uploads/               # Saved user uploads (created automatically)
    └── results/               # Generated segmentation, detection, and video outputs
```

---

## 2. Core Functionalities

### Semantic Segmentation (19 Cityscapes Classes)
- **Models**: Choose between **U-Net** (`unet`) and **DeepLabV3+** (`deeplab`).
- **Preprocessing**: Input frames are resized to `512 × 1024` (`INPUT_HEIGHT = 512`, `INPUT_WIDTH = 1024`), converted from BGR to RGB, and normalized via `tf.keras.applications.mobilenet_v2.preprocess_input`.
- **Classes Predicted**: `Road`, `Sidewalk`, `Building`, `Wall`, `Fence`, `Pole`, `Traffic Light`, `Traffic Sign`, `Vegetation`, `Terrain`, `Sky`, `Person`, `Rider`, `Car`, `Truck`, `Bus`, `Train`, `Motorcycle`, `Bicycle`.
- **Outputs**: Resizes the predicted mask back to the original input dimensions (`INTER_NEAREST`), applies the Cityscapes color palette, and blends it over the original image/frame (`0.5/0.5` for images; `0.55/0.45` for videos).

### Object Detection (YOLOv8n)
- Optional toggle (`Include YOLOv8 bounding boxes`, enabled by default) running `yolov8n.pt` at `conf=0.25` and `iou=0.45`.
- Produces annotated bounding-box images for single images and blends detections (`0.65/0.35`) onto segmented frames for videos.
- Aggregates per-class object counts (`car`, `person`, `truck`, `bus`, `traffic light`, etc.).

### Web Frontend (`templates/index.html`, `static/css/style.css`, `static/js/script.js`)
- **Coffee & Cream Theme**: Clean, distraction-free layout using dark espresso (`#231710`), roasted coffee (`#5c3a21`), caramel (`#9c6b43`), and warm cream surfaces (`#f4ede3` / `#fcf9f4`).
- **Media Upload**: Supports drag-and-drop and local file selection for images (`JPG`, `JPEG`, `PNG`) and videos (`MP4`, `AVI`, `MOV`) up to `500 MB`.
- **Visual Outputs**:
  - **Images**: Side-by-side Original, Segmentation, and Object Detection views.
  - **Videos**: Original vs. Processed Video player with a **Download MP4** link and an automatic fallback notice if the browser cannot play OpenCV's `mp4v` codec inline.
- **Scene Summary & Analytics**: Displays Total Objects / Frame Detections, Cars, Persons, Processing Time, detected class tags, and a **Chart.js** horizontal bar chart colored with the exact Cityscapes RGB palette.

---

## 3. API Endpoints

| Endpoint | Method | Form Fields | Description |
| :--- | :--- | :--- | :--- |
| `/` | `GET` | — | Renders the web UI (`templates/index.html`). |
| `/api/analyze` | `POST` | `file` (`.jpg/.jpeg/.png`), `model` (`unet` \| `deeplab`), `use_yolo` (`true` \| `false`) | Returns `original`, `segmentation`, `detection` URLs, `total_objects`, `object_counts`, `class_distribution` (`class`, `pixels`, `percentage`), `processing_time`, `image_width`, `image_height`. |
| `/api/analyze-video` | `POST` | `file` (`.mp4/.avi/.mov`), `model` (`unet` \| `deeplab`), `use_yolo` (`true` \| `false`) | Returns `video` URL, `frames_processed`, `original_fps`, `duration_seconds`, `processing_time`, `total_detections`, and `detection_counts`. |

---

## 4. Quick Start (Local Usage)

### Prerequisites
- Python **3.11** (64-bit)
- Dependencies listed in `requirements.txt` (`flask`, `tensorflow`, `ultralytics`, `opencv-python`, `numpy`, `torch`, `torchvision`, `werkzeug`)

### Installation
```powershell
pip install -r requirements.txt
```
*(Optional)* If using an isolated virtual environment:
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Run the Application
```powershell
python app.py
```
Wait for U-Net, DeepLabV3+, and YOLOv8n to finish loading (~20–30 seconds on startup), then open:
```text
http://127.0.0.1:5000
```

### Using the Interface
1. Select **U-Net** or **DeepLabV3+** from the **Model** dropdown.
2. Keep **Include YOLOv8 bounding boxes** checked to run object detection alongside segmentation (or uncheck for segmentation only).
3. Drag and drop an image or video into the upload box (or click **Choose File**).
4. Click **Analyze Scene** and review the visual outputs, summary cards, and semantic class coverage chart below.

---

## 5. Deployment Guide

`app.py` loads all three deep learning models once at module startup and uses Flask's development server when invoked directly (`app.run(debug=True, use_reloader=False)`). For production or shared network deployment, use a production WSGI server with **a single worker process** (to avoid loading multiple copies of TensorFlow and YOLO into memory) and multiple threads/long request timeouts for video jobs.

### Option A: Windows Server (`waitress`)
```powershell
pip install waitress
waitress-serve --host=0.0.0.0 --port=5000 --threads=4 app:app
```

### Option B: Linux / Container (`gunicorn`)
```bash
pip install gunicorn
gunicorn -w 1 --threads 4 -b 0.0.0.0:5000 --timeout 600 app:app
```
- `-w 1`: Prevents duplicating U-Net, DeepLabV3+, and YOLOv8n memory across multiple worker processes.
- `--timeout 600`: Prevents worker timeouts during frame-by-frame video inference.

### Reverse Proxy (Nginx) Notes
If placing Nginx in front of Waitress/Gunicorn, align upload and timeout limits with `app.py` (`MAX_CONTENT_LENGTH = 500 * 1024 * 1024`):
```nginx
client_max_body_size 500M;
proxy_read_timeout 600s;
proxy_send_timeout 600s;
```

### Operational Considerations
- **Hardware Acceleration**: Native Windows TensorFlow `>= 2.11` runs on CPU. For GPU-accelerated inference, deploy on Linux or Windows WSL2 with CUDA/cuDNN configured for TensorFlow and PyTorch.
- **Video Codec Playback**: `app.py` writes processed videos with OpenCV's `mp4v` FourCC codec. Some browsers cannot stream `mp4v` inline; the UI exposes a **Download MP4** button for local playback (e.g., VLC).
- **Disk Cleanup**: Processed inputs and outputs are stored in `static/uploads/` and `static/results/`. Set up a scheduled cleanup task in production to purge old files.
