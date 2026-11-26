# Road Scene Understanding — Performance & Accuracy Benchmarks

**Test Hardware**: Windows 11 · Python 3.11.9 · **NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM, Driver 592.82, CUDA 13.1)**  
**Test Media**: Deterministic `1280 × 720` @ 30 FPS road scene video (`90` frames = `3.0 s`; `5`-frame subset for slow TF-CPU U-Net baseline) and `bus.jpg` (`YOLOv8n` validation).  
**Model Input Resolution**: `512 × 1024` (19 Cityscapes classes, unaltered).

---

## 1. Before vs. After End-to-End Video Comparison

> **Speedup note**: Because running 90 frames of unoptimized U-Net on TensorFlow CPU takes ~57 minutes (`38.02 s/frame` end-to-end on the 5-frame run = `190.09 s`), U-Net speedups are computed against the `90`-frame equivalent (`3,421.6 s`). DeepLabV3+ baseline was measured directly on all `90` frames (`85.73 s`).

| Stage | Model | Processing Mode | Device | Batch (Seg / YOLO) | Frames (Analyzed / Total) | Seg Time / Frame | Total Video Time | Processing FPS | Process RAM | System VRAM | Speedup vs Baseline | TF Mask Agreement |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Before)** | U-Net | High Quality (`stride=1`) | CPU (TF 2.21 + CPU Torch) | `1 / 1` | `5 / 5` | `30,529.0 ms` | `190.09 s` | `0.026` | `1,872.8 MB` | `1,629.8 MB` | `1.0×` | `100.0%` |
| **Baseline (Before)** | DeepLabV3+ | High Quality (`stride=1`) | CPU (TF 2.21 + CPU Torch) | `1 / 1` | `90 / 90` | `838.0 ms` | `85.73 s` | `1.05` | `1,864.3 MB` | `1,769.7 MB` | `1.0×` | `100.0%` |
| **Optimized (GPU)** | **U-Net** | High Quality (`stride=1`) | GPU (`RTX 4060`) | `1 / 8` | `90 / 90` | `41.88 ms` | `5.36 s` | `16.78` | `1,778.0 MB` | `2,357.5 MB` | **`637.8×`** | **`99.9992%`** |
| **Optimized (GPU)** | **U-Net** | **Balanced (`stride=3`, Default)** | GPU (`RTX 4060`) | `1 / 8` | `30 / 90` | `41.88 ms` | **`2.36 s`** | **`38.20`** | `1,792.5 MB` | `2,369.0 MB` | **`1,452.3×`** | **`99.9992%`** |
| **Optimized (GPU)** | **U-Net** | Fast (`stride=5`) | GPU (`RTX 4060`) | `1 / 8` | `18 / 90` | `41.88 ms` | **`1.73 s`** | **`52.11`** | `1,783.5 MB` | `2,370.2 MB` | **`1,981.0×`** | **`99.9992%`** |
| **Optimized (GPU)** | DeepLabV3+ | High Quality (`stride=1`) | GPU (`RTX 4060`) | `1 / 8` | `90 / 90` | `53.97 ms` | `6.98 s` | `12.89` | `1,809.6 MB` | `3,355.5 MB` | **`12.3×`** | **`99.9964%`** |
| **Optimized (GPU)** | DeepLabV3+ | Balanced (`stride=3`) | GPU (`RTX 4060`) | `1 / 8` | `30 / 90` | `53.97 ms` | `2.85 s` | `31.54` | `1,820.0 MB` | `3,354.8 MB` | **`30.0×`** | **`99.9964%`** |
| **Optimized (GPU)** | DeepLabV3+ | Fast (`stride=5`) | GPU (`RTX 4060`) | `1 / 8` | `18 / 90` | `53.97 ms` | `2.25 s` | `39.94` | `1,809.4 MB` | `3,326.9 MB` | **`38.0×`** | **`99.9964%`** |
| **CPU Fallback** | U-Net | Balanced (`stride=3`) | CPU (`ONNX CPU` + `CPU Torch`) | `1 / 4` | `30 / 90` | `551.1 ms` | `23.75 s` | `3.79` | `1,356.6 MB` | `0.0 MB` (unused) | **`144.0×`** | **`99.9989%`** |
| **CPU Fallback** | DeepLabV3+ | Balanced (`stride=3`) | CPU (`ONNX CPU` + `CPU Torch`) | `1 / 4` | `30 / 90` | `1,104.5 ms` | `26.39 s` | `3.41` | `3,222.1 MB` | `0.0 MB` (unused) | **`3.2×`** | **`99.9809%`** |

Raw machine-readable data is stored in [benchmark_results.csv](benchmark_results.csv).

---

## 2. Batch-Size Sweep on RTX 4060 (8 GB VRAM)

Pure warmed-up inference benchmarks across batch sizes `1, 2, 4, 8, 16`:

### Semantic Segmentation (ONNX Runtime CUDA, `use_tf32=0`, GPU ArgMax)

| Model | Batch Size | Latency / Frame | Throughput (FPS) | System VRAM (Delta over Idle) | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **U-Net** | **`1`** | **`25.92 ms`** | **`38.58 FPS`** | `2,312 MB` (`+528 MB`) | **Optimal (fastest + lowest VRAM)** |
| U-Net | `2` | `28.96 ms` | `34.53 FPS` | `2,824 MB` (`+1,040 MB`) | Slightly slower |
| U-Net | `4` | `32.46 ms` | `30.81 FPS` | `4,872 MB` (`+3,088 MB`) | High activation VRAM |
| U-Net | `8` | `34.76 ms` | `28.77 FPS` | `7,861 MB` (`+6,078 MB`) | Near 8 GB ceiling |
| U-Net | `16` | `1,046.95 ms` | `0.96 FPS` | `7,920 MB` (`+6,137 MB`) | Shared RAM spill (thrashing) |
| **DeepLabV3+** | **`1`** | **`45.35 ms`** | **`22.05 FPS`** | `2,959 MB` (`+1,414 MB`) | **Optimal (`kSameAsRequested`)** |
| DeepLabV3+ | `2` | `95.26 ms` | `10.50 FPS` | `7,995 MB` (`+6,450 MB`) | Hits 8 GB limit |
| DeepLabV3+ | `4` | `63.90 ms` | `15.65 FPS` | `8,002 MB` (`+6,456 MB`) | Shared RAM spill |

### YOLOv8n (`PyTorch 2.14.1+cu130` on `cuda:0`, FP32, `1280 × 720`)

| Batch Size | Latency / Frame | Throughput (FPS) | PyTorch Peak VRAM | Notes |
| :---: | :---: | :---: | :---: | :--- |
| `1` | `25.98 ms` | `38.49 FPS` | `47 MB` | Single-frame baseline on GPU |
| `2` | `15.76 ms` | `63.44 FPS` | `50 MB` | — |
| `4` | `8.50 ms` | `117.64 FPS` | `77 MB` | — |
| **`8`** | **`5.36 ms`** | **`186.64 FPS`** | **`138 MB`** | **Selected optimal batch (4.8× faster, tiny VRAM)** |
| `16` | `4.19 ms` | `238.91 FPS` | `226 MB` | Marginal gain; larger chunk buffering delay |

---

## 3. Why U-Net Became the Recommended Video Model

- **On unoptimized TF CPU (Before)**: DeepLabV3+ (`0.84 s/frame`) was **36× faster** than U-Net (`30.53 s/frame`) because `deeplabv3plus_savedmodel` ran as a frozen TF graph while `best_unet_model.keras` hit uncompiled per-layer dispatch overhead.
- **After ONNX + RTX 4060 GPU acceleration (and ONNX CPU fallback)**: U-Net runs in **`~42 ms/frame`** (vs **`~54 ms/frame`** for DeepLabV3+) and uses **~1 GB less VRAM**, finishing the 90-frame Balanced video in **`2.36 s` (`38.2 FPS`)** vs **`2.85 s` (`31.5 FPS`)** for DeepLabV3+. Therefore, the UI dynamically recommends **U-Net** on the active GPU/ONNX runtime (and would recommend DeepLabV3+ if falling back to raw TensorFlow CPU).
