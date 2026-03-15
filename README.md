# imx500-od-demo

Real-time object detection demo for Raspberry Pi AI Camera (IMX500) — visualizes on-sensor inference tensors and bounding boxes.

---

## Overview

The **Raspberry Pi AI Camera** features Sony's **IMX500** intelligent vision sensor, which integrates a DNN accelerator directly on the camera chip. Unlike conventional AI pipelines, **inference runs entirely inside the camera module** — the Raspberry Pi CPU receives only the inference results (output tensors), not raw image data for processing.

This demo visualizes that process in real time:

- **Terminal**: raw output tensors from IMX500, followed by parsed detection results
- **Preview window**: live camera feed with bounding boxes and labels overlay

```
┌─────────────────────────────┐
│        IMX500 (on-chip)     │
│  Sensor → ISP → DNN         │
│              ↓              │
│       Output Tensors        │
└──────────────┬──────────────┘
               │ MIPI CSI-2 (Virtual Channel)
               ↓
┌─────────────────────────────┐
│       Raspberry Pi          │
│  libcamera → Picamera2      │
│  → parse → draw BBOX        │
└─────────────────────────────┘
```

---

## Hardware Requirements

- Raspberry Pi 4 or 5
- [Raspberry Pi AI Camera](https://www.raspberrypi.com/products/ai-camera/) (IMX500)

---

## Software Requirements

```bash
sudo apt update && sudo apt full-upgrade
sudo apt install -y imx500-all python3-picamera2 python3-opencv
sudo reboot
```

---

## Installation

```bash
git clone https://github.com/<your-username>/imx500-od-demo.git
cd imx500-od-demo
```

---

## Usage

```bash
# SSD MobileNetV2 (default)
python3 imx500_od_demo.py

# EfficientDet Lite0
python3 imx500_od_demo.py --model efficientdet

# NanoDet Plus
python3 imx500_od_demo.py --model nanodet

# Specify threshold manually
python3 imx500_od_demo.py --model ssd --threshold 0.6

# Use a custom RPK model file directly
python3 imx500_od_demo.py --model /path/to/your_model.rpk
```

### Keyboard Controls (while running)

| Key | Action |
|-----|--------|
| `+` / `=` | Increase threshold by 0.05 |
| `-` / `_` | Decrease threshold by 0.05 |
| `1` – `9` | Set threshold directly to 0.1 – 0.9 |
| `q` | Quit |

---

## Supported Models

| Name | Model file | mAP (COCO) | Notes |
|------|-----------|------------|-------|
| `ssd` | `imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk` | ~22% | Fast, default |
| `efficientdet` | `imx500_network_efficientdet_lite0_pp.rpk` | ~26% | Balanced |
| `nanodet` | `imx500_network_nanodet_plus_416x416_pp.rpk` | ~34% | Most accurate, anchor-free |

All models use **TF OD API-compatible output format**:

```
tensor[0] boxes    (1, N, 4)  [ymin, xmin, ymax, xmax]
tensor[1] scores   (1, N)     confidence scores
tensor[2] classes  (1, N)     class IDs
tensor[3] max_det  (1, 1)     max detections (fixed, model-dependent)
```

---

## Terminal Output Example

```
============================================================
[RAW OUTPUT TENSORS from IMX500 @ 14:32:05]
  tensor[0] boxes    各候補のBBOX座標 [ymin,xmin,ymax,xmax]
             shape=(1, 100, 4)
             [[0.0000,0.0690,1.0000,0.9141], [0.0156,0.8125,0.3438,1.0000], [0.0078,0.0078,0.9922,0.8828], ...]
             → 座標形式: 0~1正規化済み
  tensor[1] scores   各候補の信頼スコア 0~1
             shape=(1, 100)  [0.7773 0.5000 0.4375 0.3789 ...]
             → threshold=0.50 以上: 1個
  tensor[2] classes  各候補のクラスID (整数)
             shape=(1, 100)  [0.0000 83.0000 31.0000 72.0000 ...]
             → 上位5候補のラベル: ['person', 'book', 'tie', 'laptop', 'book']
  tensor[3] max_det  このモデルの候補数上限 (固定値)
             shape=(1, 1)    [100.0000 ...]
[PARSED DETECTIONS]  (threshold=0.50  +/-/1-9キーで変更)
  検出数: 1
  [1] person                score=0.777  box=[ymin=0.000 xmin=0.069 ymax=1.000 xmax=0.911]
============================================================
```

---

## How It Works

The IMX500 transmits both the **camera image** and **output tensors** simultaneously over a single MIPI CSI-2 cable, using separate virtual channels:

| MIPI Virtual Channel | Content |
|---------------------|---------|
| Ch.0 | Video frame (YUV/RGB) |
| Ch.1 | Output tensors (inference results) |

`libcamera` receives both streams and exposes the tensors as frame metadata. `Picamera2`'s `IMX500` class extracts them via `get_outputs(metadata)`. No CPU inference is involved.

---

## License

MIT
