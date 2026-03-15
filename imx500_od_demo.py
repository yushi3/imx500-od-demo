#!/usr/bin/env python3
"""
IMX500 Object Detection Demo
- Supports TF OD API compatible output format models
- Model selectable via --model argument
- Auto-detects BBOX coordinate normalization
- Displays RAW Output Tensors and parsed detection results
- Threshold can be changed in real-time from imx500_od_ctrl.py
"""

import cv2
import os
import time
import threading
import argparse
import numpy as np
from picamera2 import Picamera2, MappedArray
from picamera2.devices.imx500 import IMX500, NetworkIntrinsics

MODELS = {
    "ssd":          "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk",
    "efficientdet": "/usr/share/imx500-models/imx500_network_efficientdet_lite0_pp.rpk",
    "nanodet":      "/usr/share/imx500-models/imx500_network_nanodet_plus_416x416_pp.rpk",
}

DEFAULT_THRESHOLD = {
    "ssd":          0.50,
    "efficientdet": 0.50,
    "nanodet":      0.25,
}

PIPE_PATH      = "/tmp/imx500_ctrl"
DETECTION_ROWS = 10  # Fixed number of rows in detection display

# --- Arguments ---
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="ssd",
                    help=f"Model name ({', '.join(MODELS.keys())}) or direct path to RPK file")
parser.add_argument("--threshold", type=float, default=None,
                    help="Detection score threshold (default: model-specific value)")
args = parser.parse_args()

model_path = MODELS.get(args.model, args.model)
model_name = args.model
threshold  = args.threshold if args.threshold is not None else DEFAULT_THRESHOLD.get(model_name, 0.50)

# --- IMX500 initialization ---
imx500 = IMX500(model_path)
intrinsics = imx500.network_intrinsics or NetworkIntrinsics()
intrinsics.task = "object detection"
intrinsics.update_with_defaults()

picam2 = Picamera2(imx500.camera_num)
config = picam2.create_preview_configuration(
    controls={"FrameRate": 30}, buffer_count=12)

# --- Shared state for threshold ---
class State:
    def __init__(self, initial):
        self._threshold = initial
        self.lock = threading.Lock()

    def set(self, val):
        with self.lock:
            self._threshold = round(max(0.01, min(0.99, val)), 2)
        print(f"[CTRL] threshold = {self._threshold:.2f}")

    def get(self):
        with self.lock:
            return self._threshold

state = State(threshold)

# --- Named pipe listener thread ---
def pipe_listener():
    """
    Receives commands from /tmp/imx500_ctrl (named pipe) and updates threshold.
    Commands:
      inc      -> threshold + 0.01
      dec      -> threshold - 0.01
      set 0.65 -> set threshold to 0.65
    """
    if not os.path.exists(PIPE_PATH):
        os.mkfifo(PIPE_PATH)
    print(f"[CTRL] named pipe ready: {PIPE_PATH}")
    while True:
        try:
            with open(PIPE_PATH, 'r') as pipe:
                for line in pipe:
                    cmd = line.strip()
                    if cmd == "inc":
                        state.set(state.get() + 0.01)
                    elif cmd == "dec":
                        state.set(state.get() - 0.01)
                    elif cmd.startswith("set "):
                        try:
                            state.set(float(cmd.split()[1]))
                        except ValueError:
                            pass
        except Exception as e:
            print(f"[CTRL] pipe error: {e}")
            time.sleep(0.5)

threading.Thread(target=pipe_listener, daemon=True).start()

# --- Utilities ---
def get_label(cls: int) -> str:
    if intrinsics.labels and cls < len(intrinsics.labels):
        return intrinsics.labels[cls]
    return f"class_{cls}"

def normalize_boxes(boxes, np_outputs):
    """
    Normalize BBOX coordinates to 0~1 range.
    - max value <= 1.1 : already normalized (allow slight overshoot)
    - max value >  1.1 : absolute pixel values, divide by input size
    Input size is estimated from max ymax/xmax, rounded to nearest multiple of 32.
    """
    if len(boxes) == 0:
        return boxes, None
    all_boxes = np_outputs[0][0]
    if float(all_boxes.max()) <= 1.1:
        return boxes, None
    input_size = float(all_boxes[:, 2:].max())
    input_size = round(input_size / 32) * 32
    return boxes / input_size, input_size

def parse_ssd(np_outputs, thr):
    """
    TF OD API compatible format:
      tensor[0] boxes   (1, N, 4)  [ymin, xmin, ymax, xmax]
      tensor[1] scores  (1, N)     confidence scores, sorted descending
      tensor[2] classes (1, N)     class IDs
      tensor[3] max_det (1, 1)     max detections upper limit (unused)
    """
    boxes   = np_outputs[0][0]
    scores  = np_outputs[1][0]
    classes = np_outputs[2][0].astype(int)
    mask    = scores > thr
    boxes   = boxes[mask]
    scores  = scores[mask]
    classes = classes[mask]
    boxes, input_size = normalize_boxes(boxes, np_outputs)
    return boxes, scores, classes, input_size

def fmt_boxes(np_outputs, n_show=3):
    """Format tensor[0] as [[ymin,xmin,ymax,xmax], ...] for display."""
    raw   = np_outputs[0][0][:n_show]
    items = [f"[{','.join(f'{v:.4f}' for v in row)}]" for row in raw]
    return "[" + ", ".join(items) + ", ...]"

def draw_detections(request):
    metadata   = request.get_metadata()
    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    if np_outputs is None:
        return

    thr = state.get()

    # -- 1. Raw tensor display --
    print(f"\n{'='*60}")
    print(f"[RAW OUTPUT TENSORS from IMX500 @ {time.strftime('%H:%M:%S')}]")

    tensor_meta = [
        ("boxes  ", "BBOX coordinates [ymin,xmin,ymax,xmax] per candidate"),
        ("scores ", "Confidence score 0~1 per candidate"),
        ("classes", "Class ID (integer) per candidate"),
        ("max_det", "Max candidates upper limit (fixed value)"),
    ]

    for i, tensor in enumerate(np_outputs):
        flat = tensor.flatten()
        name, desc = tensor_meta[i] if i < len(tensor_meta) else (f"output{i}", "")
        print(f"  tensor[{i}] {name}  {desc}")

        if i == 0:
            print(f"             shape={tensor.shape}")
            print(f"             {fmt_boxes(np_outputs, n_show=3)}")
            is_norm = float(np_outputs[0][0].max()) <= 1.1
            print(f"             -> format: {'normalized (0~1)' if is_norm else 'absolute pixels (auto-normalize)'}")
        else:
            sample = " ".join(f"{v:.4f}" for v in flat[:12])
            print(f"             shape={tensor.shape}  [{sample} ...]")
            if i == 1:
                n_over = int((flat > thr).sum())
                print(f"             -> above threshold={thr:.2f}: {n_over} candidates")
            if i == 2:
                top_labels = [get_label(int(c)) for c in flat[:5]]
                print(f"             -> top 5 labels: {top_labels}")

    # -- 2. Parsed detection display (fixed height = DETECTION_ROWS lines) --
    boxes, scores, classes, input_size = parse_ssd(np_outputs, thr)

    lines = []
    if input_size:
        lines.append(f"  (coordinates normalized: {input_size:.0f}px -> 0~1)")
    if len(scores) == 0:
        lines.append("  (no detections)")
    else:
        lines.append(f"  count: {len(scores)}")
        for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
            lines.append(
                f"  [{i+1}] {get_label(cls):20s}  score={score:.3f}  "
                f"box=[ymin={box[0]:.3f} xmin={box[1]:.3f} "
                f"ymax={box[2]:.3f} xmax={box[3]:.3f}]"
            )

    # Pad or truncate to exactly DETECTION_ROWS lines to prevent layout shift
    lines = lines[:DETECTION_ROWS]
    while len(lines) < DETECTION_ROWS:
        lines.append("")

    print(f"[PARSED DETECTIONS]  (threshold={thr:.2f})")
    for line in lines:
        print(f"{line:<78}")
    print(f"{'='*60}")

    # -- 3. Draw BBOX on preview --
    with MappedArray(request, "main") as m:
        h, w = m.array.shape[:2]
        for box, score, cls in zip(boxes, scores, classes):
            ymin, xmin, ymax, xmax = box
            x0, y0 = int(xmin * w), int(ymin * h)
            x1, y1 = int(xmax * w), int(ymax * h)
            label = f"{get_label(cls)} {score:.0%}"
            cv2.rectangle(m.array, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(m.array, label, (x0, max(y0 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(m.array,
                    f"IMX500: On-Sensor AI ({model_name})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.putText(m.array,
                    f"threshold: {thr:.2f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

imx500.show_network_fw_progress_bar()
picam2.pre_callback = draw_detections
picam2.start(config, show_preview=True)

print(f"Demo started [model={model_name}  threshold={state.get():.2f}]")
print(f"Run in another terminal: python3 imx500_od_ctrl.py")
try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nStopped.")
    if os.path.exists(PIPE_PATH):
        os.remove(PIPE_PATH)
    picam2.stop()
