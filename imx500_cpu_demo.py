#!/usr/bin/env python3
"""
IMX500 CPU Inference Demo
- Uses IMX500 as a plain camera (DNN disabled)
- Object detection runs on Raspberry Pi CPU via OpenCV DNN
- Comparison target for imx500_od_demo.py
"""

import cv2
import os
import time
import threading
import argparse
import numpy as np
import psutil
from picamera2 import Picamera2, MappedArray
from picamera2.devices.imx500 import IMX500

COCO_LABELS = [
    "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "-", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "-", "backpack", "umbrella", "-",
    "-", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "-", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "-", "dining table", "-", "-", "toilet",
    "-", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "-", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

MODEL_PB    = "frozen_inference_graph.pb"
MODEL_PBTXT = "ssd_mobilenet_v2.pbtxt"
PIPE_PATH   = "/tmp/imx500_ctrl"

# --- Arguments ---
parser = argparse.ArgumentParser()
parser.add_argument("--threshold", type=float, default=None,
                    help="Detection score threshold (default: 0.50)")
args = parser.parse_args()
threshold = args.threshold if args.threshold is not None else 0.50

# --- Check model files ---
for f in (MODEL_PB, MODEL_PBTXT):
    if not os.path.exists(f):
        print(f"Error: {f} not found. Please download the model files first.")
        exit(1)

# --- Load OpenCV DNN model (CPU) ---
net = cv2.dnn.readNetFromTensorflow(MODEL_PB, MODEL_PBTXT)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
print("OpenCV DNN model loaded.")

# --- Use IMX500 as plain camera (DNN disabled) ---
# IMX500() requires a network_file argument.
# "inputtensoronly" model passes input through without inference.
imx500 = IMX500("/usr/share/imx500-models/imx500_network_inputtensoronly.rpk")
picam2 = Picamera2(imx500.camera_num)
config = picam2.create_preview_configuration(
    main={"size": (640, 480)},
    controls={"FrameRate": 30}, buffer_count=12)

# --- Shared state ---
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

# --- CPU monitor ---
class CpuMonitor:
    def __init__(self):
        self._usage = 0.0
        self.lock   = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            val = psutil.cpu_percent(interval=1)
            with self.lock:
                self._usage = val

    def get(self):
        with self.lock:
            return self._usage

cpu_monitor = CpuMonitor()

# --- Named pipe listener ---
def pipe_listener():
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

# --- Inference timing ---
inference_ms = 0.0
inference_lock = threading.Lock()

def get_label(cls: int) -> str:
    if 0 <= cls < len(COCO_LABELS):
        return COCO_LABELS[cls]
    return f"class_{cls}"

def draw_detections(request):
    global inference_ms
    thr = state.get()

    with MappedArray(request, "main") as m:
        h, w = m.array.shape[:2]
        frame = m.array.copy()

        # -- CPU inference via OpenCV DNN --
        # picamera2 outputs XBGR8888 (4ch), DNN requires BGR (3ch)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        # TF SSD MobileNetV2: input range 0-255, no mean subtraction, swapRB for RGB input
        blob = cv2.dnn.blobFromImage(
            frame_bgr, size=(300, 300),
            mean=(0, 0, 0),
            scalefactor=1.0, swapRB=True)

        t0 = time.perf_counter()
        net.setInput(blob)
        detections = net.forward()
        elapsed = (time.perf_counter() - t0) * 1000
        with inference_lock:
            inference_ms = elapsed

        # detections shape: (1, 1, N, 7)
        # each row: [_, class_id, score, xmin, ymin, xmax, ymax]
        results = detections[0, 0]
        mask    = results[:, 2] > thr
        results = results[mask]

        # -- Draw BBOX --
        for det in results:
            _, cls, score, xmin, ymin, xmax, ymax = det
            x0, y0 = int(xmin * w), int(ymin * h)
            x1, y1 = int(xmax * w), int(ymax * h)
            label = f"{get_label(int(cls))} {score:.0%}"
            cv2.rectangle(m.array, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(m.array, label, (x0, max(y0 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # -- Overlay --
        cpu = cpu_monitor.get()
        with inference_lock:
            ms = inference_ms

        # CPU usage bar
        bar_x, bar_y, bar_w, bar_h = 10, 10, 160, 18
        filled = int(bar_w * cpu / 100)
        bar_color = (0, 255, 0) if cpu < 50 else (0, 165, 255) if cpu < 80 else (0, 0, 255)
        cv2.rectangle(m.array, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
        cv2.rectangle(m.array, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), bar_color, -1)
        cv2.rectangle(m.array, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (180, 180, 180), 1)
        cv2.putText(m.array, f"CPU: {cpu:4.1f}%",
                    (bar_x + bar_w + 8, bar_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.putText(m.array,
                    "CPU Inference (OpenCV DNN)",
                    (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
        cv2.putText(m.array,
                    f"threshold: {thr:.2f}  inference: {ms:.1f}ms",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

picam2.pre_callback = draw_detections
picam2.start(config, show_preview=True)

print(f"Demo started [CPU inference  threshold={state.get():.2f}]")
print(f"Run in another terminal: python3 imx500_od_ctrl.py")
try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nStopped.")
    if os.path.exists(PIPE_PATH):
        os.remove(PIPE_PATH)
    picam2.stop()
