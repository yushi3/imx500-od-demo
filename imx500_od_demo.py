#!/usr/bin/env python3
"""
IMX500 Object Detection Demo
- TF OD API互換フォーマットのモデルに対応
- モデルは --model で切り替え可能
- BBOX座標の正規化を自動判定
- RAW Output Tensorとパース結果を両方表示
- threshold をキー入力でリアルタイム変更可能
    +    → +0.05
    -    → -0.05
    1〜9 → 0.1〜0.9 に直接セット
"""

import cv2
import sys
import time
import tty
import termios
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

# --- 引数 ---
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="ssd",
                    help=f"モデル名 ({', '.join(MODELS.keys())}) またはRPKファイルの直接パス")
parser.add_argument("--threshold", type=float, default=None,
                    help="検出スコア閾値 (省略時はモデルごとのデフォルト値を使用)")
args = parser.parse_args()

model_path = MODELS.get(args.model, args.model)
model_name = args.model
threshold  = args.threshold if args.threshold is not None else DEFAULT_THRESHOLD.get(model_name, 0.50)

# --- IMX500 初期化 ---
imx500 = IMX500(model_path)
intrinsics = imx500.network_intrinsics or NetworkIntrinsics()
intrinsics.task = "object detection"
intrinsics.update_with_defaults()

picam2 = Picamera2(imx500.camera_num)
config = picam2.create_preview_configuration(
    controls={"FrameRate": 30}, buffer_count=12)

# --- threshold をスレッド間で共有 ---
class State:
    def __init__(self, initial):
        self.threshold = initial
        self.lock = threading.Lock()

    def set(self, val):
        with self.lock:
            self.threshold = round(max(0.05, min(0.95, val)), 2)
            print(f"\n>>> threshold = {self.threshold:.2f} <<<")

    def get(self):
        with self.lock:
            return self.threshold

state = State(threshold)

# --- キー入力スレッド ---
def key_listener():
    """
    ターミナルをrawモードにしてキー入力を監視する。
    +/=  → +0.05
    -/_  → -0.05
    1〜9 → 0.1〜0.9 に直接セット
    q    → 終了
    """
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch in ('+', '='):
                state.set(state.get() + 0.05)
            elif ch in ('-', '_'):
                state.set(state.get() - 0.05)
            elif ch.isdigit() and ch != '0':
                state.set(int(ch) * 0.1)
            elif ch in ('q', 'Q', '\x03'):  # q or Ctrl+C
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
                import os, signal
                os.kill(os.getpid(), signal.SIGINT)
                break
    except Exception:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

threading.Thread(target=key_listener, daemon=True).start()

# --- ユーティリティ ---
def get_label(cls: int) -> str:
    if intrinsics.labels and cls < len(intrinsics.labels):
        return intrinsics.labels[cls]
    return f"class_{cls}"

def normalize_boxes(boxes, np_outputs):
    """
    BBOX座標を 0~1 正規化に統一する。
      - 全候補の最大値が 1.1 以下 → 正規化済みとみなす（わずかな超過を許容）
      - 最大値が 1.1 超            → 絶対ピクセル値、入力サイズで割って正規化
    入力サイズは全候補のymax/xmax最大値から推定し32の倍数に丸める（320, 416等）。
    """
    if len(boxes) == 0:
        return boxes, None

    all_boxes = np_outputs[0][0]
    all_max   = float(all_boxes.max())

    if all_max <= 1.1:
        return boxes, None

    input_size = float(all_boxes[:, 2:].max())
    input_size = round(input_size / 32) * 32
    return boxes / input_size, input_size

def parse_ssd(np_outputs, thr):
    """
    TF OD API互換フォーマット:
      tensor[0] boxes   (1, N, 4)  [ymin, xmin, ymax, xmax]
      tensor[1] scores  (1, N)     信頼スコア 降順ソート済み
      tensor[2] classes (1, N)     クラスID
      tensor[3] max_det (1, 1)     候補数上限 (固定値、使用しない)
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
    """
    tensor[0] を [[ymin,xmin,ymax,xmax], ...] 形式で表示する。
    n_show: 表示する候補数
    """
    raw = np_outputs[0][0][:n_show]  # (n_show, 4)
    items = [f"[{','.join(f'{v:.4f}' for v in row)}]" for row in raw]
    return "[" + ", ".join(items) + ", ...]"

def draw_detections(request):
    metadata   = request.get_metadata()
    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    if np_outputs is None:
        return

    thr = state.get()

    # ── ① 生テンソル表示 ─────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[RAW OUTPUT TENSORS from IMX500 @ {time.strftime('%H:%M:%S')}]")

    tensor_meta = [
        ("boxes  ", "各候補のBBOX座標 [ymin,xmin,ymax,xmax]"),
        ("scores ", "各候補の信頼スコア 0~1"),
        ("classes", "各候補のクラスID (整数)"),
        ("max_det", "このモデルの候補数上限 (固定値)"),
    ]

    for i, tensor in enumerate(np_outputs):
        flat = tensor.flatten()
        name, desc = tensor_meta[i] if i < len(tensor_meta) else (f"output{i}", "")

        print(f"  tensor[{i}] {name}  {desc}")

        if i == 0:
            # boxes は [[y,x,y,x], ...] 形式で表示
            print(f"             shape={tensor.shape}")
            print(f"             {fmt_boxes(np_outputs, n_show=3)}")
            all_max = float(np_outputs[0][0].max())
            is_norm = all_max <= 1.1
            print(f"             → 座標形式: {'0~1正規化済み' if is_norm else '絶対ピクセル値 (自動正規化します)'}")
        else:
            sample = " ".join(f"{v:.4f}" for v in flat[:12])
            print(f"             shape={tensor.shape}  [{sample} ...]")

            if i == 1:  # scores
                n_over = int((flat > thr).sum())
                print(f"             → threshold={thr:.2f} 以上: {n_over}個")

            if i == 2:  # classes
                top_labels = [get_label(int(c)) for c in flat[:5]]
                print(f"             → 上位5候補のラベル: {top_labels}")

    # ── ② パース後の結果表示 ──────────────────────────────
    boxes, scores, classes, input_size = parse_ssd(np_outputs, thr)

    print(f"[PARSED DETECTIONS]  (threshold={thr:.2f}  +/-/1-9キーで変更)")
    if input_size:
        print(f"  (座標を {input_size:.0f}px → 0~1 に正規化)")
    if len(scores) == 0:
        print("  (検出なし)")
    else:
        print(f"  検出数: {len(scores)}")
        for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
            print(f"  [{i+1}] {get_label(cls):20s}  score={score:.3f}  "
                  f"box=[ymin={box[0]:.3f} xmin={box[1]:.3f} "
                  f"ymax={box[2]:.3f} xmax={box[3]:.3f}]")
    print(f"{'='*60}")

    # ── ③ BBOX描画 ────────────────────────────────────────
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

        # threshold をオーバーレイ表示
        cv2.putText(m.array,
                    f"IMX500: On-Sensor AI ({model_name})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.putText(m.array,
                    f"threshold: {thr:.2f}  (+/-/1-9 to change)",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

imx500.show_network_fw_progress_bar()
picam2.pre_callback = draw_detections
picam2.start(config, show_preview=True)

print(f"デモ開始 [model={model_name}  threshold={state.get():.2f}]")
print(f"キー操作: +/- で±0.05、1〜9で直接セット、q で終了")
try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\n終了")
    picam2.stop()
