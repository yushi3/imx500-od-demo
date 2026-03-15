# imx500-od-demo

Raspberry Pi AI Camera (IMX500) 向けリアルタイム物体検出デモ — オンセンサー推論テンソルとバウンディングボックスを可視化します。

[English](README.md)

---

## 概要

**Raspberry Pi AI Camera** は、Sony の **IMX500** インテリジェントビジョンセンサーを搭載しています。IMX500 はカメラチップ上に DNN アクセラレータを内蔵しており、**推論はすべてカメラモジュール内部で完結**します。Raspberry Pi の CPU には推論結果（出力テンソル）のみが送られ、画像データを CPU で処理する必要はありません。

このデモはその仕組みをリアルタイムで可視化します：

- **ターミナル A**: IMX500 からの生出力テンソルと、パース後の検出結果を表示
- **ターミナル B**: threshold をリアルタイムに変更するコントローラー
- **プレビューウィンドウ**: バウンディングボックスとラベルを重畳したライブ映像

```
┌─────────────────────────────┐
│        IMX500 (オンチップ)   │
│  センサー -> ISP -> DNN      │
│              |              │
│       Output Tensors        │
└──────────────┬──────────────┘
               │ MIPI CSI-2 (Virtual Channel)
               ↓
┌─────────────────────────────┐
│       Raspberry Pi          │
│  libcamera -> Picamera2     │
│  -> パース -> BBOX 描画      │
└─────────────────────────────┘
```

---

## 必要なハードウェア

- Raspberry Pi 4 または 5
- [Raspberry Pi AI Camera](https://www.raspberrypi.com/products/ai-camera/) (IMX500)

---

## 必要なソフトウェア

```bash
sudo apt update && sudo apt full-upgrade
sudo apt install -y imx500-all python3-picamera2 python3-opencv
sudo reboot
```

---

## インストール

```bash
git clone https://github.com/<your-username>/imx500-od-demo.git
cd imx500-od-demo
```

---

## 使い方

### ターミナル A — デモを起動

```bash
# SSD MobileNetV2（デフォルト）
python3 imx500_od_demo.py

# EfficientDet Lite0
python3 imx500_od_demo.py --model efficientdet

# NanoDet Plus
python3 imx500_od_demo.py --model nanodet

# threshold を手動で指定
python3 imx500_od_demo.py --model ssd --threshold 0.60

# RPK ファイルを直接指定
python3 imx500_od_demo.py --model /path/to/your_model.rpk
```

### ターミナル B — threshold をリアルタイムに変更

```bash
python3 imx500_od_ctrl.py
```

先に `imx500_od_demo.py` を起動してから、別ターミナルでコントローラーを起動してください。

### コントローラーのキー操作

| キー | 操作 |
|------|------|
| `+` / `=` | threshold を +0.01 |
| `-` / `_` | threshold を -0.01 |
| `1` – `9` | threshold を 0.1 〜 0.9 に直接セット |
| `q` | コントローラーを終了 |

---

## ファイル構成

| ファイル | 説明 |
|----------|------|
| `imx500_od_demo.py` | メインデモスクリプト |
| `imx500_od_ctrl.py` | リアルタイム threshold コントローラー |

---

## 対応モデル

| 名前 | モデルファイル | COCO mAP | デフォルト threshold | 備考 |
|------|--------------|----------|----------------------|------|
| `ssd` | `imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk` | ~22% | 0.50 | 高速、デフォルト |
| `efficientdet` | `imx500_network_efficientdet_lite0_pp.rpk` | ~26% | 0.50 | バランス型 |
| `nanodet` | `imx500_network_nanodet_plus_416x416_pp.rpk` | ~34% | 0.25 | 最高精度、Anchor-free |

すべてのモデルは **TF OD API 互換の出力フォーマット** に従っています：

```
tensor[0] boxes    (1, N, 4)  各候補の BBOX 座標 [ymin, xmin, ymax, xmax]
tensor[1] scores   (1, N)     信頼スコア（降順ソート済み）
tensor[2] classes  (1, N)     クラス ID
tensor[3] max_det  (1, 1)     候補数上限（固定値、モデル依存）
```

モデルが絶対ピクセル値で座標を出力する場合、自動的に 0~1 に正規化されます。

---

## ターミナル出力例

```
============================================================
[RAW OUTPUT TENSORS from IMX500 @ 14:32:05]
  tensor[0] boxes    BBOX coordinates [ymin,xmin,ymax,xmax] per candidate
             shape=(1, 100, 4)
             [[0.0000,0.0690,1.0000,0.9141], [0.0156,0.8125,0.3438,1.0000], [0.0078,0.0078,0.9922,0.8828], ...]
             -> format: normalized (0~1)
  tensor[1] scores   Confidence score 0~1 per candidate
             shape=(1, 100)  [0.7773 0.5000 0.4375 0.3789 ...]
             -> above threshold=0.50: 1 candidates
  tensor[2] classes  Class ID (integer) per candidate
             shape=(1, 100)  [0.0000 83.0000 31.0000 72.0000 ...]
             -> top 5 labels: ['person', 'book', 'tie', 'laptop', 'book']
  tensor[3] max_det  Max candidates upper limit (fixed value)
             shape=(1, 1)    [100.0000 ...]
[PARSED DETECTIONS]  (threshold=0.50)
  count: 1
  [1] person                score=0.777  box=[ymin=0.000 xmin=0.069 ymax=1.000 xmax=0.911]
============================================================
```

---

## 仕組み

IMX500 は **カメラ映像** と **出力テンソル** を、1本の MIPI CSI-2 ケーブル上の別々の仮想チャンネルで同時に送信します：

| MIPI Virtual Channel | 内容 |
|---------------------|------|
| Ch.0 | 映像フレーム（YUV/RGB） |
| Ch.1 | 出力テンソル（推論結果） |

`libcamera` が両ストリームを受信し、テンソルをフレームのメタデータとして提供します。`Picamera2` の `IMX500` クラスが `get_outputs(metadata)` でそれを取り出します。CPU での推論処理は一切行われません。

---

## ライセンス

MIT
