---
title: "16-①Kaggle実践2「Text-to-Image Generation Challenge」コンペ用にローカル検証環境を構築してみた"
emoji: "🎨"
type: "tech"
topics: ["kaggle", "diffusers", "yolo", "pytorch", "uv"]
published: true
published_at: "2026-07-09 22:44"
---

[Kaggle実践2『Tex2Img Gen』1.ローカルPCに検証環境をつくってみた](https://zenn.dev/rg687076/articles/zenn_20260709_2244_baseline_sd_yolo)

# Abstract
* 過去コンペ「Text-to-Image Generation Challenge」に参加
* ローカル検証環境を構築してみた。

# 背景
Kaggle Titanicを卒業して、次何やるかって考えたときに、「やっぱ次は画像系やろ！」って心の声に従って、このコンペを選択しました。

# インストールライブラリ
- python3.12 ... 3.14が最新だけどそれを使うと、pytorchでGPUが使えず。なので3.12。
- uv ... pythonのパッケージマネージャ
- yolo ... 
- sd ... 

# マシン構成
- OS: Windows11
- メモリ: 32GB
- CPU: 論理コア数24
- GPU: Geforce RTX 4060
- ストレージ: HDD 空き500GB

# 1. インストール手順
```powershell
# 1. uv のインストール
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Python 3.12 を指定した仮想環境の作成 (uvが自動で対象Pythonバージョンをダウンロードしてくれます)
uv venv --python 3.12

# 3. CUDA 12.4 対応の PyTorch および Torchvision を仮想環境にインストール
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 4. 残りの依存ライブラリ (diffusers, transformers, accelerate, pandas, ultralytics) をインストール
uv pip install diffusers "transformers<5.0.0" accelerate pandas ultralytics
```
これで GPU設定(**NVIDIA GeForce RTX 4060**)も完了し、CUDAを利用した高速な画像生成環境が整いました。

# 2. 一回通して実行してみる
構築した環境でがちゃんと動作するのか、実装→実行→出力を一通り実行する。

## 2.1. 実行の流れの解説

構築したベースラインコードの要点は以下の通りです。

*   **入力データ**: Kaggle API経由で取得した `DreamLayer-Prompt-Kaggle.txt`（49個のテスト用プロンプトが含まれるテキストファイル）
*   **画像生成**: 高速化のために **`stabilityai/sd-turbo`** (1ステップ生成モデル) を採用。
*   **ローカル評価**: 生成画像に対して `yolov8n.pt` (YOLOv8 nano) で物体検出を行い、プロンプト内の名詞キーワードとマッチングしてF1スコアを算出。

### 主なソースコード (`src/baseline.py` の抜粋)

```python
import torch
import pandas as pd
import re
from pathlib import Path
from diffusers import StableDiffusionPipeline
from ultralytics import YOLO

# 共通のCOCOクラスに対応するオブジェクト
common_objects = {
    'man', 'woman', 'person', 'dog', 'cat', 'car', 'truck', 'train', 'airplane',
    'pizza', 'cake', 'donut', 'chair', 'table', 'bed', 'toilet', 'sink', 'mirror', 'clock', 'umbrella'
    # ... (一部省略)
}

def extract_expected_objects(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return set(word for word in words if word in common_objects)

def calculate_f1_score(expected, detected):
    if len(expected) == 0 and len(detected) == 0:
        return 1.0
    if len(expected) == 0 or len(detected) == 0:
        return 0.0
    true_positives = len(expected.intersection(detected))
    precision = true_positives / len(detected)
    recall = true_positives / len(expected)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)
```

画像生成処理では、再現性担保のために `torch.Generator` に固定シード値（`seed = 42`）を設定して画像を出力し、検出したスコアから `submission.csv` を生成します。

---

## 2.2. 実行結果とローカル評価スコア

ベースラインスクリプトを GPU 上で実行した結果、以下の出力を得ました。

```text
Using device: cuda
Reading prompts from input\DreamLayer-Prompt-Kaggle.txt...
Loaded 49 prompts.
Loading pipeline for stabilityai/sd-turbo...
...
[49/49] Generating: 'A group of people standing on a snow covered hill.' -> 0049.png
Image generation complete.
Loading YOLOv8 model for local evaluation...
Running YOLO detection and F1 score calculation...
Saved results.csv to output\results.csv
Saved submission.csv to output\submission.csv

==================================================
LOCAL EVALUATION COMPLETE
Mean F1 Score: 0.5102
==================================================
```

SD-Turbo（1ステップモデル）を使ったローカルのベースライン F1 スコアは **`0.5102`** となりました。

### 出力されたファイル群 (output/)
- `output/images/0001.png` 〜 `0049.png` (生成された画像)
- `output/results.csv` (プロンプト、検出オブジェクト、F1スコアの詳細ログ)
- `output/submission.csv` (Kaggle提出用CSV)
- `output/config-dreamlayer.json` (生成時のパラメータ設定)

---

# 5. 今後の方針

ベースラインでスコア `0.5102` を得られたため、ここからスコアをさらに向上させるための具体的な施策がクリアになりました。

1.  **高精度な生成モデルへの切り替え**:
    SD-Turbo は速度重視ですが、描写力（特に細かい物体の配置や形状）が甘いため、**SDXL (`stabilityai/stable-diffusion-xl-base-1.0`)** や **Flux** などのより表現力の高いモデルに変更する。
2.  **プロンプト調整 (Prompt Engineering)**:
    YOLOv8 が検出しやすいサイズや配置でターゲットオブジェクトが描かれるよう、プロンプトに品質ワードや強調表現をブレンドする。
3.  **セルフフィードバックイテレーション (最大化ハック)**:
    ローカル環境でシード値を変えて複数枚の画像を生成し、**「ローカルの YOLOv8 で最もF1スコアが高くなった画像」を自動選別して提出用画像とするシステム**を組む。これにより、評価器（YOLOv8）の特性に過学習（最適化）させた高スコアな提出用セットを作成可能です。

次回は、この「セルフフィードバック自動選別システム」の実装と、高画質モデルへの切り替えによるスコア変化を検証していきます！
