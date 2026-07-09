---
title: "Kaggle Text-to-Image: uvでPython3.12 GPU環境を構築し、SD-TurboベースラインとYOLOv8ローカル評価を動かす"
emoji: "🎨"
type: "tech"
topics: ["kaggle", "diffusers", "yolo", "pytorch", "uv"]
published: false
---

コンペ（Kaggle: Text-to-Image Generation Challenge）に参加するにあたり、Windowsローカル環境の構築から、Stable DiffusionとYOLOv8を組み合わせたローカル評価パイプライン（ベースライン）の構築までを行いました。その過程をまとめます。

# 1. 開発環境の構築とGPU/CUDA有効化の課題

今回のローカルPCには **Python 3.14** がグローバルにインストールされていましたが、ディープラーニング環境を構築する上で以下の問題に直面しました。

### 直面した問題
1. グローバルの Python 3.14 で `pip install torch` を行うと、デフォルトの PyPI インデックスから CPU-only 版の PyTorch がインストールされてしまい、GPUが使えない。
2. PyTorch の公式インデックス（`download.pytorch.org/whl/cu124` など）には、現時点で Python 3.14 (cp314) 用の CUDA プレビルドホイールが登録されておらず、エラーになる。

### 解決方法：`uv` パッケージマネージャーの導入と Python 3.12 仮想環境の構築
この互換性問題を解決するため、Rust製の超高速Pythonパッケージマネージャー **`uv`** を導入し、Stable Diffusion や PyTorch で動作実績の豊富な **Python 3.12** の仮想環境を切り出すことにしました。

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
これにより、ローカルの GPU (**NVIDIA GeForce RTX 4060**) が正しく認識され、CUDAを利用した高速な画像生成環境が整いました。

---

# 2. パイプライン概要

本プロジェクトのベースラインおよびローカル評価の全体の流れは、以下の図の通りです。

![Kaggle Text-to-Image Pipeline](https://github.com/kito2718/zenn_articles/raw/main/articles/images/zenn_20260709_2245_pipeline.jpg)

### なぜ YOLOv8 で評価するのか？
このコンペティション（Text-to-Image Generation Challenge）の評価指標は **Composition Correctness (F1-score)** です。
プロンプト内に含まれるオブジェクト（名詞）を抽出し、生成された画像の中にそれらが実際に描かれているかを **YOLOv8** などの物体検出器で検出させ、その「期待されたオブジェクト」と「検出されたオブジェクト」の重複度（F1スコア）でアライメントの正しさをスコア化します。

このパイプラインをローカルに構築することで、Kaggleにアップロードする前に**「自分の生成した画像がどれだけ評価器に適合しているか」を完全にシミュレーションすることが可能**になります。

---

# 3. ベースラインの実装 (`src/baseline.py`)

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

# 4. 実行結果とローカル評価スコア

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

# 5. 今後のスコアアップに向けた展望

ベースラインでスコア `0.5102` を得られたため、ここからスコアをさらに向上させるための具体的な施策がクリアになりました。

1.  **高精度な生成モデルへの切り替え**:
    SD-Turbo は速度重視ですが、描写力（特に細かい物体の配置や形状）が甘いため、**SDXL (`stabilityai/stable-diffusion-xl-base-1.0`)** や **Flux** などのより表現力の高いモデルに変更する。
2.  **プロンプト調整 (Prompt Engineering)**:
    YOLOv8 が検出しやすいサイズや配置でターゲットオブジェクトが描かれるよう、プロンプトに品質ワードや強調表現をブレンドする。
3.  **セルフフィードバックイテレーション (最大化ハック)**:
    ローカル環境でシード値を変えて複数枚の画像を生成し、**「ローカルの YOLOv8 で最もF1スコアが高くなった画像」を自動選別して提出用画像とするシステム**を組む。これにより、評価器（YOLOv8）の特性に過学習（最適化）させた高スコアな提出用セットを作成可能です。

次回は、この「セルフフィードバック自動選別システム」の実装と、高画質モデルへの切り替えによるスコア変化を検証していきます！
