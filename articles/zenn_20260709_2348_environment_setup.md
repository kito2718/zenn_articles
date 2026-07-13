---
title: "17-①Kaggle実践2「Stanford RNA 3D Folding Part 2」コンペ用にローカル検証環境を構築してみた"
emoji: "🧬"
type: "tech"
topics: ["kaggle", "rna", "python", "pytorch", "bioinformatics"]
published: true
series: "Stanford RNA 3D Folding Part 2"
tags: ["kaggle", "rna", "python", "pytorch"]
---

[![アイキャッチ画像](https://github.com/kito2718/zenn_articles/raw/main/articles/images/zenn_eyecatch_environment_setup.png)](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2)*Stanford RNA 3D Folding Part 2*

:::message alert
このコンペは「Late Submit」ができません。
※完了コンペは、賞金等には参加できないものの「Late Submit」って形で提出できるのですが、このコンペはそれができません。
:::

# Abstruct
- 過去コンペ「Stanford RNA 3D Folding Part 2(2026.1.8-2026.3.26)」に参加
- ローカル検証環境を構築
- ベースライン提出
- inputデータは300GB超え。

# 概要
Kaggleにはほんとに大量多彩のデータサイエンスのテーマがあってですね。中には「どうやんの？」って解法がみえないものもある訳ですよ。データサイエンスを目指す者として、今回はそんなぱっと見で「どうやるのかさっぱりわからん」ってのを勉強したかったんです。

で、見つけた「[Stanford RNA 3D Folding Part 2](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2)」ですよ。
このコンペティションはですね、「RNAの一次構造(配列情報)から3次元座標(3D structures)を予測する、生命科学と機械学習が融合した高度な課題」ですと。もう何を言っているのかわかりません。

まさに望むところ。理解したるばい。

# 環境構築
基本、pythonのインストールと依存関係のインストールです。新しいです。

## 1. Pythonのライブラリ管理ツール「uv」
`uv`使います。`pip`よりいろいろいいみたいです。`pip`より新しいです。

**uvの使い方**
- `uvのインストール`: `pip install uv`
- `uv init`: プロジェクト開始時に1回。`pyproject.toml`の雛形が生成される。
- `uv sync`: 環境流用時に1回。仮想環境からパッケージまで環境を合わせてくれる。
- `uv run`: python実行時に毎回。activateを省ける。

## 2. もろもろインストール
下記`pyproject.toml`を準備して、`uv sync`を実行。
あとはよしなにやってくれる。

### `pyproject.toml` の定義
```toml
[project]
name = "stanford-rna-3d-folding-ii"
version = "0.1.0"
description = "Stanford RNA 3D Folding Part 2 Kaggle Competition"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0.0",
    "lightning>=2.0.0",
    "numpy>=1.20.0",
    "pandas>=1.3.0",
    "scipy>=1.7.0",
    "scikit-learn>=1.0.0",
    "biopython>=1.80",
    "pyyaml>=6.0",
    "tqdm>=4.60.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
]
```

## 3. Kaggleデータの取得
このコンペの全データは300GB超え!
空き容量と相談しながらのダウンロードですね。

![download all](https://github.com/kito2718/zenn_articles/raw/main/articles/images/zenn_20260712_1229_downloadall.png)

## 4. 一通りの動作確認
データの動作確認と簡易構造分析を実施します。

### 4.1. データを読み込んでみる
`train_labels.csv` はサイズが大きいので、メモリ不足(OOM)を防ぐために `nrows` 引数を指定して先頭部分のみをロードするのがポイントです。

```python
import pandas as pd
import os

input_dir = "input"

# 1. 配列データの読み込み
train_seq = pd.read_csv(os.path.join(input_dir, "train_sequences.csv"))
print(f"train_sequences shape: {train_seq.shape}")

# 2. 構造ラベルデータの読み込み (巨大なため先頭のみ制限ロード)
train_labels = pd.read_csv(os.path.join(input_dir, "train_labels.csv"), nrows=100)
print(f"train_labels (first 100) shape: {train_labels.shape}")
```

##### 出力結果とデータ構造の解説
スクリプトを実行し、以下の形式でデータが正しく取得できていることを確認しました。

```text
train_sequences shape: (5716, 8)
  target_id  ...                   ligand_SMILES
0      4TNA  ...                          [Mg+2]
1      6TNA  ...                          [Mg+2]

train_labels (first 100 rows) shape: (100, 8)
       ID resname  resid    x_1    y_1     z_1 chain  copy
0  157D_1       C      1  4.843 -5.640  13.265     A     1
1  157D_2       G      2  3.385 -7.613   8.267     A     1
```

- **配列データ (`train_sequences.csv`)**: 5,716件のRNA配列データが格納されています。予測の主対象であるRNA配列自体の他に、結合しているリガンド情報(`ligand_SMILES`)など、モデルの特徴量として有用な情報が含まれています。
- **構造ラベルデータ (`train_labels.csv`)**: RNAを構成する各ヌクレオチド(塩基単位、`resname`, `resid`)ごとの3次元座標(`x_1`, `y_1`, `z_1`)が格納されています。この座標を予測することが本コンペのゴールになります。

---

### 4.2. ベースラインの生成
環境が構築できたら、次はコンペへ提出(Submit)するための予測データを作成します。
ここでは、最もシンプルなアプローチとして、訓練データにおけるC1'原子座標の平均値(重心)を算出し、テストデータのすべてのヌクレオチドに対してその平均座標を割り当てる「平均値ベースライン」を作成します。

以下のスクリプトを `src/baseline_mean.py` として作成します。

```python
import pandas as pd
import numpy as np
import os
import sys

def main():
    input_dir = "input"
    output_dir = "output"
    
    train_labels_path = os.path.join(input_dir, "train_labels.csv")
    sample_sub_path = os.path.join(input_dir, "sample_submission.csv")
    output_path = os.path.join(output_dir, "submission.csv")
    
    # フォルダの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 訓練データの座標の平均値(代表値)の計算
    print("代表値(平均座標)の計算中...")
    if not os.path.exists(train_labels_path):
        print(f"Error: {train_labels_path} が見つかりません。", file=sys.stderr)
        sys.exit(1)
        
    # メモリ節約のため、先頭100万行から平均を算出
    train_chunk = pd.read_csv(train_labels_path, nrows=1000000)
    x_mean = train_chunk['x_1'].mean()
    y_mean = train_chunk['y_1'].mean()
    z_mean = train_chunk['z_1'].mean()
    print(f"算出された平均座標: x={x_mean:.3f}, y={y_mean:.3f}, z={z_mean:.3f}")
    
    # 2. 提出用サンプルの読み込み
    print("提出用サンプル (sample_submission.csv) の読み込み中...")
    if not os.path.exists(sample_sub_path):
        print(f"Error: {sample_sub_path} が見つかりません。", file=sys.stderr)
        sys.exit(1)
        
    sub = pd.read_csv(sample_sub_path)
    
    # 3. 5つの予測モデルすべてに平均座標を割り当て
    print("平均座標の割り当て中...")
    for i in range(1, 6):
        sub[f'x_{i}'] = x_mean
        sub[f'y_{i}'] = y_mean
        sub[f'z_{i}'] = z_mean
        
    # 4. 座標のクリップ (-999.999 から 9999.999)
    print("座標のクリップ処理中 (-999.999 から 9999.999) ...")
    coord_cols = [f'{axis}_{i}' for i in range(1, 6) for axis in ['x', 'y', 'z']]
    sub[coord_cols] = sub[coord_cols].clip(-999.999, 9999.999)
    
    # 5. CSVファイルとして保存
    print(f"結果を {output_path} に保存中...")
    sub.to_csv(output_path, index=False)
    print(f"提出用ファイルが作成されました！形状: {sub.shape}")

if __name__ == "__main__":
    main()
```

このスクリプトを実行します。

```bash
uv run python src/baseline_mean.py
```

実行すると、以下のように `output/submission.csv` が出力されます。

```text
代表値(平均座標)の計算中...
算出された平均座標: x=83.655, y=89.019, z=85.731
提出用サンプル (sample_submission.csv) の読み込み中...
平均座標の割り当て中...
座標のクリップ処理中 (-999.999 から 9999.999) ...
結果を output\submission.csv に保存中...
提出用ファイルが作成されました！形状: (9762, 18)
```

### 4.3. 初回提出
:::message alert
このコンペは「Late Submit」ができませんでした。orz
:::

だれかの役に立つかもなので、コードは下記に公開します。

```python
import pandas as pd
import numpy as np
import os
import sys

def main():
    # Kaggle環境用のインプットパス
    input_dir = "/kaggle/input/competitions/stanford-rna-3d-folding-2"
    output_dir = "."  # カレントディレクトリ直下(/kaggle/working/)
    
    train_labels_path = os.path.join(input_dir, "train_labels.csv")
    sample_sub_path = os.path.join(input_dir, "sample_submission.csv")
    output_path = os.path.join(output_dir, "submission.csv")
    
    # 1. 訓練データの座標の平均値(代表値)の計算
    print("代表値(平均座標)の計算中...")
    # メモリ節約のため、先頭100万行から平均を算出
    train_chunk = pd.read_csv(train_labels_path, nrows=1000000)
    x_mean = train_chunk['x_1'].mean()
    y_mean = train_chunk['y_1'].mean()
    z_mean = train_chunk['z_1'].mean()
    print(f"算出された平均座標: x={x_mean:.3f}, y={y_mean:.3f}, z={z_mean:.3f}")
    
    # 2. 提出用サンプルの読み込み
    print("提出用サンプル (sample_submission.csv) の読み込み中...")
    sub = pd.read_csv(sample_sub_path)
    
    # 3. 5つの予測モデルすべてに平均座標を割り当て
    print("平均座標の割り当て中...")
    for i in range(1, 6):
        sub[f'x_{i}'] = x_mean
        sub[f'y_{i}'] = y_mean
        sub[f'z_{i}'] = z_mean
        
    # 4. 座標のクリップ (-999.999 から 9999.999)
    print("座標のクリップ処理中 (-999.999 から 9999.999) ...")
    coord_cols = [f'{axis}_{i}' for i in range(1, 6) for axis in ['x', 'y', 'z']]
    sub[coord_cols] = sub[coord_cols].clip(-999.999, 9999.999)
    
    # 5. CSVファイルとして保存
    print(f"結果を {output_path} に保存中...")
    sub.to_csv(output_path, index=False)
    print(f"提出用ファイルが作成されました！形状: {sub.shape}")

if __name__ == "__main__":
    main()
```

### まとめ
このコンペも「Late Submit」ができず残念。
いちおうローカル検証環境の構築から、データの初期ロード検証、そして平均値を用いた超シンプルなベースラインモデルによる初回 Notebook の作成まではできた。

お役に立てれば。
