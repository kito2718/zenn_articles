---
title: "15-①[AI][Kaggle]Kaggle実践1 ローカルPCにKaggle Titanicの実行環境をつくる"
emoji: "🙆‍♀️"
type: "tech"
topics:
  - "ai"
  - "python"
  - "kaggle"
  - "titanic"
published: true
published_at: "2026-06-27 08:55"
---

[Kaggle実践1(『Titanic生存者予測』1.ローカルPCにKaggle Titanicの実行環境をつくる](https://zenn.dev/rg687076/articles/zenn_260627_0000_00_create_local_titanic_env)
[Kaggle実践1(『Titanic生存者予測』2.初回提出](https://zenn.dev/rg687076/articles/zenn_260627_0000_01_first_submission)
[Kaggle実践1(『Titanic生存者予測』3.Cabinの特徴量エンジニアリング](https://zenn.dev/rg687076/articles/zenn_260627_1940_01_cabin_feature)
[Kaggle実践1(『Titanic生存者予測』4.特徴量エンジニアリング(ランダムフォレストによる年齢補完)](https://zenn.dev/rg687076/articles/zenn_20260702_2031_age_imputation)
[Kaggle実践1(『Titanic生存者予測』5.特徴量エンジニアリング(数値特徴量の非線形変換とビン化)](https://zenn.dev/rg687076/articles/zenn_20260703_2025_fare_log_and_age_binning)

https://www.kaggle.com/c/titanic

← [Kaggle入門14(ゲームAIと強化学習入門)](https://zenn.dev/rg687076/articles/49e1d162bfdeec)
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;[Kaggle実践2(xxxx)](https://zenn.dev/rg687076/articles/xxxx) →

[github公開中](https://github.com/kito2718/KaggleTitanic)

# Abstract
- ローカルでKaggle Titanicの実行環境を構築してみた。

# 概要
KaggleのTitanicコンペに参加してスコアアップを頑張っているのですが、Kaggleにログインして提出するのが面倒になってきた。とりあえず実行するには、やはりローカルPCの環境でサクっと実行したいというモチベーションで構築してみた。

# 環境まとめ
要はpythonをインストールすれば万事OKだった。

| 項目 | 内容 |
|------|------|
| Python | 3.14.6 |
| 仮想環境 | `.venv` (venv) |
| プロジェクトパス | `51_googleantigravity\1st_` |
| pip | 26.1.x |

---

# 注意点(ハマりポイント)
ちょっとだけハマったので残しておきます。
:::message
ポイント

- **仮想環境のパスに日本語を含めたらダメだった。**
`scipy` や `scikit-learn` などの C拡張ライブラリ（DLL）が、日本語を含むパスからは正しく読み込めんかったっす。
:::

# 1. 構築手順
## 1.1. pythonのインストール

[python公式](https://www.python.org/downloads/release/python-3146/)からダウンロード → インストール。
![](https://static.zenn.studio/user-upload/1d1950b8d6ca-20260627.png =500x)*一番下まで移動すると表示されます*

ついでにパスも通しておきます。
```batch
setx PATH "%PATH%;pythonインストールフォルダ"
```

## 1.2. プロジェクトフォルダの作成
自分のPCなので好きな場所に。
```batch
mkdir 51_googleantigravity\1st_
```

## 1.3. 仮想環境の作成
pythonの仮想環境を生成。
```batch
# プロジェクトフォルダに移動
cd 51_googleantigravity\1st_
python.exe -m venv .venv
```
作成後の構造:
```tree
1st_/
└── .venv/
    ├── Scripts/
    │   ├── activate
    │   ├── python.exe
    │   └── pip.exe
    └── Lib/
```

## 1.4. パッケージのインストール
pythonコードで使用するいろいろをインストールする。
```batch
python.exe -m pip install --upgrade pip pandas numpy scikit-learn matplotlib seaborn jupyter kaggle xgboost lightgbm
```

インストールされる主要パッケージは以下の通り:

| パッケージ | バージョン | 用途 |
|-----------|-----------|------|
| pandas | 3.0.3 | データ操作 |
| numpy | 2.5.0 | 数値計算 |
| scikit-learn | 1.9.0 | 機械学習 |
| matplotlib | 3.11.0 | グラフ描画 |
| seaborn | 0.13.2 | 統計グラフ |
| xgboost | 3.3.0 | 勾配ブースティング |
| lightgbm | 4.6.0 | 勾配ブースティング |
| jupyter / notebook | 7.6.0 | Notebook 環境 |
| kaggle | 2.2.3 | Kaggle CLI |

## 1.5. フォルダ整理
画像出力とか、notebookの置き場所とか決めておきたい。
```batch
mkdir data\raw
mkdir data\processed
mkdir notebooks
mkdir submissions
```
整理後のフォルダ構成
```
1st_/
├── .venv/
├── data/
│   ├── raw/          ← Kaggle データ置き場
│   └── processed/    ← グラフ・前処理済みデータ出力先
├── notebooks/        ← Notebook 置き場
└── submissions/      ← 提出ファイル出力先
```

## 1.6. requirements.txt の作成
```batch
python.exe -m pip freeze > requirements.txt
```

## 1.7. Notebook の作成
`notebooks\titanic_eda.ipynb` に下記を作成。

内容(セル構成):
| セル | 内容 |
|------|------|
| 1 | ライブラリの読み込み |
| 2 | データの読み込み（train.csv / test.csv） |
| 3 | 探索的データ分析（欠損値・生存率・年齢分布・相関ヒートマップ） |
| 4 | 特徴量エンジニアリング（称号抽出・年齢補完・家族人数等） |
| 5 | モデル生成 |
| 6 | 提出ファイルの生成（submissions/submission.csv 出力） |


## 1.8. Notebook の起動
Notebookを起動。
```batch
# プロジェクトフォルダに移動
cd 51_googleantigravity\1st_
# 仮想環境の有効化
activate
# Notebook起動
jupyter notebook notebooks\titanic_eda.ipynb
```
ブラウザが自動で開く。開かない場合は表示された URL をコピーしてブラウザに貼る。

## 1.9. Kaggle操作の準備
1. [kaggleのsettigs](https://www.kaggle.com/settings)にアクセス → API Tokensタブ
2. Legacy API Credentialsの方の"Create Legacy API Key"を押下
3. ダウンロードされた kaggle.json を C:\Users\xxx\\.kaggle\kaggle.json に配置
4. 以下を実行:

```batch
# 仮想環境の有効化
activate
# Kaggle APIでTitanicデータダウンロード
kaggle competitions download -c titanic -p data\raw
#  ZIPを解凍(Windows標準のtarコマンドを使用)
tar -xf data\raw\titanic.zip -C data\raw
```

いちおう、手動ダウンロードの方法も。
> https://www.kaggle.com/c/titanic/data から以下3ファイルをダウンロードして data\raw\ に配置:
> ・train.csv
> ・test.csv
> ・gender_submission.csv

## 1.10. Kaggle への提出
```batch
kaggle competitions submit -c titanic -f submissions\submission.csv -m "コメント"
```
よし！できた。
これで、ローカル環境で動作確認 → Kaggle提出までができるようになったばい。

お役に立てれば。

English version:
[Kaggle Practice 1: Setting Up a Local Environment for the Kaggle Titanic Competition](https://dev.to/kito2718/setting-up-kaggle-titanic-environment-on-a-local-pc-336k)
[Kaggle Practice 2: First Submission](https://dev.to/kito2718/kaggle-titanic-my-first-submission-eda-feature-engineering-and-model-evaluation-896)
[Kaggle Practice 3: Feature Engineering for Cabin](https://dev.to/kito2718/kaggle-titanic-cabin-feature-engineering-is-it-really-effective-44nc)
[Kaggle Practice 4: Feature Engineering (Imputing Age with Random Forest)](https://dev.to/kito2718/kaggle-titanic-improving-survival-prediction-with-random-forest-age-imputation-5b3l)
