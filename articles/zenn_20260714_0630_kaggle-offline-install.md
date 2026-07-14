---
title: "(Kaggle提出のための)Offline環境でプリインストールされていないライブラリをインストールする方法"
emoji: "📦"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["kaggle", "python", "pip", "offline"]
published: true
series: "Biohub細胞トラッキング挑戦記"
tags: ["kaggle", "pip", "offline-install", "zarr"]
---

[![アイキャッチ画像](https://github.com/kito2718/zenn_articles/raw/main/articles/images/zenn_20260714_0630_BitCellTracking_eyecatch.png =500x)](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
*Biohub - Cell Tracking During Development*

## Abstruct
- (Kaggle用に)プリインストールされていない外部ライブラリのインストール手順のまとめ

## 概要
Kaggleのコードコンペティションって、Notebook提出のとき、インターネット接続不可でも動くのが条件になんですが、プリインストールされていない外部ライブラリ(例: `zarr`)だと、import エラーになります。かつ、コードの先頭で`!pip install zarr`ってやってもエラーになります。
その問題を安全かつ確実にオフラインインストールで解決する方法を解説します。

## 全体フロー
オフラインインストールまでの手順です。
まぁ早い話が、一回ローカルPCにDLしてそれをKaggleにupするってことですね。

```mermaid
flowchart TD
    A[1. ローカルPCにダウンロード] -->|Kaggle環境を強制指定してwhlを取得| B(zarr_wheels_fixed フォルダ)
    B -->|2. KaggleにDatasetとしてアップロード| C[(Kaggle Dataset: zarr-offline-whl)]
    C -->|3. 本番用Notebookに追加| D(Kaggle Notebook / Internet off)
    D -->|4. オフラインインストール実行| E[正常にライブラリが利用可能に！]
```

## 手順

#### 1. ローカルPCでパッケージファイルをダウンロードする
コマンドプロンプト(cmd)で、以下コマンド実行。
`--python-version 3.12`(Kaggle環境（Linux/Python 3.12）を強制指定するオプション)を指定するのがポイント。

```batch
pip download zarr -d ./zarr_wheels_fixed --only-binary=:all: --platform manylinux2014_x86_64 --python-version 3.12 --implementation cp
```

これによって、指定したフォルダ内に `numcodecs` などの依存関係も含めた複数の `.whl` ファイルがダウンロードされます。
ダウンロード完了後、以下のコマンドでフォルダを開いておきます。

```cmd
start zarr_wheels_fixed
```

#### 2. Kaggleにデータセットとしてアップロードする
1. Kaggleの「Datasets」ページから **「New Dataset」** をクリックします。
2. タイトルを入力(例: `zarr-offline-whl`)します。
3. 先ほど開いたエクスプローラーから、ダウンロードしたファイルをすべてドラッグ&ドロップし、アップロードします。
4. アップロード完了後、**「Create」** を押してデータセットを作成します。

#### 3. ノートブックにマウントしてオフラインインストールを実行する
1. 本番用のノートブックの右側パネルで、**「Internet off」** に設定します。
2. **「+ Add Input」** をクリックし、作成したデータセットをノートブックに追加します。

以下の画像のように、ご自身のWorkから検索して追加します。

![Kaggle Dataset Search Filter](https://github.com/kito2718/zenn_articles/raw/main/articles/images/zenn_20260714_0630_kaggle-offline-install_search.png)

3. ノートブックの一番最初のセルで、以下のコマンドを実行します。

```python
!pip install --no-index --find-links=/kaggle/input/datasets/aaaa1597/zarr-offline-installation-wheels/zarr_wheels_fixed zarr
```
これで通信を行わずにローカルマウントされたパスから安全にインストールが完了します。

### まとめ
インターネットオフの制約があるコンペでも、この方法であれば、どんなライブラリでも自由に使用できます。

お役に立てれば。

※補足
`zarr`なら、Notebookの先頭に下記コードを埋めるだけで、インストールが成功しますよ。是非使ってください。
```python
!pip install --no-index --find-links=/kaggle/input/datasets/aaaa1597/zarr-offline-installation-wheels/zarr_wheels_fixed zarr
```
