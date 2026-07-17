---
title: "18-①Kaggle実践3 Biohub細胞トラッキング：環境構築から初回提出までの手順"
emoji: "🔬"
type: "tech"
topics: ["kaggle", "python", "bioinformatics", "celltracking"]
published: true
series: "Biohub細胞トラッキング挑戦記"
tags: ["kaggle", "python", "bioinformatics", "cell-tracking"]
---

[![アイキャッチ画像](https://github.com/kito2718/zenn_articles/raw/main/articles/images/zenn_20260714_0630_bitcelltracking_eyecatch_.png =500x)](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
*Biohub - Cell Tracking During Development*

Kaggle実践3『Biohub細胞トラッキング』1.Kaggle Notebookに実行環境をつくる(この記事)

## Abstruct
- Kaggleの生物画像コンペティション「Biohub - Cell Tracking During Development」に参加
- 環境構築～初回提出(Submit)までの手順まとめ
- windows 11

## 概要
初の開催中コンペに挑みます。生命科学とか全くやったことがありませんが、社会の役に立つ気が猛烈にしています。たまーにNHKのドキュメンタリーでも、細胞の分裂や移動をの映像を見たことがあって、そこに役立つスキルのはず。このコンペティションは、本質的には3D顕微鏡画像の時間変化データから細胞の位置(ノード)を検出し、フレーム間でそれらを結びつけるタスクということです。モチベーションとしては、「AIがどう関係すんの？」てムズそうなのをやり切って、自分のスキルアップに繋げたい。

データサイズが非常に大きいです。でもKaggle Notebooksは無料GPU枠を提供してくれてて、そこに開発環境を構築しました。ローカルPC環境もおいおい必要に応じて構築する予定です。

## Kaggle Notebook環境構築手順

### 開発環境のセットアップと事前準備

本コンペティションに参加するには、いくつかの事前準備が必要です。

1. **コンペティション規約への同意**
   [Biohub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)のページにアクセスし、「Join Competition」ボタンを押して規約に同意します。これを行わないと、データのダウンロードや提出ができません。

2. **Kaggle APIトークンの取得**
   Kaggleアカウントの設定ページ(Settings)から「Create New Token」をクリックし、`kaggle.json`ファイルをダウンロードします。ローカル環境から提出する際や、CLIを使う際に必要となります。置き場所はユーザー配下でいいかと。
```tree
C:\user\xxx\.kaggle\kaggle.json
```

3. Kaggle Notebookを作成
[Codeタブ](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/code) → NewNotebookを押下して生成します。  
![](https://github.com/user-attachments/assets/bfbb4995-a225-4078-b74b-1db9d19ddb59 =500x)

4. Notebookの中身を下記に公開しています。Copy and Editから始めると簡単です。  
https://www.kaggle.com/code/aaaa1597/biocell-track-by-hgs

:::message alert
"import zarr"でエラー!!
kaggleには`zarr`がないので、別途インストールする必要があります。ですが提出時、internet off って設定にする必要があって、これが原因で、Notebookの方に `!pip install zarr`を記述することができません(提出時にエラーになるため)。

そのあたりを↓にまとめました。
[(Kaggle提出のための)Offline環境でプリインストールされていないライブラリをインストールする方法](https://zenn.dev/rg687076/articles/zenn_20260714_0630_kaggle-offline-install)
見てね。
:::

### Run All
Notebookが準備できたら、実行してエラーが発生しないかを確認します。
Run Allを押下しましょう。
![](https://github.com/user-attachments/assets/c6ae8c36-fab1-4285-b6f7-0e6243c2db73 =500x)

経過は左下にPOPUPで表示されます。10分ぐらいかかるので一息入れて待ちましょう。
![](https://github.com/user-attachments/assets/7086ad23-32de-4b27-ac67-5a6c90a72c83 =500x)

### Save Version
Run Allが正常完了したら、Save Versionで保存しましょう。
![](https://github.com/user-attachments/assets/87a8733a-e149-42db-80f9-4ba639aafe54 =500x)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  ↓
![](https://github.com/user-attachments/assets/6cf719a5-a745-4b33-a21c-0639119b7558 =500x)
また、Run Allが実行されます。経過は左下にPOPUPで表示されますので、気長に待ちましょう。

### Submit to competition
やっと提出できるようになりました。Submit to competitionを開いてSubmitを押下します。
![](https://github.com/user-attachments/assets/e69ab6d3-4d76-47a0-924b-644b255b0ad5 =500x)
入力欄はデフォルトでいいです。Submitを押下しましょう。
![](https://github.com/user-attachments/assets/ea485f6a-f5eb-48a1-bf81-dddb338c0ff5 =500x)
提出までできました。

### 結果を確認する。
[submissions](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/submissions)にアクセスします。
下図のVersion 7が成功した状態です。
Version 6は失敗ですね。この時は提出ファイルのフォーマットが合ってなくてエラーになりました。修正してまたRun Allからやり直したら成功しました。
![](https://github.com/user-attachments/assets/37283caa-fff3-4736-8e23-820903cb93ae =500x)
これで、提出まで完了です。
ここからスコアアップのサイクルが始まります。
頑張ってください!!

### まとめ
これで、まずは正常に提出が通ることを確認できました。これをベースに、より高度な3D細胞検出モデル(CellposeやStarDistなど)の適用や、カルマンフィルター、ネットワークフローを用いたトラッキング手法の改善に進むことができます。

### 初回提出コード
初回提出コードは下記です。スコア0.505ですね。
@[gist](https://gist.github.com/kito2718/2bbb9dec9908464191eb008643fc33c8)

お役に立てれば。
