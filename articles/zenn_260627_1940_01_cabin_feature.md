---
title: "15-③[AI][Kaggle]Kaggle実践1 Cabinの特徴量エンジニアリング"
emoji: "🚢"
type: "tech"
topics:
  - "ai"
  - "python"
  - "kaggle"
  - "titanic"
published: true
published_at: "2026-06-28 17:45"
---

[Kaggle実践1『Titanic生存者予測』1.ローカルPCにKaggle Titanicの実行環境をつくる](https://zenn.dev/rg687076/articles/zenn_260627_0000_00_create_local_titanic_env)
[Kaggle実践1『Titanic生存者予測』2.初回提出](https://zenn.dev/rg687076/articles/zenn_260627_0000_01_first_submission)
[Kaggle実践1『Titanic生存者予測』3.Cabinの特徴量エンジニアリング](https://zenn.dev/rg687076/articles/zenn_260627_1940_01_cabin_feature)
[Kaggle実践1『Titanic生存者予測』4.特徴量エンジニアリング(ランダムフォレストによる年齢補完](https://zenn.dev/rg687076/articles/zenn_20260702_2031_age_imputation)
[Kaggle実践1『Titanic生存者予測』5.特徴量エンジニアリング(数値特徴量の非線形変換とビン化](https://zenn.dev/rg687076/articles/zenn_20260703_2025_fare_log_and_age_binning)
[Kaggle実践1『Titanic生存者予測』6.避難行動を捉えるグループ統計量の追加とCatBoost×Optuna自動最適化](https://zenn.dev/rg687076/articles/zenn_20260706_2030_catboost_optuna_tuning)

https://www.kaggle.com/c/titanic

← [Kaggle入門14(ゲームAIと強化学習入門)](https://zenn.dev/rg687076/articles/49e1d162bfdeec)
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;[Kaggle実践2(xxxx)](https://zenn.dev/rg687076/articles/xxxx) →

[github公開中](https://github.com/kito2718/KaggleTitanic)

# Abstract
- Cabinの頭文字から客室のデッキ(階層)情報を抽出して特徴量にしてみた
- LightGBM等のモデルで5-Fold CV(交差検証)を評価した結果のまとめ

# 概要
ひきつづきKaggleのタイタニックコンペから。
タイタニックのデータにはCabin(客室番号)があるんですが、これが70%以上欠損値だらけで外してました。
で、生存の可能性を探るにあたり、救命ボートへの距離とか沈没時の浸水スピードとかはデッキによって違った可能性があって、意味ある特徴量になりえるじゃないかと。

# 実装
Cabinの頭文字を抽出、欠損値は 'U' (Unknown) で埋めるように前処理関数を実装。
コードは下記。

```python
def feature_engineering(df):
    df = df.copy()
    # 既存の前処理(Title抽出やAgeの補完など)
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Cabinの頭文字をDeckとして抽出
    df['Deck'] = df['Cabin'].fillna('U').apply(lambda x: x[0])
    
    le = LabelEncoder()
    for col in ['Sex', 'Embarked', 'Title', 'Deck']:
        df[col] = le.fit_transform(df[col])
    return df
```

前処理した後にLabelEncoderで数値に変換して、モデルの入力特徴量に `Deck` を追加。

# 結果
5-Fold CVで評価した結果:

| モデル | 変更前(ベースライン) | 変更後(Cabin追加) | 変動 |
| :--- | :--- | :--- | :--- |
| Logistic Regression | 0.8014 +/- 0.0133 | 0.7991 +/- 0.0199 | -0.0023 |
| Random Forest | 0.8227 +/- 0.0077 | 0.8148 +/- 0.0149 | -0.0079 |
| XGBoost | 0.8181 +/- 0.0220 | 0.8227 +/- 0.0159 | +0.0046 |
| LightGBM | 0.8350 +/- 0.0178 | 0.8361 +/- 0.0278 | +0.0011 |

LightGBMの精度が 0.8350 から 0.8361 にちょっとだけ上がった。
決定木モデルのXGBoostでも精度が向上してるので、効果はありそう。


# Kaggle への提出
結果をタイタニックコンペに提出したら、スコアが 0.77272 → 0.77033 に微減した。単純な特徴量追加だけではノイズになっている可能性がありそう。

# まとめ
提出したら、スコアダウンとの結果を踏まえて、次はCabinをグループにした特徴量を試してみます。

お役に立てれば。

English version:
[Kaggle Practice 1: Setting Up a Local Environment for the Kaggle Titanic Competition](https://dev.to/kito2718/setting-up-kaggle-titanic-environment-on-a-local-pc-336k)
[Kaggle Practice 2: First Submission](https://dev.to/kito2718/kaggle-titanic-my-first-submission-eda-feature-engineering-and-model-evaluation-896)
[Kaggle Practice 3: Feature Engineering for Cabin](https://dev.to/kito2718/kaggle-titanic-cabin-feature-engineering-is-it-really-effective-44nc)
[Kaggle Practice 4: Feature Engineering (Imputing Age with Random Forest)](https://dev.to/kito2718/kaggle-titanic-improving-survival-prediction-with-random-forest-age-imputation-5b3l)
