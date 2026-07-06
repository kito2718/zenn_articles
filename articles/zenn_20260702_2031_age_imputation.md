---
title: "15-④[AI][Kaggle]Kaggle実践1 特徴量エンジニアリング(ランダムフォレストによる年齢補完)"
emoji: "🚢"
type: "tech"
topics:
  - "ai"
  - "python"
  - "kaggle"
  - "titanic"
published: true
published_at: "2026-07-02 20:31"
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
- 年齢(Age) の欠損値を敬称ごとの中央値補完 → RandomForestRegressor を用いた予測補完に変更。
- 5-Fold CV(交差検証) の最高スコアが 0.8507 → 0.8519 (Logistic Regression) に向上。
- KaggleのPublic Scoreが 0.78708 → 0.78947 にスコアアップ。
- 検証コードは github: [titanic_eda_20260702_2031_age_imputation.ipynb](https://github.com/kito2718/KaggleTitanic/blob/main/notebooks/titanic_eda_20260702_2031_age_imputation_.ipynb) にコミット。

# 概要
タイタニック生存者予測コンペにおいて、乗客の年齢 (Age) は生存に大きく関わる重要な要素です。
今までは敬称 (Title: Mr, Miss, Mrs, Master, Rare) ごとの中央値で欠損値を埋めてたのを、今回は他の特徴量 (Pclass, Sex, SibSp, Parch, Fare, Embarked, Deck) の情報を使って、機械学習モデル (RandomForestRegressor) で年齢を予測して補完するアプローチを試してみました。

# 実装
実装した前処理のコードは下記。
`RandomForestRegressor` を用いて、年齢が既知のデータで学習し、欠損している乗客の年齢を予測して埋めてる。

```python
from sklearn.ensemble import RandomForestRegressor

# 説明変数として使う特徴量を指定
age_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'Deck', 'FamilySize', 'IsAlone', 'Age']
df_age_prep = df_all[age_features].copy()

# カテゴリ変数をOne-Hotダミー変数化
cat_cols_for_age = ['Sex', 'Embarked', 'Title', 'Deck']
df_age_encoded = pd.get_dummies(df_age_prep, columns=cat_cols_for_age, drop_first=True)

# 訓練データと欠損データに分割
train_age = df_age_encoded[df_age_encoded['Age'].notnull()]
test_age = df_age_encoded[df_age_encoded['Age'].isnull()]

X_train_age = train_age.drop(columns=['Age'])
y_train_age = train_age['Age']
X_test_age = test_age.drop(columns=['Age'])

# 年齢予測用モデルの学習と予測
age_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
age_regressor.fit(X_train_age, y_train_age)
predicted_ages = age_regressor.predict(X_test_age)

# 元のデータフレームの欠損値で埋める
df_all.loc[df_all['Age'].isnull(), 'Age'] = predicted_ages
```

# 検証結果
5-Fold CV (交差検証) で各モデルを評価した結果のまとめ

| モデル | 変更前 (敬称別中央値補完) | 変更後 (RandomForest予測補完) | 変動 |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** | 0.8507 +/- 0.0104 | **0.8519 +/- 0.0115** | **+0.0012** |
| Random Forest | 0.8204 +/- 0.0193 | 0.8249 +/- 0.0348 | +0.0045 |
| XGBoost | 0.8215 +/- 0.0241 | 0.8226 +/- 0.0244 | +0.0011 |
| LightGBM | 0.8496 +/- 0.0211 | 0.8485 +/- 0.0147 | -0.0011 |

ロジスティック回帰 (Logistic Regression) において、最高CV精度が `0.8519` となり、自身最高値。
さらに決定木モデルの Random Forest や XGBoost でもスコア向上が見られた。

# Kaggleへの提出結果
このモデルでテストデータを予測し、Kaggleに提出。
結果、Public Score が **0.78708 から 0.78947 に向上**！
ローカルのCVスコアの改善が、しっかりとPublic Scoreの改善にも繋がっとることが確認できました。

# まとめと今後の展望
年齢を単純な中央値ではなく、他の情報から推測して埋めることで、よりリアルな乗客像をモデルに学習させることができたと思う。

お役に立てれば。

English version:
[Kaggle Practice 1: Setting Up a Local Environment for the Kaggle Titanic Competition](https://dev.to/kito2718/setting-up-kaggle-titanic-environment-on-a-local-pc-336k)
[Kaggle Practice 2: First Submission](https://dev.to/kito2718/kaggle-titanic-my-first-submission-eda-feature-engineering-and-model-evaluation-896)
[Kaggle Practice 3: Feature Engineering for Cabin](https://dev.to/kito2718/kaggle-titanic-cabin-feature-engineering-is-it-really-effective-44nc)
[Kaggle Practice 4: Feature Engineering (Imputing Age with Random Forest)](https://dev.to/kito2718/kaggle-titanic-improving-survival-prediction-with-random-forest-age-imputation-5b3l)
