---
title: "Kaggle実践4 ランダムフォレストによる年齢補完でタイタニック生存予測の精度向上ばい"
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

# Abstract
- 年齢 (Age) の欠損値を敬称ごとの中央値補完から RandomForestRegressor を用いた高度な予測補完に変更したばい。
- 5-Fold CV (交差検証) の最高スコアが 0.8507 から 0.8519 (Logistic Regression) に向上したばい。
- KaggleのPublic Scoreも 0.78708 から 0.78947 にアップしたばい。
- 検証コードはノートブック [titanic_eda_20260702_2031_age_imputation.ipynb](file:///d:/BizOwn/000_Biw2/51_googleantigravity/1st_/notebooks/titanic_eda_20260702_2031_age_imputation.ipynb) にまとめてコミットしとるばい。

# 概要
タイタニック生存者予測コンペにおいて、乗客の年齢 (Age) は生存に大きく関わる重要な要素やね。
これまでは敬称 (Title: Mr, Miss, Mrs, Master, Rare) ごとの中央値で欠損値を埋めとったばってん、今回は他の特徴量 (Pclass, Sex, SibSp, Parch, Fare, Embarked, Deck) の情報も交えて、機械学習モデル (RandomForestRegressor) で年齢を予測して補完するアプローチを試したばい。

# 実装
実装した前処理のコードは以下やね。
`RandomForestRegressor` を用いて、年齢が既知のデータで学習し、欠損している乗客の年齢を予測して埋めるようにしたばい。

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

# 元のデータフレームの欠損値を埋めるばい
df_all.loc[df_all['Age'].isnull(), 'Age'] = predicted_ages
```

# 検証結果
5-Fold CV (交差検証) で各モデルを評価した結果が以下やね。

| モデル | 変更前 (敬称別中央値補完) | 変更後 (RandomForest予測補完) | 変動 |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** | 0.8507 +/- 0.0104 | **0.8519 +/- 0.0115** | **+0.0012** |
| Random Forest | 0.8204 +/- 0.0193 | 0.8249 +/- 0.0348 | +0.0045 |
| XGBoost | 0.8215 +/- 0.0241 | 0.8226 +/- 0.0244 | +0.0011 |
| LightGBM | 0.8496 +/- 0.0211 | 0.8485 +/- 0.0147 | -0.0011 |

ロジスティック回帰 (Logistic Regression) において、最高CV精度が `0.8519` となり、ベースラインを更新したばい！
さらに決定木モデルの Random Forest や XGBoost でもスコア向上が見られたばい。

# Kaggleへの提出結果
このモデルでテストデータを予測し、Kaggleに提出したばい。
結果、Public Score が **0.78708 から 0.78947 に向上** したがばい！
ローカルのCVスコアの改善が、しっかりとPublic Scoreの改善にも繋がっとることが確認できて嬉しいやね。

# まとめと今後の展望
年齢を単純な中央値ではなく、他の情報から推測して埋めることで、よりリアルな乗客像をモデルに学習させることができたばい。
次はさらなるスコアアップに向けて、ハイパーパラメータの自動チューニング (Optuna) やアンサンブル学習を試していこうと思うばい。

お役に立てれば嬉しいばい。
