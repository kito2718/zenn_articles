---
title: "15-②[AI][Kaggle]Kaggle実践1 初回提出(EDA→特徴量エンジニアリング→モデル評価(5つ))"
emoji: "🦁"
type: "tech"
topics:
  - "ai"
  - "python"
  - "kaggle"
published: true
published_at: "2026-06-28 11:17"
---

[Kaggle実践1『Titanic生存者予測』1.ローカルPCにKaggle Titanicの実行環境をつくる](https://zenn.dev/rg687076/articles/zenn_260627_0000_00_create_local_titanic_env)
[Kaggle実践1『Titanic生存者予測』2.初回提出](https://zenn.dev/rg687076/articles/zenn_260627_0000_01_first_submission)
[Kaggle実践1『Titanic生存者予測』3.Cabinの特徴量エンジニアリング](https://zenn.dev/rg687076/articles/zenn_260627_1940_01_cabin_feature)
[Kaggle実践1『Titanic生存者予測』4.特徴量エンジニアリング(ランダムフォレストによる年齢補完](https://zenn.dev/rg687076/articles/zenn_20260702_2031_age_imputation)
[Kaggle実践1『Titanic生存者予測』5.特徴量エンジニアリング(数値特徴量の非線形変換とビン化](https://zenn.dev/rg687076/articles/zenn_20260703_2025_fare_log_and_age_binning)
[Kaggle実践1『Titanic生存者予測』6.避難行動を捉えるグループ統計量の追加とCatBoost×Optuna自動最適化](https://zenn.dev/rg687076/articles/zenn_20260706_2030_catboost_optuna_tuning)

https://www.kaggle.com/c/titanic

← [Kaggle入門14(ゲームAIと強化学習入門)](https://zenn.dev/rg687076/articles/49e1d162bfdeec)
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;[Kaggle実践2(Text-to-Image)](https://zenn.dev/rg687076/articles/zenn_20260709_2244_baseline_sd_yolo) →

[github公開中](https://github.com/kito2718/KaggleTitanic)

# Abstract
- Kaggleのタイタニックコンペに参加、データを探索的に分析(EDA)した
- 敬称(Title)の抽出、家族人数(FamilySize)などの基本的な特徴量エンジニアリングを実施
- モデル生成を5つのロジック(ロジスティック回帰、ランダムフォレスト、XGBoost、LightGBM)で試し、5-Fold CVで比較
- LightGBMが一番良かった。スコア0.77272

# 概要
引き続き、タイタニックのデータ分析。
欠損値の確認や生存率に関わりそうな変数(性別や客室クラスなど)を可視化(EDA)して考察。
その後、名前から敬称（Title）を抽出、家族の人数から新しい特徴量(家族サイズ)を作ったりで、5つのモデルで交差検証(5-Fold CV)をやってみた、その一連の流れのまとめ。

# データ探索(EDA)
タイタニックのデータは訓練データが891行、テストデータが418行あるのだけど、欠損値の存在がめんどい。

- **Cabin（客室番号）**: 77.1%（ほとんど欠損）
- **Age（年齢）**: 19.9%（約2割が欠損）
- **Embarked（乗船港）**: 0.2%（2件だけ欠損）

## 可視化
まずは基本的な変数と生存率の関係ばグラフにプロットしてみた。

### 1. 生存率と性別・客室クラスの関係
性別を客室クラス(Pclass)ごとの生存率がアツい。女性の生存率が圧倒的に高くて、Pclassも1等客室(1)の生存率が一番高いことが分かる。

![生存率と性別・客室クラスの関係](https://static.zenn.studio/user-upload/0d803e9b5315-20260628.png)

### 2. 年齢分布
年齢全体のヒストグラムと、生存・死亡別の年齢分布をプロットしてみた。若い年齢(特に乳幼児)の生存率が高くて、特定の年齢層で死亡率が高い傾向がみてとれる。

![年齢分布のヒストグラム](https://static.zenn.studio/user-upload/c95d3e0068ea-20260628.png)

### 3. 相関関係
数値特徴量同士の相関係数をヒートマップで確認してみた。Pclassと運賃(Fare)の間に強い負の相関があることが見て取れる。

![相関係数のヒートマップ](https://static.zenn.studio/user-upload/cd4aebddf5ca-20260628.png)

相関の算出式は下記っす。
$$r = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^{n} (y_i - \bar{y})^2}}$$

共分散と標準偏差の言葉を使ってシンプル化すると
$$r = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$
$\text{Cov}(X, Y)$： $X$ と $Y$ の共分散。2つの変数がどれだけ一緒に動くか(一方が増えたときにもう一方も増えるかの指標)。
$\sigma_X, \sigma_Y$（分母）： それぞれの変数の標準偏差。データのばらつきの大きさのこと。

# 前処理と特徴量エンジニアリング
EDAの結果を踏まえて、特徴量エンジニアリングと前処理を行う関数 `feature_engineering` を作成してみた。

1. **Title(敬称)の抽出**: Nameから Mr, Miss, Mrs, Master などの敬称を正規表現で抽出、代表値にマッピング。少数の敬称は `Rare` に。
2. **Age(年齢)の補完**: 全体の代表値(中央値など)じゃなくて、抽出した敬称(Title)ごとの中央値で補完。これで、子供(Master)や大人(Mr/Mrs)の属性に合わせた適切な年齢で埋めることができると期待。
3. **FamilySize(家族人数)とIsAlone(単身フラグ)**: 兄弟・配偶者数(SibSp)と親子数(Parch)に自分自身(1)を足して `FamilySize` 特徴量を新規作成。で、家族人数が1人の場合は `IsAlone`特徴量を追加1に設定。
4. **その他の補完**: `Fare` は中央値、`Embarked` は最頻値で補完。
5. **カテゴリ変数のエンコーディング**: `Sex`, `Embarked`, `Title` は `LabelEncoder` で数値に変換した。

コードは下記。

```python
def feature_engineering(df):
    df = df.copy()
    # Nameから敬称（Title）を抽出
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    title_map = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
        'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
        'Capt': 'Rare', 'Sir': 'Rare'
    }
    df['Title'] = df['Title'].map(title_map).fillna('Rare')
    
    # 敬称グループごとの年齢中央値で補完
    df['Age'] = df.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
    
    # 家族サイズ特徴量
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # 運賃と乗船港の補完
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # カテゴリ変数のラベルエンコーディング
    le = LabelEncoder()
    for col in ['Sex', 'Embarked', 'Title']:
        df[col] = le.fit_transform(df[col])
    return df
```

この前処理を実施後、採用した特徴量は10個。
`['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone']`。

# モデル評価と結果
作成した特徴量で、ロジスティック回帰(Logistic Regression)、ランダムフォレスト(Random Forest)、XGBoost、LightGBMの4つのモデルで Stratified 5-Fold CV(層化5分割交差検証)で比較してみた。

検証コードは下記。

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost':             xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0),
    'LightGBM':           lgb.LGBMClassifier(n_estimators=100, random_state=42, verbosity=-1, max_depth=3, num_leaves=7, learning_rate=0.05)
}
results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    results[name] = scores
    print(f'{name:<25}: {scores.mean():.4f} +/- {scores.std():.4f}')
```

で、各モデルのCVスコア平均のまとめ。
| モデル | 5-Fold CV スコア(Accuracy) |
| :--- | :--- |
| Logistic Regression | 0.8014 +/- 0.0133 |
| Random Forest | 0.8227 +/- 0.0077 |
| XGBoost | 0.8181 +/- 0.0220 |
| LightGBM | 0.8350 +/- 0.0178 |

交差検証の結果のばらつきを箱ひげ図でプロットした結果。

![モデル精度比較](https://static.zenn.studio/user-upload/04350e7741f2-20260628.png)

LightGBMが **0.8350** で一番良い結果となった。
ランダムフォレストも0.8227で結構よかった。

## 特徴量重要度(Feature Importance)
一番スコアの良かったLightGBMモデルの特徴量重要度を可視化してみた。

![特徴量重要度](https://static.zenn.studio/user-upload/e3613eb19157-20260628.png)

「性別(Sex)」、「敬称(Title)」、「運賃(Fare)」、「年齢(Age)」が予測するのに凄く重要視してるのがわかる。

# Kaggleへの初提出
LightGBMの予測を提出ファイル `submission.csv` として作成。

```python
preds = best_model.predict(X_test)
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': preds})
submission.to_csv('../submissions/submission.csv', index=False)
```

スコア: **0.77272** 。
しれっと自己最高。
CVスコア(0.8350)に比べるとちょっと下がってしまうのはそんなもんらしい。

# まとめ
EDA→特徴量エンジニアリング→モデル評価(5つ)の手順を実施してみた。特に、モデル評価(5つ)ってのはだいぶ意味ありそう。
ここからさらにスコアアップのために、次は欠損値だらけの「Cabin(客室番号)」を特徴量エンジニアリングします。

お役に立てれば。

English version:
[Kaggle Practice 1: Setting Up a Local Environment for the Kaggle Titanic Competition](https://dev.to/kito2718/setting-up-kaggle-titanic-environment-on-a-local-pc-336k)
[Kaggle Practice 2: First Submission](https://dev.to/kito2718/kaggle-titanic-my-first-submission-eda-feature-engineering-and-model-evaluation-896)
[Kaggle Practice 3: Feature Engineering for Cabin](https://dev.to/kito2718/kaggle-titanic-cabin-feature-engineering-is-it-really-effective-44nc)
[Kaggle Practice 4: Feature Engineering (Imputing Age with Random Forest)](https://dev.to/kito2718/kaggle-titanic-improving-survival-prediction-with-random-forest-age-imputation-5b3l)

