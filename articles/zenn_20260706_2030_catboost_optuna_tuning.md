---
title: "CatBoostとOptunaチューニングでタイタニック生存予測のCVスコア過去最高0.8563を達成しました"
emoji: "🚢"
type: "tech"
topics: ["kaggle", "titanic", "catboost", "optuna", "machinelearning"]
published: false
---

[Kaggle実践1(『Titanic生存者予測』1.ローカルPCにKaggle Titanicの実行環境をつくる](https://zenn.dev/rg687076/articles/zenn_260627_0000_00_create_local_titanic_env)
[Kaggle実践1(『Titanic生存者予測』2.初回提出](https://zenn.dev/rg687076/articles/zenn_260627_0000_01_first_submission)
[Kaggle実践1(『Titanic生存者予測』3.Cabinの特徴量エンジニアリング](https://zenn.dev/rg687076/articles/zenn_260627_1940_01_cabin_feature)
[Kaggle実践1(『Titanic生存者予測』4.特徴量エンジニアリング(ランダムフォレストによる年齢補完)](https://zenn.dev/rg687076/articles/zenn_20260702_2031_age_imputation)
[Kaggle実践1(『Titanic生存者予測』5.特徴量エンジニアリング(数値特徴量の非線形変換とビン化)](https://zenn.dev/rg687076/articles/zenn_20260703_2025_fare_log_and_age_binning)
[Kaggle実践1(『Titanic生存者予測』6.特徴量エンジニアリング(グループ特徴量の高度化)とCatBoostの導入とOptunaチューニング)](https://zenn.dev/rg687076/articles/zenn_20260706_2030_catboost_optuna_tuning.md)

https://www.kaggle.com/c/titanic

← [Kaggle入門14(ゲームAIと強化学習入門)](https://zenn.dev/rg687076/articles/49e1d162bfdeec)
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;[Kaggle実践2(xxxx)](https://zenn.dev/rg687076/articles/xxxx) →

[github公開中](https://github.com/kito2718/KaggleTitanic)

# Abstract
* 姓・チケット番号をもとにした同乗グループ統計量特徴量(Group_Size, Group_Female_Child_Ratio等)を追加。
* カテゴリ変数に強い CatBoost Classifier を追加導入し、Optuna にてハイパーパラメータを自動最適化。
* 検証スコア(5-Fold CV Accuracy)が **0.8563** に向上したものの、Public Score は **0.79665** のまま。

# 概要
今回は特徴量設計の強化と新モデル(CatBoost)の導入、パラメータチューニング(Optunaによる自動調整)を組み合わせて精度向上を実施。

## 1. グループ特徴量の高度化
タイタニック号の避難行動では「家族や同伴グループで行動を共にした」乗客が多かったらしいとのことなので、チケット番号や姓が同じ乗客を同一グループに分類した。
具体的には、以下のグループ統計量を新規に作成。

1. **グループサイズ (`Group_Size`)**: チケットや姓・運賃で同定した実質的なグループ同伴人数。戸籍上の家族数を表す `Family_Size` と異なり、友人や主従関係などの実質的な行動グループを捉えることが目的です。
2. **グループ内女性・子供比率 (`Group_Female_Child_Ratio`)**: グループ内における優先救助対象(女性または16歳未満の子供)の割合。
3. **グループ平均年齢 (`Group_Mean_Age`)**: グループ全体の平均年齢。
4. **グループ運賃中央値差分 (`Group_Fare_Median_Diff`)**: 同一クラス(Pclass)の運賃中央値とグループ運賃の差分。

### 設計したグループ特徴量の可視化と考察
それぞれの特徴量と生存率の関係を、個別グラフに分けて見てみた。

#### ① グループサイズと生存率の関係 (`Group_Size`)
![グループサイズと生存率](https://github.com/kito2718/zenn_articles/raw/main/articles/images/group_size_survival.png)

* **見方**: 横軸はグループの人数、縦軸はそのグループに属する乗客の生存率。
* **考察**: 2〜4人の中規模グループは生存率が 50〜70% と非常に高くなっているのに対して、1人(単身)や5人以上の大家族グループは生存率が下がってる。緊急時に協力し合える適度なグループサイズが生存に有利に働いたと想定できそう。

#### ② グループ内の女性・子供比率の分布 (`Group_Female_Child_Ratio`)
![グループ内女性・子供比率](https://github.com/kito2718/zenn_articles/raw/main/articles/images/group_female_child_ratio_survival.png)

* **見方**: 横軸はグループ内の女性・子供の割合(0.0〜1.0)、縦軸は確率密度。赤い山が「死亡した乗客(Deceased)」、青い山が「生存した乗客(Survived)」の分布。
* **考察**:
  * **左端 (0.0付近)**: 女性・子供比率が 0.0（全員大人の男性のグループ）は、死亡の赤い山が圧倒的に高い。避難優先度が最も低かったためと思われ。
  * **右端 (1.0付近)**: 女性・子供比率が 1.0（全員女性・子供のグループ）は、生存の青い山が高い。最優先で救助されたためと思われ。
  * **ポイント**: 個人単体では「生存率の低い大人の男性」であっても、女性や子供が多いグループに属していれば、一緒にボートに案内されて生存できた可能性が高くなるという「グループ単位の運命」をモデルに学習させることができそう。

#### ③ グループの平均年齢の分布 (`Group_Mean_Age`)
![グループ平均年齢](https://github.com/kito2718/zenn_articles/raw/main/articles/images/group_mean_age_survival.png)

* **見方**: 横軸はグループの平均年齢、縦軸は確率密度。赤い山が「死亡した乗客(Deceased)」、青い山が「生存した乗客(Survived)」。
* **考察**: 生存グループ (青い山) は、死亡グループ (赤い山) に比べて「平均年齢がやや若い側 (特に子供が含まれる 20代以下)」と「35〜40代の中堅家族リーダー層」の2つでDeceasedを上回る。グループ全体の年齢層の若さや構成も避難行動に影響したと考えられる。

#### ④ 客室クラス内での運賃乖離度 (`Group_Fare_Median_Diff`)
![グループ運賃乖離度](https://github.com/kito2718/zenn_articles/raw/main/articles/images/group_fare_median_diff_survival.png)

* **見方**: 生存(1)・死亡(0)ごとに、同一客室クラスの運賃中央値からどのくらい運賃が乖離していたかを箱ひげ図で表示。
* **考察**: 生存した乗客 (1) の方が、死亡した乗客 (0) に比べて、中央値より高い運賃を払っているグループに属していた傾向が見れる。同じ客室クラス内であっても、より好立地（避難経路に近い、または上層階など）の部屋に配置されていた可能性が高そう。

### 実装コード
特徴量エンジニアリングでの具体的な実装コードは以下です。

```python
# --- 高度なグループ特徴量の作成 ---
# Ticketが同じ、または姓と運賃が同じ乗客を同一グループ(Group_Id)として定義します
df_all['Ticket_Group_Size'] = df_all.groupby('Ticket')['PassengerId'].transform('count')
df_all['Group_Id'] = df_all['Ticket']
mask = df_all['Ticket_Group_Size'] == 1
df_all.loc[mask, 'Group_Id'] = df_all.loc[mask, 'Last_Name'] + '_' + df_all.loc[mask, 'Fare'].astype(str)

# 1. グループサイズ
df_all['Group_Size'] = df_all.groupby('Group_Id')['PassengerId'].transform('count')

# 2. グループ内女性・子供比率
df_all['Is_Female_or_Child'] = ((df_all['Sex'] == 'female') | (df_all['Age'] < 16)).astype(int)
df_all['Group_Female_Child_Ratio'] = df_all.groupby('Group_Id')['Is_Female_or_Child'].transform('mean')

# 3. グループ平均年齢
df_all['Group_Mean_Age'] = df_all.groupby('Group_Id')['Age'].transform('mean')

# 4. 客室クラス中央値からの運賃乖離度
pclass_fare_median = df_all.groupby('Pclass')['Fare'].transform('median')
df_all['Group_Fare_Median_Diff'] = df_all['Fare'] - pclass_fare_median

# 不要な中間列の削除
df_all = df_all.drop(columns=['Ticket_Group_Size', 'Is_Female_or_Child'])
```

## 2. CatBoostの導入とOptunaチューニング
タイタニックデータセットは `Sex`, `Embarked`, `Title`, `Deck` などカテゴリ変数が非常に多いので、カテゴリ変数の高度な処理(ターゲット統計量など)を内部で自動的に行う勾配ブースティングライブラリ **CatBoost** (CatBoostClassifier) を新規導入。

さらに、これまでパラメータを調整していなかった LightGBM や XGBoost も含め、`optuna` ライブラリを使って 5-Fold CV Accuracy を指標にパラメータ自動最適化を実施。

### パラメータチューニングの実装コード
Optunaを用いたパラメータ自動チューニングの実装コードは以下。各モデルで 30 trials の探索を行い、クロスバリデーションスコアが最も高くなるパラメータを探索するように実装。

```python
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
import lightgbm as lgb
from catboost import CatBoostClassifier

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- LightGBM のチューニング定義 ---
def objective_lgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'num_leaves': trial.suggest_int('num_leaves', 7, 63),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'verbosity': -1,
        'random_state': 42
    }
    model = lgb.LGBMClassifier(**params)
    return cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy').mean()

# 最適化の実行
study_lgb = optuna.create_study(direction='maximize')
study_lgb.optimize(objective_lgb, n_trials=30)
print(f"Best LightGBM CV Score: {study_lgb.best_value:.4f}")

# --- CatBoost のチューニング定義 ---
def objective_cat(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 50, 300),
        'depth': trial.suggest_int('depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'random_seed': 42,
        'verbose': 0
    }
    model = CatBoostClassifier(**params)
    return cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy').mean()

# 最適化の実行
study_cat = optuna.create_study(direction='maximize')
study_cat.optimize(objective_cat, n_trials=30)
print(f"Best CatBoost CV Score: {study_cat.best_value:.4f}")
```

探索したパラメータ範囲と最適パラメータによる結果です。

* **LightGBM**: CV `0.8530` (向上)
* **XGBoost**: CV `0.8530` (大幅向上)
* **CatBoost**: CV **0.8563** (最高CV更新！)
  * 最適パラメータ: `iterations=186, depth=4, learning_rate=0.068, l2_leaf_reg=5.63`

## 3. 検証結果と提出スコア

| モデル | 変更前CV (年齢予測補完ベースライン) | 特徴量追加後CV | パラメータ最適化後CV | Kaggle Public Score |
| :--- | :---: | :---: | :---: | :---: |
| Logistic Regression | **0.8519** | 0.8485 | - | - |
| LightGBM | 0.8485 | 0.8518 | 0.8530 | 0.79425 (ステップ1) |
| XGBoost | 0.8226 | 0.8272 | 0.8530 | - |
| **CatBoost** | - | - | **0.8563** | **0.79665** (変わらず!) |

スタッキングによるアンサンブル(Meta: Ridge Classifier, CV: 0.8485)も試したけど、データサイズが小さいためかメタモデルの過学習が起きてるようで、今回は単体で高度にチューニングした CatBoost が最良のスコアでした。

# まとめ
グループ特徴量の高度化と、CatBoost × Optuna チューニングの組み合わせによって、CV スコアはアップ **0.8563** した。
けど、Kaggle の Public Score は **0.79665** の変化なし。やって意味あったのかと思う結果になりました。

お役に立てれば。
