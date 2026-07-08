---
title: "15-⑧[AI][Kaggle]Kaggle実践1(最終回): Kaggle Titanicで過学習と戦う！グループ生存情報と最強正則化アンサンブル"
emoji: "ship"
type: "tech"
topics: ["kaggle", "machinelearning", "ensemble", "python", "optuna"]
published: false
---

もうここら辺が限界かな～。いろいろやってもスコアアップしないや。

# Abstract

* CVスコアは高い(0.868超え)のに本番(Public Score)が下がる過学習の問題へ対応を頑張ってみた。
* グループ生存特徴量ば完全排除したアプローチで、Public Score が 0.77033 へ急落。グループ情報の重要性(遭難時に運命ば共にする物理的事実)が逆説的に実証された。
* なので、グループ特徴量をフル活用しつつ、決定木の深さを3〜4に厳しく制限、かつ、L2正則化を通常値の10倍以上に高めた「超・強正則化モデル」を構築。
* だけど結果は、Public Score は 0.79425。↓

# 概要:
**Titanicに潜むCVスコアの罠**

Kaggle の Titanic チュートリアルでは、「同じグループや家族の生存情報」をリークなしで計算してモデルに与える手法は、非常に強力なシグナルらしい。
ということで、これを頑張って、CVスコアアップ、でも、Public Scoreダウンの現象に直面した。
* OOF(Out-of-Fold)でのCVスコア **0.86869** (↑過去最高)
* Kaggle Public Score: **0.78229** (↓)

訓練データ内ではいいスコアでも、テストデータ(Survivedが未知)に対して適用した瞬間に、モデルが訓練時の「生存グループのルール」に過剰適合(過学習)してしもうて、本番スコアが下がるという。

この過学習、どう抑え込むかをいろいろ検証してみた。

# 検証1: グループ情報の完全排除は正解か？

※グループ情報 ... 「同じグループや近くの客室にいた他の乗客たちの生存状況(誰が助かって誰が亡くなったか)」から切り出した情報のこと
具体的には下記。
- Family_Survival ... 同じ「家族・同乗グループ」のメンバーが生存したかどうかの情報
- OOF_Ticket_Neighbor_Survival ... チケット番号が近い(＝客室が隣近所で、同じ避難ボート乗り場や避難経路などの「客室エリアグループ」にいた)乗客たちの生存情報
- WCG (Woman-Child-Group) ... 「同一チケットグループ」の中の優先救助対象である女性や子供たちの生存情報

考え方としては、「グループ情報が過学習のノイズになるなら、いっそ全部消して個人属性だけで勝負すればいいいじゃね！」という考え方。

### 検証内容
* `Family_Survival` およびチケット近接生存率特徴量(OOF_Ticket_Neighbor_Survival)をドロップ。
* 代わりに、個人の「性別 × 客室クラス」「年齢 × 客室クラス」「運賃のクラス内偏差」といった非線形な交互作用特徴量を掛け合わせ。
  1. 性別 × 客室クラス (Sex_Pclass) ... 意図としては「女性は助かりやすい」「3等は助かりにくい」という単体ルールじゃなく、「1等の男性(男性だけど避難が優先された)」や「3等の女性(女性だけｄ避難経路が遠くて助かりにくかった)」といった、性別と階級による生存確率の歪みを抽出したい。
  2. 年齢 × 客室クラス (Age_Pclass) ... 意図としては「3等の高齢者(体力がなく避難が最も困難)」や「1等の子供(最優先で救命ボートに乗せられた)」といった、年齢と客室位置の非線形な関係を抽出したい。
  3. 客室クラス内の相対運賃 (Fare_Pclass_Normalized) ... 意図としては、同じ「運賃30ドル」でも、3等客室の中では「超富裕層(最上級部屋)」だけど、1等客室の中では「最安値(貧民部屋)」になる。額面通りの数値では意味がブレるので、各客室クラス(1〜3等)ごとに運賃の平均と標準偏差を算出し、そのクラス内での相対的な富裕度(Z-score / クラス内偏差値)をモデルに学習させたい。
  4. 敬称 × 年齢 (Title_Age_Interaction) ... 意図としては、「Mr(成人男性)の40代」と「Master(男児)の10代」では、同じ男性で年齢が近くても生存期待値が全く異なるため、社会的役割を表す「敬称」と「実年齢」を掛け合わせることで、優先救助の判断基準をよりシャープにモデルに学習させたい。
* 7つのベースモデル(LightGBM, XGBoost, CatBoost, ロジスティック回帰, SVM, KNN, MLP)ばブレンド。

```python
# --- 2. 個人属性の交互作用特徴量 (Interaction Features) ---
df_all['Sex_Pclass'] = (df_all['Sex'] == 'male').astype(int) * df_all['Pclass']
df_all['Age_Pclass'] = df_all['Age'] * df_all['Pclass']

# Fare の各客室クラス内偏差値 (正規化)
df_all['Fare_Mean_Class'] = df_all.groupby('Pclass')['Fare'].transform('mean')
df_all['Fare_Std_Class'] = df_all.groupby('Pclass')['Fare'].transform('std')
df_all['Fare_Pclass_Normalized'] = (df_all['Fare'] - df_all['Fare_Mean_Class']) / df_all['Fare_Std_Class']
df_all['Fare_Pclass_Normalized'] = df_all['Fare_Pclass_Normalized'].fillna(0)
df_all = df_all.drop(columns=['Fare_Mean_Class', 'Fare_Std_Class'])

# Title x Age 交互作用
df_all['Title_Mr_Age'] = (df_all['Title'] == 'Mr').astype(int) * df_all['Age']
df_all['Title_Mrs_Age'] = (df_all['Title'] == 'Mrs').astype(int) * df_all['Age']
df_all['Title_Miss_Age'] = (df_all['Title'] == 'Miss').astype(int) * df_all['Age']
df_all['Title_Rare_Age'] = (df_all['Title'] == 'Rare').astype(int) * df_all['Age']
```

### 結果
* **Weighted Blending CV Accuracy**: **0.84175**
* **Kaggle Public Score**: **0.77033** (大幅ダウン)

### 考察
グループ生存情報を排除すると、大幅スコアダウン。グループ生存情報は必要だったということが分かった。
これは、「遭難時にチケットや家族が一緒の乗客は運命を共にする」という事実が、Titanicではかなりいい特徴量であるということが分かった。
**ここの結論**
グループ生存情報は完全排除するのではなく、使いつつモデル側で過学習を抑え込むのが正解！

# 検証2: 「グループ特徴量 ＋ 超・強正則化」のアンサンブル

グループ情報が意味あるってのがわかったので、グループ情報をフル活用した状態で、モデルが「生存ルール」に過剰に依存するのを防ぐため、決定木モデルの複雑さを制限する「強正則化アプローチ」を検証してみた。

```python
# --- 9. 各ベースモデルの強正則化パラメータ最適化 (Optuna) ---

# LightGBM Tuning (木の深さ:3〜4, L2正則化:10.0〜100.0)
def objective_lgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 4), 
        'num_leaves': trial.suggest_int('num_leaves', 7, 15), 
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08),
        'min_child_samples': trial.suggest_int('min_child_samples', 15, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 1.0, 50.0), 
        'reg_lambda': trial.suggest_float('reg_lambda', 10.0, 100.0), 
        'subsample': trial.suggest_float('subsample', 0.5, 0.7), 
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.7), 
        'verbosity': -1,
        'random_state': 42
    }
    return evaluate_base_model_cv(params, 'lgb', train_fe, y_train, df_all_prep)

# XGBoost Tuning (木の深さ:3〜4, L2正則化:10.0〜100.0)
def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 4), 
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08),
        'min_child_weight': trial.suggest_int('min_child_weight', 5, 20),
        'reg_alpha': trial.suggest_float('reg_alpha', 1.0, 50.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 10.0, 100.0), 
        'subsample': trial.suggest_float('subsample', 0.5, 0.7),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.7),
        'eval_metric': 'logloss',
        'verbosity': 0,
        'random_state': 42
    }
    return evaluate_base_model_cv(params, 'xgb', train_fe, y_train, df_all_prep)

# CatBoost Tuning (木の深さ:3〜4, L2正則化:10.0〜100.0)
def objective_cat(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 50, 200),
        'depth': trial.suggest_int('depth', 3, 4), 
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 10.0, 100.0), 
        'subsample': trial.suggest_float('subsample', 0.5, 0.7),
        'verbose': 0,
        'random_seed': 42
    }
    return evaluate_base_model_cv(params, 'cat', train_fe, y_train, df_all_prep)
```

### 1. 決定木の構造と正則化の極限制限
木モデルに対して、以下のパラメータ空間で Optuna チューニングを実施。

* **LightGBM / XGBoost**:
  * `max_depth` (木の深さ): 3〜4 に制限 (通常は6〜7や制限なし)
  * `num_leaves` (葉の数): 7〜15 に制限
  * `reg_lambda` (L2正則化): 10.0〜100.0 に超強化 (通常は0.0〜10.0)
  * `subsample` / `colsample_bytree`: 0.5〜0.7 に制限し、データのバリアンスば低減
* **CatBoost**:
  * `depth` (深さ): 3〜4 に制限
  * `l2_leaf_reg` (L2正則化): 10.0〜100.0 に強化

これにより、モデルは「グループが同じなら全員助かる」といった複雑で極端な決定境界じゃなくなって、乗客自身の基本属性(年齢や性別)とのバランスば取ったマイルドな予測をするようになる。

![過学習と汎化の決定境界のイメージ](https://github.com/kito2718/zenn_articles/raw/main/articles/images/zenn_20260708_2055_overfitting_generalization.png =500x)

### 2. 7つの多様なベースモデルのブレンド
木モデル以外のアルゴリズムに対しても、過学習を防ぐ強めの正則化を適用。
* ロジスティック回帰: `C=0.01`
* SVM (RBFカーネル): `C=0.1`
* MLP (ニューラルネットワーク): `alpha=10.0`, 構造ば (16, 8) に縮小
* KNN: `n_neighbors=15` (マイルド化)

これら 7モデル of the classの OOF 予測確率から、ブレンド比率(重み)を Optuna (100 trials) で最適化。

![多様なモデルの加重平均ブレンドのイメージ](https://github.com/kito2718/zenn_articles/raw/main/articles/images/zenn_20260708_2055_ensemble_blending.png =500x)

# 実行結果と仮説の立証

### 1. 各モデルの CV Accuracy (最適化後)
* **LGB**: 0.85522
* **XGB**: 0.85073
* **CAT**: 0.86308
* **ロジスティック回帰**: 0.85522
* **SVM**: 0.83838
* **KNN**: 0.84400
* **MLP**: 0.85297
* **Weighted Blending CV**: **0.86420**

以前の自己最高CV(0.86869) に比べると、木モデル単体の過学習が抑えられたことで CV全体は少し低下した。モデルの頑健性は格段に上がっとるはず。

★★★

### 2. 提出 (Kaggle Public Score)
* **Public Score**: **0.79425**(届かず!)
うーん、なかなか、ムズい。

★★★

お役に立てれば。