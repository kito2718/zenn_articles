---
title: "15-⑦[AI][Kaggle]Kaggle実践1: チケット番号の近接性による客室配置の擬似復元とモデルの死角分析"
emoji: "🚢"
type: "tech"
topics: ["kaggle", "titanic", "catboost", "machinelearning"]
published: true
---

[Kaggle実践1『Titanic生存者予測』1.ローカルPCにKaggle Titanicの実行環境をつくる](https://zenn.dev/rg687076/articles/zenn_260627_0000_00_create_local_titanic_env)
[Kaggle実践1『Titanic生存者予測』2.初回提出](https://zenn.dev/rg687076/articles/zenn_260627_0000_01_first_submission)
[Kaggle実践1『Titanic生存者予測』3.Cabinの特徴量エンジニアリング](https://zenn.dev/rg687076/articles/zenn_260627_1940_01_cabin_feature)
[Kaggle実践1『Titanic生存者予測』4.特徴量エンジニアリング(ランダムフォレストで年齢補完)](https://zenn.dev/rg687076/articles/zenn_20260702_2031_age_imputation)
[Kaggle実践1『Titanic生存者予測』5.特徴量エンジニアリング(数値特徴量の非線形変換とビン化](https://zenn.dev/rg687076/articles/zenn_20260703_2025_fare_log_and_age_binning)
[Kaggle実践1『Titanic生存者予測』6.避難行動を捉えるグループ統計量の追加とCatBoost×Optuna自動最適化](https://zenn.dev/rg687076/articles/zenn_20260706_2030_catboost_optuna_tuning)
[Kaggle実践1『Titanic生存者予測』7.チケット番号の近接性による客室配置の擬似復元とモデルの死角分析](https://zenn.dev/rg687076/articles/zenn_20260707_2118_ticket_neighbor_survival)

https://www.kaggle.com/c/titanic

← [Kaggle入門14(ゲームAIと強化学習入門)](https://zenn.dev/rg687076/articles/49e1d162bfdeec)
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;[Kaggle実践2(xxxx)](https://zenn.dev/rg687076/articles/xxxx) →

[github公開中](https://github.com/kito2718/KaggleTitanic)

# Abstract
- エラー分析で、3等客室の乗客の誤分類が全体の約57%を占める死角があるのが分かったので、特定してみた。
- CVで 0.8687、だけど Public Scoreは、0.78947 で下がってる。

# データから見えたモデルの死角
前回の検証でスコアの良かった CatBoost モデルを使って、実際にどの乗客を誤分類しているかエラー分析を行った。
結果、誤分類130サンプルのうち、**56.9%** が3等客室(Pclass 3) の乗客(特に男性)というのがわかった。

![エラー分析 (Pclass別の誤分類分布)](https://github.com/kito2718/KaggleTitanic/raw/main/zenn_articles/articles/images/zenn_20260707_2118_error_analysis_pclass.png)*エラー分析(Pclass別の誤分類分布)*

# チケット近接性による客室配置の擬似復元
タイタニック号の客室番号 (Cabin) はデータの約77%が欠損してるけれども、船室の物理的な配置は生死に直結する重要な要素みたい。つまりは、船内のどのエリアの部屋に居たかは、最上階の避難ボートまでの距離や、浸水が始まった場所からの脱出経路に直結しとると考えられ。
なので、チケット番号 (Ticket) の数値部分の規則性に着目。チケット番号が近い乗客 (下数桁の差が 5 以内) は、同時期に同じ場所でチケットを購入し、船内で隣接する客室に配置されとった可能性が高いと考察。
つまり、「チケット番号が近い (＝同じエリアに寝まりしとった) 近傍の人たちの生存率」 をモデルに教えることで、「そのエリアは避難しやすかった恵まれた場所か、それとも逃げ遅れやすい脱出困難な場所やったか」という物理的な避難環境を表現できる特徴量になるかと！

![Ticket近接生存率 (OOF) の生存・死亡別分布](https://github.com/kito2718/KaggleTitanic/raw/main/zenn_articles/articles/images/zenn_20260707_2118_ticket_neighbor_survival_dist.png)*Ticket近接生存率 (OOF) の生存・死亡別分布*

このドメイン知識に基づいて、以下の特徴量エンジニアリングを実施。

1. **チケット近接生存率 (OOF_Ticket_Neighbor_Survival)**:
   自分以外のチケット番号が前後 5 以内の近傍グループの生存率を、情報リークを防ぐために Out-of-Fold (OOF) 方式で計算して追加。
2. **3等客室プレフィックス相互作用 (Prefix_[prefix]_3rd)**:
   主要なチケットプレフィックス (a5, pc, ca, stono, sotono2) と Pclass 3 の掛け合わせフラグを作成、3等客室特有の配置セクターを表現できるようにした。

```python
# チケット近接生存率 (OOF) の計算ロジック
cv_for_oof = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_neighbor_survival = pd.Series(0.5, index=train_fe.index)

for train_idx, val_idx in cv_for_oof.split(train_fe, y_train):
    tr_df = train_fe.iloc[train_idx].copy()
    tr_df['Survived'] = y_train.iloc[train_idx]
    
    for idx in val_idx:
        row = train_fe.iloc[idx]
        t_num = row['Ticket_Num']
        if not pd.isna(t_num):
            # 自分以外のチケット番号が+-5の範囲内にある近傍サンプルを抽出
            neighbors = tr_df[
                (tr_df['Ticket_Num'] >= t_num - 5) & 
                (tr_df['Ticket_Num'] <= t_num + 5) & 
                (tr_df['PassengerId'] != row['PassengerId'])
            ]
            if len(neighbors) > 0:
                oof_neighbor_survival.loc[idx] = neighbors['Survived'].mean()
            else:
                p_s_mean = tr_df[(tr_df['Pclass'] == row['Pclass']) & (tr_df['Sex_male'] == row['Sex_male'])]['Survived'].mean()
                oof_neighbor_survival.loc[idx] = p_s_mean
        else:
            p_s_mean = tr_df[(tr_df['Pclass'] == row['Pclass']) & (tr_df['Sex_male'] == row['Sex_male'])]['Survived'].mean()
            oof_neighbor_survival.loc[idx] = p_s_mean
```

# 驚異的な精度向上と Kaggle 提出
下記コードを実行して複数のパターンを 5-Fold Stratified CV で評価した。

:::details evaluate_step8.py (一時検証スクリプトの全コード)
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostClassifier
import optuna
import os

# --- 1. データロードと共通前処理 (ベースライン) ---
print("--- Loading Data and Basic Processing ---")
train = pd.read_csv('data/raw/train.csv')
test = pd.read_csv('data/raw/test.csv')

df_all = pd.concat([train, test], sort=False).reset_index(drop=True)
df_all['Last_Name'] = df_all['Name'].apply(lambda x: x.split(',')[0])

# グループ生存特徴量の基本設定 (Family_Survival)
DEFAULT_SURVIVAL_VALUE = 0.5
df_all['Family_Survival'] = DEFAULT_SURVIVAL_VALUE

for grp, grp_df in df_all.groupby(['Last_Name', 'Fare']):
    if len(grp_df) > 1:
        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if smax == 1.0:
                df_all.loc[df_all['PassengerId'] == passID, 'Family_Survival'] = 1.0
            elif smin == 0.0:
                df_all.loc[df_all['PassengerId'] == passID, 'Family_Survival'] = 0.0

for grp, grp_df in df_all.groupby('Ticket'):
    if len(grp_df) > 1:
        for ind, row in grp_df.iterrows():
            passID = row['PassengerId']
            if df_all.loc[df_all['PassengerId'] == passID, 'Family_Survival'].values[0] == 0.5:
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                if smax == 1.0:
                    df_all.loc[df_all['PassengerId'] == passID, 'Family_Survival'] = 1.0
                elif smin == 0.0:
                    df_all.loc[df_all['PassengerId'] == passID, 'Family_Survival'] = 0.0

df_all['Title'] = df_all['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
title_map = {'Mr':'Mr','Miss':'Miss','Mrs':'Mrs','Master':'Master','Dr':'Rare','Rev':'Rare','Col':'Rare','Major':'Rare','Mlle':'Miss','Countess':'Rare','Ms':'Miss','Lady':'Rare','Jonkheer':'Rare','Don':'Rare','Dona':'Rare','Mme':'Mrs','Capt':'Rare','Sir':'Rare'}
df_all['Title'] = df_all['Title'].map(title_map).fillna('Rare')

df_all['Fare'] = df_all['Fare'].fillna(df_all['Fare'].median())
df_all['Embarked'] = df_all['Embarked'].fillna(df_all['Embarked'].mode()[0])
df_all['Deck'] = df_all['Cabin'].fillna('U').apply(lambda x: x[0])
df_all['FamilySize'] = df_all['SibSp'] + df_all['Parch'] + 1
df_all['IsAlone'] = (df_all['FamilySize'] == 1).astype(int)

# Ageの予測補完
age_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'Deck', 'FamilySize', 'IsAlone', 'Age']
df_age_prep = df_all[age_features].copy()
cat_cols_for_age = ['Sex', 'Embarked', 'Title', 'Deck']
df_age_encoded = pd.get_dummies(df_age_prep, columns=cat_cols_for_age, drop_first=True)
train_age = df_age_encoded[df_age_encoded['Age'].notnull()]
test_age = df_age_encoded[df_age_encoded['Age'].isnull()]
X_train_age = train_age.drop(columns=['Age'])
y_train_age = train_age['Age']
X_test_age = test_age.drop(columns=['Age'])

age_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
age_regressor.fit(X_train_age, y_train_age)
predicted_ages = age_regressor.predict(X_test_age)
df_all.loc[df_all['Age'].isnull(), 'Age'] = predicted_ages

# 高度なグループ特徴量
df_all['Ticket_Group_Size'] = df_all.groupby('Ticket')['PassengerId'].transform('count')
df_all['Group_Id'] = df_all['Ticket']
mask = df_all['Ticket_Group_Size'] == 1
df_all.loc[mask, 'Group_Id'] = df_all.loc[mask, 'Last_Name'] + '_' + df_all.loc[mask, 'Fare'].astype(str)

df_all['Group_Size'] = df_all.groupby('Group_Id')['PassengerId'].transform('count')
df_all['Is_Female_or_Child'] = ((df_all['Sex'] == 'female') | (df_all['Age'] < 16)).astype(int)
df_all['Group_Female_Child_Ratio'] = df_all.groupby('Group_Id')['Is_Female_or_Child'].transform('mean')
df_all['Group_Mean_Age'] = df_all.groupby('Group_Id')['Age'].transform('mean')
pclass_fare_median = df_all.groupby('Pclass')['Fare'].transform('median')
df_all['Group_Fare_Median_Diff'] = df_all['Fare'] - pclass_fare_median

# 後で使うため Ticket_Group_Size は残す
df_all = df_all.drop(columns=['Is_Female_or_Child'])


# --- 2. 新規特徴量の設計 (データソース) ---

# 一人あたり運賃の計算
df_all['Fare_per_person'] = df_all['Fare'] / df_all['Ticket_Group_Size']

# 施策1: 3等客室の社会的・物理的環境の細分化
# Pclass == 3 の中での Fare_per_person の10%パーセンタイル値を取得
fare_threshold_3rd = df_all[df_all['Pclass'] == 3]['Fare_per_person'].quantile(0.1)
df_all['Is_Ultra_Poor_3rd'] = ((df_all['Pclass'] == 3) & (df_all['Fare_per_person'] <= fare_threshold_3rd)).astype(int)

# Embarked と Pclass 3 の掛け合わせフラグ (Embarked_S_3rd, Embarked_C_3rd, Embarked_Q_3rd)
df_all['Embarked_S_3rd'] = ((df_all['Pclass'] == 3) & (df_all['Embarked'] == 'S')).astype(int)
df_all['Embarked_C_3rd'] = ((df_all['Pclass'] == 3) & (df_all['Embarked'] == 'C')).astype(int)
df_all['Embarked_Q_3rd'] = ((df_all['Pclass'] == 3) & (df_all['Embarked'] == 'Q')).astype(int)


# 施策2: チケット近接性による客室配置の擬似復元
# チケット番号の数値部分を抽出
def extract_ticket_num(ticket):
    parts = ticket.split()
    if len(parts) == 0:
        return np.nan
    last_part = parts[-1]
    if last_part.isdigit():
        return int(last_part)
    return np.nan

df_all['Ticket_Num'] = df_all['Ticket'].apply(extract_ticket_num)

# チケットプレフィックスの抽出
def extract_ticket_prefix(ticket):
    parts = ticket.split()
    if len(parts) > 1:
        prefix = "".join(parts[:-1])
        prefix = prefix.replace(".", "").replace("/", "").lower()
        return prefix
    return 'none'
df_all['Ticket_Prefix'] = df_all['Ticket'].apply(extract_ticket_prefix)

# 3等客室と特定プレフィックスの相互作用 (主要なプレフィックスに限定)
major_prefixes = ['a5', 'pc', 'ca', 'stono', 'sotono2']
for pref in major_prefixes:
    col_name = f'Prefix_{pref}_3rd'
    df_all[col_name] = ((df_all['Pclass'] == 3) & (df_all['Ticket_Prefix'] == pref)).astype(int)


# 施策3: 家族構成に基づく避難時の行動シグナル化
# 3等客室の成人男性で家族同伴
df_all['Male_3rd_Has_Family'] = ((df_all['Pclass'] == 3) & (df_all['Sex'] == 'male') & (df_all['Age'] >= 16) & (df_all['SibSp'] + df_all['Parch'] > 0)).astype(int)

# 同一グループ内に子供 (Age < 16) がいる3等客室の保護者/親族
df_all['Group_Has_Child'] = df_all.groupby('Group_Id')['Age'].transform(lambda x: (x < 16).any()).astype(int)
df_all['Is_3rd_Parent_Guardian'] = ((df_all['Pclass'] == 3) & (df_all['Age'] >= 16) & (df_all['Group_Has_Child'] == 1)).astype(int)

df_all = df_all.drop(columns=['Group_Has_Child', 'Ticket_Group_Size'])


# --- 3. 検証パターン定義 ---
base_features = [
    'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Family_Survival', 'FamilySize', 'IsAlone',
    'Group_Size', 'Group_Female_Child_Ratio', 'Group_Mean_Age', 'Group_Fare_Median_Diff',
    # One-hot カラム
    'Sex_male', 'Embarked_Q', 'Embarked_S', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Rare',
    'Deck_B', 'Deck_C', 'Deck_D', 'Deck_E', 'Deck_F', 'Deck_G', 'Deck_T', 'Deck_U'
]

features_pattern_a = [
    'Is_Ultra_Poor_3rd', 'Embarked_S_3rd', 'Embarked_C_3rd', 'Embarked_Q_3rd'
]

features_pattern_c = [
    'Male_3rd_Has_Family', 'Is_3rd_Parent_Guardian'
]


# --- 4. 共通ダミー変数化 ---
cat_cols = ['Sex', 'Embarked', 'Title', 'Deck']
df_encoded = pd.get_dummies(df_all, columns=cat_cols, drop_first=True)

# 訓練・テスト分割
train_fe = df_encoded.iloc[:len(train)].copy()
test_fe = df_encoded.iloc[len(train):].copy()
y_train = train_fe['Survived'].astype(int)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Optuna用の目的関数ジェネレータ
def get_objective(X_tr_full, y_tr):
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 50, 300),
            'depth': trial.suggest_int('depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'verbose': 0,
            'random_seed': 42
        }
        model = CatBoostClassifier(**params)
        scores = []
        for train_idx, val_idx in cv.split(X_tr_full, y_tr):
            X_tr, y_tr_fold = X_tr_full.iloc[train_idx], y_tr.iloc[train_idx]
            X_va, y_va_fold = X_tr_full.iloc[val_idx], y_tr.iloc[val_idx]
            model.fit(X_tr, y_tr_fold)
            preds = model.predict(X_va)
            scores.append(np.mean(preds == y_va_fold))
        return np.mean(scores)
    return objective

optuna.logging.set_verbosity(optuna.logging.WARNING)

results = {}

# --- [Baseline] 施策なし ---
print("\n--- Evaluating Baseline CatBoost (No New Features) ---")
X_train_base = train_fe[base_features]
study = optuna.create_study(direction='maximize')
study.optimize(get_objective(X_train_base, y_train), n_trials=20)
results['Baseline'] = {
    'accuracy': study.best_value,
    'params': study.best_params,
    'features': base_features
}
print(f"Baseline Best CV: {study.best_value:.4f}")

# --- [Pattern A] 施策1 (社会的・物理的環境) のみ ---
print("\n--- Evaluating Pattern A (Step 1 Features: Ultra Poor & Embarked Sector) ---")
features_a = base_features + features_pattern_a
X_train_a = train_fe[features_a]
study = optuna.create_study(direction='maximize')
study.optimize(get_objective(X_train_a, y_train), n_trials=20)
results['Pattern A'] = {
    'accuracy': study.best_value,
    'params': study.best_params,
    'features': features_a
}
print(f"Pattern A Best CV: {study.best_value:.4f}")


# --- [Pattern B] 施策2 (チケット近接性による客室配置の擬似復元) のみ ---
print("\n--- Evaluating Pattern B (Step 2 Features: Ticket Neighbor OOF & Prefix Interaction) ---")

# チケット近接生存率を K-Fold OOF で計算する
oof_neighbor_survival = pd.Series(0.5, index=train_fe.index)
for train_idx, val_idx in cv.split(train_fe, y_train):
    tr_df = train_fe.iloc[train_idx].copy()
    tr_df['Survived'] = y_train.iloc[train_idx]
    
    # 検証Foldの各乗客について近接チケット乗客を探索
    for idx in val_idx:
        row = train_fe.iloc[idx]
        t_num = row['Ticket_Num']
        
        if not pd.isna(t_num):
            # チケット数値が自分以外の +-5 の範囲にある乗客を抽出
            neighbors = tr_df[
                (tr_df['Ticket_Num'] >= t_num - 5) & 
                (tr_df['Ticket_Num'] <= t_num + 5) & 
                (tr_df['PassengerId'] != row['PassengerId'])
            ]
            if len(neighbors) > 0:
                oof_neighbor_survival.loc[idx] = neighbors['Survived'].mean()
            else:
                # 近傍がいない場合は Pclass/Sex の平均生存率で補正
                p_s_mean = tr_df[(tr_df['Pclass'] == row['Pclass']) & (tr_df['Sex_male'] == row['Sex_male'])]['Survived'].mean()
                oof_neighbor_survival.loc[idx] = p_s_mean
        else:
            # チケット番号が非数値の場合は Pclass/Sex の平均
            p_s_mean = tr_df[(tr_df['Pclass'] == row['Pclass']) & (tr_df['Sex_male'] == row['Sex_male'])]['Survived'].mean()
            oof_neighbor_survival.loc[idx] = p_s_mean

train_fe_b = train_fe.copy()
train_fe_b['OOF_Ticket_Neighbor_Survival'] = oof_neighbor_survival

features_b = base_features + [f'Prefix_{pref}_3rd' for pref in major_prefixes] + ['OOF_Ticket_Neighbor_Survival']
X_train_b = train_fe_b[features_b]

study = optuna.create_study(direction='maximize')
study.optimize(get_objective(X_train_b, y_train), n_trials=20)
results['Pattern B'] = {
    'accuracy': study.best_value,
    'params': study.best_params,
    'features': features_b
}
print(f"Pattern B Best CV: {study.best_value:.4f}")


# --- [Pattern C] 施策3 (避難時の家族優先行動) のみ ---
print("\n--- Evaluating Pattern C (Step 3 Features: Male 3rd Family & Parent Guardian) ---")
features_c = base_features + features_pattern_c
X_train_c = train_fe[features_c]
study = optuna.create_study(direction='maximize')
study.optimize(get_objective(X_train_c, y_train), n_trials=20)
results['Pattern C'] = {
    'accuracy': study.best_value,
    'params': study.best_params,
    'features': features_c
}
print(f"Pattern C Best CV: {study.best_value:.4f}")


# --- [Pattern D] 施策1〜3すべてを同時追加 (シナジー検証) ---
print("\n--- Evaluating Pattern D (All Features Combined: Synergy) ---")
train_fe_d = train_fe_b.copy() # OOF_Ticket_Neighbor_Survivalを保持
train_fe_d['Is_Ultra_Poor_3rd'] = train_fe['Is_Ultra_Poor_3rd']
train_fe_d['Embarked_S_3rd'] = train_fe['Embarked_S_3rd']
train_fe_d['Embarked_C_3rd'] = train_fe['Embarked_C_3rd']
train_fe_d['Embarked_Q_3rd'] = train_fe['Embarked_Q_3rd']
train_fe_d['Male_3rd_Has_Family'] = train_fe['Male_3rd_Has_Family']
train_fe_d['Is_3rd_Parent_Guardian'] = train_fe['Is_3rd_Parent_Guardian']

features_d = features_b + features_pattern_a + features_pattern_c
X_train_d = train_fe_d[features_d]

study = optuna.create_study(direction='maximize')
study.optimize(get_objective(X_train_d, y_train), n_trials=20)
results['Pattern D'] = {
    'accuracy': study.best_value,
    'params': study.best_params,
    'features': features_d
}
print(f"Pattern D Best CV: {study.best_value:.4f}")


# --- 5. 結果のまとめと最良モデルの選定 ---
print("\n--- Final CV Comparison ---")
for key, val in results.items():
    print(f"{key}: {val['accuracy']:.5f}")

best_pattern = max(results, key=lambda k: results[k]['accuracy'])
print(f"\nBest Performing Pattern: {best_pattern} with CV: {results[best_pattern]['accuracy']:.5f}")
```
:::

で、結果のまとめ
- **ベースライン (施策前 CatBoost)**: CV 0.8530
- **Pattern A (社会的・物理的環境のみ)**: CV 0.8530
- **Pattern B (施策2：チケット近接性 OOF & プレフィックス相互作用)**: **CV 0.8687 (大幅向上！)**
- **Pattern C (施策3：家族同伴 & 保護者役割のみ)**: CV 0.8574
- **Pattern D (全施策同時投入)**: CV 0.8642

チケット近接性を用いた Pattern B が **CV: 0.8687** スコアとなった！
この Pattern B でテストデータを予測し、Kaggle に提出。
けど Public Score は、**0.78947** 伸びず。

# まとめ
まだまだ、満足には遠いなぁ。

お役に立てれば。
