---
title: "『Titanic生存者予測』3.Name(敬称)の特徴量エンジニアリング"
emoji: "🍣"
type: "tech"
topics:
  - "ai"
  - "python"
  - "kaggle"
  - "titanic"
published: true
published_at: "2026-05-02 11:15"
---

[1.まず一通りやってみる](https://zenn.dev/rg687076/articles/76b1608f4ffe36)
[2.ブラッシュアップのロードマップ)](https://zenn.dev/rg687076/articles/f0c1aa0b59ea76)
[3.Name(敬称)の特徴量エンジニアリング)](https://zenn.dev/rg687076/articles/858ea82fddadc1)
[4.家族人数の特徴量エンジニアリング)](https://zenn.dev/rg687076/articles/52d7e8f375e9ba)
[5.年齢の特徴量エンジニアリング)](https://zenn.dev/rg687076/articles/04f64c76d7ffb6)


https://www.kaggle.com/c/titanic

# Abstract
- [初回](https://zenn.dev/rg687076/articles/76b1608f4ffe36)に作成したベースラインのモデルにName列の特徴量エンジニアリングを追加。

# 概要
### Name列
KaggleのTitanicコンペは、Nameから敬称を取り出して特徴量にするのが効果があるらしく、今回は Name 列から`Mr`や`Miss`といった敬称を抽出し、新たな特徴量として追加します。
敬称には性別や社会的立場の情報が含まれており、例えば Mr は成人男性、Miss は未婚女性、Master は男児を表すため、生存率に関係する可能性があります。
実際、Titanicでは「女性や子どもが優先的に救助された」と言われているため、こうした情報を特徴量として持たせることで、Sex や Age だけでは表現しきれない違いをモデルが学習することを期待します。

#### 敬称ごと(Mr,Miss,etc...)の生存率を可視化する
まずは Name から敬称を取り出し、敬称ごとの生存率を確認します。
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. データの読み込みと敬称の抽出
train_data = pd.read_csv('/kaggle/input/competitions/titanic/train.csv') 
train_data['Title'] = train_data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
# pandas.Series.str.extract()は、pandasのSeries(列)に含まれる文字列から、
# 正規表現（Regular Expression）を使って特定のパターンを抜き出すための関数
# 主に、乱雑なデータから特定のキーワード、数値、日付などを抽出して新しい列を
# 作りたいときに非常に重宝する。
# expand ... True:戻り型pandas.DataFrame、False:戻り型pandas.Series

# 2. グラフのサイズ設定
plt.figure(figsize=(12, 5))

# 3. 棒グラフ（barplot）の作成
title_order = train_data.groupby('Title')['Survived'].agg(['count', 'mean']).index
# ※ Seabornのbarplotはデフォルトで平均値（＝生存率）を計算してくれます
sns.barplot(x='Title', y='Survived', data=train_data, palette='viridis',err_kws={'color': 'red'}, order=title_order)

# 4. グラフの装飾
plt.title('Survival Rate by Title')
plt.ylabel('Survival Rate')
plt.xlabel('Title')
plt.xticks(rotation=45) # 敬称が重ならないように斜めにする
plt.show()

train_data.groupby('Title')['Survived'].agg(['count', 'mean'])
```
![](https://static.zenn.studio/user-upload/68957fa34ed3-20260429.png)
※↑図の赤縦棒は、各Titleの生存率のばらつきを表します。棒が長いと、サンプル数が少ない/ばらつきが大きいという意味です。
※↓表からは「ばらつき大の正体はサンプル数が少ないため」というのが分かります。

```
		count	mean
Title
---------------+-------+--------
Capt		1	0.000000
Col		2	0.500000
Countess	1	1.000000
Don		1	0.000000
Dr		7	0.428571
Jonkheer	1	0.000000
Lady		1	1.000000
Major		2	0.500000
Master		40	0.575000
Miss		182	0.697802
Mlle		2	1.000000
Mme		1	1.000000
Mr		517	0.156673
Mrs		125	0.792000
Ms		1	1.000000
Rev		6	0.000000
Sir		1	1.000000
```
#### 敬称の種類をまとめる
サンプル数が少ない項目(CaptやらCountess)は、そのまま使うと、たまたま生きた/死んだだけで生存率が決まってしまい、情報がノイズになるので、少数項目をまとめます。
まとめの方針は、ざっくり以下の分け方になればいいかなと。
既婚女性 / 未婚女性 / 男児 / 成人男性 / その他

--- まとめ指針 ---
|敬称|変更先|備考|
|---|---|---|
|Mlle|Miss|フランス語。Missと同様|
|Ms|Miss|Missと同様|
|Lady|Miss|貴族階級の女性の敬称。〜夫人／令嬢／貴婦人。既婚か未婚かわからない|
|Mme|Mrs|フランス語。Mrsと同様|
|Countess|Mrs|`伯爵夫人`って意味。既婚女性なのでMrsに分類|
|Master|Master|男の子を表す。|
|Capt|Mr|Captainの略。大尉(陸軍・海軍)／船長／機長|
|Col|Mr|Colonelの略。大佐(陸軍・空軍など)|
|Major|Mr|少佐(陸軍・空軍など)|
|Don|Mr|スペイン・イタリア圏。〜閣下／旦那／先生|
|Jonkheer|Mr|オランダ・ベルギー圏。貴公子の称号(無爵貴族)|
|Sir|Mr|英国国王からナイト(勲爵士)の爵位を授与された貴族・功労者|

上記の変換コードは下記となります。
```python
##### 特徴量エンジニアリング(敬称抽出)
all_data['Title'] = all_data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
all_data['Title'] = all_data['Title'].replace(['Mlle', 'Ms',  'Lady'], 'Miss')
all_data['Title'] = all_data['Title'].replace(['Lady'],'Miss')
all_data['Title'] = all_data['Title'].replace(['Mme', 'Countess'], 'Mrs')
all_data['Title'] = all_data['Title'].replace(['Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Sir'],'Mr')
# 表示する
train_data.groupby('Title')['Survived'].agg(['count', 'mean'])
```

```
        count	mean
Title		
-------+-------+--------
Dr          7  0.428571
Master     40  0.575000
Miss      186  0.704301
Mr        525  0.160000
Mrs       127  0.795276
Rev         6  0.000000
```
期待通り、敬称がまとまっています。

#### 敬称(Title)を追加してモデル生成 → 結果の提出(LabelEncoding版)
方針が決まったので、その分も追加してモデル生成し、結果を提出します。
敬称の各項目を数値化するのに、LabelEncodingを用いています。One-Hot Encodingの方が精度が出るということなんですが、LabelEncodingでも試してみたい。
コードは下記。
```python
# 敬称追加(LabelEncoding版)

# import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# データ読み込み
train_data = pd.read_csv('/kaggle/input/competitions/titanic/train.csv') 
test_data  = pd.read_csv('/kaggle/input/competitions/titanic/test.csv')

####### 敬称追加(LabelEncoding版)処理 ここから
# ===== 敬称列(Title)生成(train&test 両方) =====
for df in [train_data, test_data]:
    # 敬称抽出→Title列生成
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    # 敬称をまとめる
    df['Title'] = df['Title'].replace(['Mlle', 'Ms',  'Lady'], 'Miss')
    df['Title'] = df['Title'].replace(['Lady'],'Miss')
    df['Title'] = df['Title'].replace(['Mme', 'Countess'], 'Mrs')
    df['Title'] = df['Title'].replace(['Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Sir'],'Mr')

# LabelEncoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
# trainとtestのTitle列を縦にくっつけて1つにして、`all_titles`を取得する。
all_titles = pd.concat([train_data['Title'], test_data['Title']])
le.fit(all_titles)

train_data['Title'] = le.transform(train_data['Title'])
test_data ['Title'] = le.transform(test_data ['Title'])
####### 敬称追加(LabelEncoding版)処理 ここまで

# 4つの特徴量を選択
features = ["Pclass", "Sex", "Age", "Fare", "Title"]

# 学習データを準備
X = train_data[features].copy()
y = train_data["Survived"]

X_test = test_data[features].copy()

# 欠損値補完
age_median = X["Age"].median()
fare_median = X["Fare"].median()

X["Age"] = X["Age"].fillna(age_median)
X_test["Age"] = X_test["Age"].fillna(age_median)

X["Fare"] = X["Fare"].fillna(fare_median)
X_test["Fare"] = X_test["Fare"].fillna(fare_median)

# Sex を数値化
X["Sex"] = X["Sex"].map({"male": 0, "female": 1})
X_test["Sex"] = X_test["Sex"].map({"male": 0, "female": 1})


# モデル作成
model = DecisionTreeClassifier(random_state=1)

# 学習
model.fit(X, y)

# 予測
predictions = model.predict(X_test)

# 提出ファイル作成
output = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": predictions
})

# 提出ファイル出力
output.to_csv("submission.csv", index=False)

# 完了
print("submission.csv を作成しました 002")
```

#### 提出結果(LabelEncoding版)
なんか、スコアは上がってるけど、順位は下がってる😥
![](https://static.zenn.studio/user-upload/fff42c4d141a-20260501.png)
![](https://static.zenn.studio/user-upload/a0b7c953fb13-20260501.png)
One-Hotとの比較で、お試し実装なので抑制的にスコアが上がってるのはまぁいいとして、それでもちょっとしか上がってなくって、なんか残念。そんなもんなんかな。

#### モデル生成 → 結果の提出(One-Hot Encoding版)
文字列の場合、各要素が大小関係/順番関係を持たない時は、One-Hot Encodingの方が有効な場合が多いらしいので、One-Hot Encodingを実装してみます。
コードは下記。
```python
train_oh = train_data.copy()
train_oh = pd.get_dummies(train_oh, columns=['Title'])
train_oh.columns
```
モデル生成 → 結果生成までのコードは下記
```python
# 敬称追加(One-Hot Encoding版)

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# データ読み込み
train_data = pd.read_csv('/kaggle/input/competitions/titanic/train.csv') 
test_data  = pd.read_csv('/kaggle/input/competitions/titanic/test.csv')

####### 敬称追加(One-Hot Encoding版)処理 ここから
# ===== 敬称列(Title)生成(train&test 両方) =====
for df in [train_data, test_data]:
    # 敬称抽出→Title列生成
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    # 敬称をまとめる
    df['Title'] = df['Title'].replace(['Mlle', 'Ms',  'Lady'], 'Miss')
    df['Title'] = df['Title'].replace(['Lady'],'Miss')
    df['Title'] = df['Title'].replace(['Mme', 'Countess'], 'Mrs')
    df['Title'] = df['Title'].replace(['Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Sir'],'Mr')

# trainとtestのTitle列を縦にくっつけて1つにして、`all_titles`を取得する。
all_data = pd.concat([train_data, test_data], ignore_index=True)
# One-Hot Encoding実行
all_data = pd.get_dummies(all_data, columns=['Title'])
# くっつけたのを元に戻す
train_data = all_data.iloc[:len(train_data)].copy()
test_data  = all_data.iloc[len(train_data):].copy()
print(train_data.columns)
####### 敬称追加(One-Hot Encoding版)処理 ここまで

# 4つの特徴量を選択
features = ["Pclass", "Sex", "Age", "Fare"] + [col for col in train_data.columns if "Title_" in col]

# 学習データを準備
X = train_data[features].copy()
y = train_data["Survived"]

X_test = test_data[features].copy()

# 欠損値補完
age_median = X["Age"].median()
fare_median = X["Fare"].median()

X["Age"] = X["Age"].fillna(age_median)
X_test["Age"] = X_test["Age"].fillna(age_median)

X["Fare"] = X["Fare"].fillna(fare_median)
X_test["Fare"] = X_test["Fare"].fillna(fare_median)

# Sex を数値化
X["Sex"] = X["Sex"].map({"male": 0, "female": 1})
X_test["Sex"] = X_test["Sex"].map({"male": 0, "female": 1})


# モデル作成
model = DecisionTreeClassifier(random_state=1)

# 学習
model.fit(X, y)

# 予測
predictions = model.predict(X_test)
predictions = predictions.astype(int)

# 提出ファイル作成
output = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": predictions
})

# 提出ファイル出力
output.to_csv("submission.csv", index=False)

# 完了
print("submission.csv を作成しました 002-2")
```
下がってしもうたー。なんでだー?
![](https://static.zenn.studio/user-upload/d00f19b15fdd-20260501.png)
理由は、モデル生成にDecisionTree使ってるからっぽい。

#### モデル生成 → 結果の提出(One-Hot Encoding版 + RandomForest)
DecisionTreeは、LabelEncodingと相性がいいらしい。なので、RandomForestでモデル生成する様に修正する。
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=1)
```

![](https://static.zenn.studio/user-upload/581b492e08b9-20260501.png)
ちょっとよくなった👍 DecisionTreeと比べると1%ぐらいスコアアップ。でもこんなもんなんだ。ちょっとづつを積み重ねる感じなんだな。
![](https://static.zenn.studio/user-upload/7bd85b6d8d0c-20260501.png)
順位もそんなに上がってない。

#### さらにモデル生成 → 結果の提出(One-Hot Encodingの分類をさらにまとめてみる)
'Dr', 'Rev', 'Dona'も、アイテム数としては少ないので、Rareにまとめてみる。
```python
for df in [train_data, test_data]:
    # 敬称をまとめる
    df['Title'] = df['Title'].replace(['Capt', 'Col', 'Countess', 'Don', 'Jonkheer', 'Lady', 'Major', 'Sir', 'Dr', 'Rev', 'Dona'],'Rare')
```
![](https://static.zenn.studio/user-upload/69cbf3c92f00-20260502.png)
また、下がってしまった。
'Dr', 'Rev', 'Dona'をRareにまとめるには、意外と有益だったということかな。

`Dr` → 比較的裕福・生存率やや高め → 少し生き残る
`Rev` → 男性・年齢高め・生存率低め → ほぼ死ぬ
`Dona` → 女性・生存率高め → 生き残る

RandomForestはそれぞれを学習してるってことが考えられます。
'Dr', 'Rev', 'Dona'をRareにまとめるのは悪手ということでやめときます。

## この章の結論
Name列を特徴量エンジニアリングした結果、下記が得られたと思います。
1. Name列から敬称を抜き出す。
2. 敬称のアイテム数が少ないものは`その他(今回はRare)`にまとめる。
   ※まとめ過ぎてもだめ。情報が欠落するので。
4. One-Hot Encodingを実行する。
5. モデルをRandomForestに変更。
6. **結果(Scoreアップ): 0.72488 → 0.73444**

コードは下記。
```python
# 敬称追加(One-Hot Encoding版)

import pandas as pd

# データ読み込み
train_data = pd.read_csv('/kaggle/input/competitions/titanic/train.csv') 
test_data  = pd.read_csv('/kaggle/input/competitions/titanic/test.csv')

####### 敬称追加(One-Hot Encoding版)処理 ここから
# ===== 敬称列(Title)生成(train&test 両方) =====
for df in [train_data, test_data]:
    # 敬称抽出→Title列生成
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    # 敬称をまとめる
    df['Title'] = df['Title'].replace(['Mlle', 'Ms',  'Lady'], 'Miss')
    df['Title'] = df['Title'].replace(['Lady'],'Miss')
    df['Title'] = df['Title'].replace(['Mme', 'Countess'], 'Mrs')
    df['Title'] = df['Title'].replace(['Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Sir'],'Mr')

# trainとtestのTitle列を縦にくっつけて1つにして、`all_titles`を取得する。
all_data = pd.concat([train_data, test_data], ignore_index=True)
# One-Hot Encoding実行
all_data = pd.get_dummies(all_data, columns=['Title'])
# くっつけたのを元に戻す
train_data = all_data.iloc[:len(train_data)].copy()
test_data  = all_data.iloc[len(train_data):].copy()
print(train_data.columns)
####### 敬称追加(One-Hot Encoding版)処理 ここまで

# 4つの特徴量を選択
features = ["Pclass", "Sex", "Age", "Fare"] + [col for col in train_data.columns if "Title_" in col]

# 学習データを準備
X = train_data[features].copy()
y = train_data["Survived"]

X_test = test_data[features].copy()

# 欠損値補完
age_median = X["Age"].median()
fare_median = X["Fare"].median()

X["Age"] = X["Age"].fillna(age_median)
X_test["Age"] = X_test["Age"].fillna(age_median)

X["Fare"] = X["Fare"].fillna(fare_median)
X_test["Fare"] = X_test["Fare"].fillna(fare_median)

# Sex を数値化
X["Sex"] = X["Sex"].map({"male": 0, "female": 1})
X_test["Sex"] = X_test["Sex"].map({"male": 0, "female": 1})

# モデル作成
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=1)

# 学習
model.fit(X, y)

# 予測
predictions = model.predict(X_test)
predictions = predictions.astype(int)

# 提出ファイル作成
output = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": predictions
})

# 提出ファイル出力
output.to_csv("submission-3.csv", index=False)

# 完了
print("submission.csv を作成しました 003")
```
以上です。

次のステップ[[4.家族人数の特徴量エンジニアリング](https://zenn.dev/rg687076/articles/52d7e8f375e9ba)]へ