import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import sklearn.datasets
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score

# Load dataset
df = pd.read_csv('census.csv')

categorical_atts = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
continuous_atts = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

# One hot enconding
for att in categorical_atts:
    onehot = pd.get_dummies(df[att])
    if '?' in onehot.columns:
        onehot = onehot.drop('?', axis=1)
    df = df.drop(att, axis=1)
    df = df.join(onehot)

# Standardization
scale = StandardScaler()
cont = df[continuous_atts]
scale.fit(cont)
scaled = scale.transform(cont)
scaled = pd.DataFrame(scaled, index=cont.index, columns=cont.columns)
df[continuous_atts] = scaled

# Label binarization
lb = LabelBinarizer()
binary = pd.DataFrame(lb.fit_transform(df['class']), index=df['class'].index, columns=['class'])
df['class'] = binary

pct = .5
df = df.sample(int(len(df)*pct))

y = df['class']
X = df.drop('class', axis=1)


# importance = dt.feature_importances_
# plt.bar([x for x in range(len(importance))], importance)
# ind = np.argpartition(importance, -10)[-10:]
# i_feat = data_full.columns[ind]
# data_full = data_full[i_feat]

# Split data into training and testing 70:30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# Pruning example
# https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html#sphx-glr-auto-examples-tree-plot-cost-complexity-pruning-py
path = DecisionTreeClassifier().cost_complexity_pruning_path(X_train, y_train)
alphas, impurities = path.ccp_alphas, path.impurities

# Create trees for different alphas
dts = []
roc_aucs = []
i = 1
for alpha in alphas:
    dt = DecisionTreeClassifier(ccp_alpha=alpha)
    dt.fit(X_train, y_train)
    pred = dt.predict(X_test)
    probs = dt.predict_proba(X_test)
    roc_aucs.append(roc_auc_score(y_test, probs[:,1]))
    dts.append(dt)

# Removing the case where tree pruned to only one node
dts = dts[:-1]
alphas = alphas[:-1]
roc_aucs = roc_aucs[:-1]

train_scores = [dt.score(X_train, y_train) for dt in dts]
test_scores = [dt.score(X_test, y_test) for dt in dts]

# Create accuracy vs alpha plots
fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
ax.plot(alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
ax.legend()
plt.savefig("census_ava.png")

plt.figure()
plt.title("ROC AUC vs. Alpha")
plt.ylabel("ROC AUC")
plt.xlabel("Alpha")
plt.plot(alphas, roc_aucs)
plt.savefig("census_rocaucs.png")

i = np.argmax(roc_aucs)
alpha = alphas[i]
print("Max ROC AUC Alpha: ", alpha)
dtree = DecisionTreeClassifier(ccp_alpha = alpha)

# Split data into training and testing 70:30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
dtree.fit(X_train, y_train)
score = dtree.score(X_test, y_test)
print("Accuracy: ", score)
