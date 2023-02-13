import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn.datasets
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

# Load dataset
# (data_full, target_full) = sklearn.datasets.fetch_covtype(return_X_y=True, as_frame=True)
data_full = pd.read_csv('beans.csv')
target_full = data_full['Class']
data_full = data_full.drop('Class', axis = 1)

dt = DecisionTreeClassifier()
dt.fit(data_full, target_full)
importance = dt.feature_importances_
plt.bar([x for x in range(len(importance))], importance)

ind = np.argpartition(importance, -10)[-10:]
i_feat = data_full.columns[ind]
data_full = data_full[i_feat]

# Sample dataset (5% subset)
pct = 1
data = data_full.sample(int(len(data_full)*pct))
target = target_full[data.index]

# Split data into training and testing 70:30
split = train_test_split(data, target, test_size = 0.3)
X_train = split[0]
X_test = split[1]
y_train = split[2]
y_test = split[3]

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
    roc_aucs.append(roc_auc_score(y_test, probs, multi_class='ovo'))
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
plt.savefig("beans_ava.png")

plt.figure()
plt.title("ROC AUC vs. Alpha")
plt.ylabel("ROC AUC")
plt.xlabel("Alpha")
plt.plot(alphas, roc_aucs)
plt.savefig("beans_rocaucs.png")

i = np.argmax(roc_aucs)
alpha = alphas[i]
print("Max ROC AUC Alpha: ", alpha)
dtree = DecisionTreeClassifier(ccp_alpha = alpha)

# Split data into training and testing 70:30
split = train_test_split(data_full, target_full, test_size = 0.3)
X_train = split[0]
X_test = split[1]
y_train = split[2]
y_test = split[3]

dtree.fit(X_train, y_train)
score = dtree.score(X_test, y_test)
print("Accuracy: ", score)
