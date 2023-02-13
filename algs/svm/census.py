import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

# Load dataset
df = pd.read_csv('census.csv')

# Preprocessing
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

# Sample dataset (5% subset) 
pct = .15 
df = df.sample(int(len(df)*pct)) 

print("N: ", len(df))

y = df['class']
X = df.drop('class', axis=1)

# Split data into training and testing 70:30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

C = [.00001, .0001, .001, .01, .1, 1, 10, 100]
kernels = ['linear', 'poly', 'sigmoid']

print("starting...")
for k in kernels:
    print(k)
    acc = []
    roc_auc = []
    for c in C:
        clf = SVC(C = c, kernel=k, probability=True)
        print(c)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        acc.append(accuracy_score(y_test, pred))
        roc_auc.append(roc_auc_score(y_test, pred))

    plt.figure()
    plt.title(str(k) + " Census ROC AUC Scores")
    plt.xscale("log")
    plt.plot(C, roc_auc)
    plt.savefig(str(k) + "_census_roc_auc.png")

    plt.figure()
    plt.title(str(k) + " Census Accuracy Scores")
    plt.xscale("log")
    plt.plot(C, acc)
    plt.savefig(str(k) + "_census_acc.png")

    ri = np.argmax(roc_auc)
    print("Max ROC AUC = ", roc_auc[ri])
    ai = np.argmax(acc)
    print("Max Acc = ", acc[ai])

