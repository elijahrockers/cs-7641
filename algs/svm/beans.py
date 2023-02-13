import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

# Load dataset
df = pd.read_csv('beans.csv')

# Sample dataset (5% subset) 
pct = .15
df = df.sample(int(len(df)*pct))

print("N: ", len(df))

# Preprocessing
X = df.drop('Class', axis = 1)
attrs = X.columns.tolist()
stdscaler = StandardScaler()
X[attrs] = stdscaler.fit_transform(X[attrs])
le = LabelEncoder()
df['Class'] = le.fit_transform(df.Class.values)
mapping = dict(zip(le.classes_, range(len(le.classes_))))
y = df['Class']

# Split data into training and testing 70:30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

C = [.00001, .0001, .001, .01, .1, 1, 10, 100]
kernels = ['linear', 'poly', 'sigmoid']

for k in kernels:
    print(k)
    acc = []
    roc_auc = []
    for c in C:
        print(c)
        clf = SVC(C = c, kernel=k, probability=True)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        acc.append(accuracy_score(y_test, pred))
        roc_auc.append(roc_auc_score(y_test, y_proba, multi_class='ovo'))

    plt.figure()
    plt.title(str(k) + " Beans ROC AUC Scores")
    plt.xscale("log")
    plt.plot(C, roc_auc)
    plt.savefig(str(k) + "_beans_roc_auc.png")

    plt.figure()
    plt.title(str(k) + " Beans Accuracy Scores")
    plt.xscale("log")
    plt.plot(C, acc)
    plt.savefig(str(k) + "_beans_acc.png")

    ri = np.argmax(roc_auc)
    print("Max ROC AUC = ", roc_auc[ri])
    ai = np.argmax(acc)
    print("Max Acc = ", acc[ai])
