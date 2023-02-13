import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

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

y = df['class']
X = df.drop('class', axis=1)

# Split data into training and testing 70:30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

plt.figure()
K = range(1, 200)
acc = []
for k in K:
    sys.stdout.write("\rTesting k = %i/200" % k)
    sys.stdout.flush()
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    test_pred = knn.predict(X_test)
    acc.append(accuracy_score(y_test, test_pred))

plt.plot(K, acc)
plt.savefig("census_acc_v_k.png")

# max_acc_k = np.argmax(acc)
# 
# knn = KNeighborsClassifier(n_neighbors = max_acc_k)
# knn.fit(X_train, y_train)
# test_pred = knn.predict(X_test)



