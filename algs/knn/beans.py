import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
df = pd.read_csv('beans.csv')

########### EXPLORATORY DATA ANALYSIS
### sample 5 random attributes (except class)
### sample 20 points from each class
#df_sample = df.groupby('Class').apply(lambda x: x.sample(40))
#col_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
#df_sample = df_sample.sample(5, axis=1, weights=col_weights)
#df_sample = df_sample.reset_index()
#df_sample = df_sample.drop("level_1", axis=1)
# 
#plt.close()
#sns.set_style("whitegrid")
#sns.pairplot(df_sample, hue="Class", height=3, plot_kws={"s": 3});
#plt.show()

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

plt.figure()
K = range(1, 200)
acc = []
for k in K:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    test_pred = knn.predict(X_test)
    acc.append(accuracy_score(y_test, test_pred))

plt.plot(K, acc)
plt.savefig("beans_acc_v_k.png")

# max_acc_k = np.argmax(acc)
# 
# knn = KNeighborsClassifier(n_neighbors = max_acc_k)
# knn.fit(X_train, y_train)
# test_pred = knn.predict(X_test)



