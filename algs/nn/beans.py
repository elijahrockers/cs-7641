import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

# Load dataset
df = pd.read_csv('beans.csv')

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

for arch in [(10,10), (10, 10, 10), (10, 10, 10, 10), (1000, 10), (500, 500, 10)]:
    print("")
    print("Architecture: ", arch)
    losses = []
    acts = []
    for act in ['identity', 'logistic', 'tanh', 'relu']:
        mlp = MLPClassifier(
                hidden_layer_sizes = arch,
                max_iter = 500,
                activation = act)

        mlp.fit(X_train, y_train)
        pred = mlp.predict(X_test)
        print("activation: ", act)
        print('current loss computed with the loss function: ',mlp.loss_)
        acts.append(act)
        losses.append(mlp.loss_)

    plt.figure()
    plt.title("Architecture: " + str(arch))
    plt.ylabel("Loss")
    plt.ylim(0, 1)
    plt.bar(acts, losses)
    plt.savefig("beans-" + str(arch) + ".png")
