import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

# Load dataset
df = pd.read_csv('census.csv')

# Sample dataset (5% subset)
pct = .15
df = df.sample(int(len(df)*pct))

print("N: ", len(df))

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
    plt.savefig("census-" + str(arch) + ".png")
