import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

depth_color_map=[(1, "black", "depth=1"),
                 (2, "yellow", "depth=2"), 
                 (3, "red", "depth=3"), 
                 (4, "blue", "depth=4"),
                 (5, "green", "depth=5"),
                 (6, "brown", "depth=6")]
N_EST=400

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

# Sample 50% of full data set
pct = 1
df = df.sample(int(len(df)*pct))

y = df['class']
X = df.drop('class', axis=1)

# Split data into training and testing 70:30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

for depth, color, label in depth_color_map:

    t0 = timeit.default_timer()
    ada = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=depth), n_estimators=N_EST, learning_rate=1
    )

    ada.fit(X_train, y_train)
    accuracy = []
    for predict_test in ada.staged_predict(X_test):
        accuracy.append(accuracy_score(y_test, predict_test))

    t1 = timeit.default_timer()

    elapsed = round((t1 - t0), 3)
    plt.plot(range(1, N_EST + 1), accuracy, c=color, label=label + ", " + str(elapsed) + "sec")

plt.legend()
plt.savefig("census_accuracy.png")
