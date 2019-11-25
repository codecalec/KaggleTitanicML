import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_path = "data"
train_data_path = os.path.join(data_path, "train.csv")
test_data_path = os.path.join(data_path, "test.csv")

train_set = pd.read_csv(train_data_path)

titanic = train_set.copy()

survived = titanic[titanic["Survived"] != 0]

def plot_hist_feature(feature, BINS=10):
    plt.hist(titanic[feature], bins=BINS)
    plt.hist(survived[feature], bins=BINS)
    plt.show()

# Preprocessing
train_set = train_set.dropna(subset=["Embarked"])
print(train_set.notna().describe())
titanic = train_set.drop("Survived", axis=1)

titanic = titanic.drop("Name", axis=1)
titanic = titanic.drop("PassengerId", axis=1)
titanic = titanic.drop("Ticket",axis=1)
titanic = titanic.drop("Cabin",axis=1)

titanic_labels = train_set["Survived"].copy()

titanic_num = titanic[["Age","SibSp","Parch","Fare"]]
titanic_cat = titanic[["Sex","Embarked"]]

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

pipeline_num = Pipeline([
    ("iterimp", IterativeImputer(missing_values=float("nan"))),
    ("minmax", MinMaxScaler())
    ])

transformed = pipeline_num.fit_transform(titanic_num)
np.savetxt("output_num",transformed)

pipeline_cat = Pipeline([
    ("onehot", OneHotEncoder())
    ])
#transformed = pipeline_cat.fit_transform(titanic_cat)
#np.savetxt("output_cat",transformed)

pipeline = ColumnTransformer([
    ("num", pipeline_num, list(titanic_num)),
    ("cat", pipeline_cat, list(titanic_cat))
    ])

titanic_processed = pipeline.fit_transform(titanic)
np.savetxt("output_final",titanic_processed)
print(titanic_processed[0:5][0:])
print(pipeline.get_params)
print(pipeline.named_transformers_)
print(list(titanic_cat))
