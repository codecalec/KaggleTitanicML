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
titanic = train_set.drop("Survived", axis=1)

titanic = titanic.drop("Name", axis=1)
titanic = titanic.drop("PassengerId", axis=1)
titanic = titanic.drop("Ticket", axis=1)
titanic = titanic.drop("Cabin", axis=1)

titanic_labels = train_set["Survived"].copy()

titanic_num = titanic[["Pclass", "Age", "SibSp", "Parch", "Fare"]]
titanic_cat = titanic[["Sex", "Embarked"]]

from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

pipeline_num = Pipeline(
    [
        ("iterimp", IterativeImputer(missing_values=float("nan"))),
        ("minmax", MinMaxScaler()),
    ]
)

transformed = pipeline_num.fit_transform(titanic_num)
np.savetxt("output_num", transformed)

pipeline_cat = Pipeline([("onehot", OneHotEncoder())])

pipeline = ColumnTransformer(
    [("num", pipeline_num, list(titanic_num)), ("cat", pipeline_cat, list(titanic_cat))]
)

titanic_prepared = pipeline.fit_transform(titanic)
np.savetxt("output_final", titanic_prepared)

# Linear Model
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()

#Logistic Model
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()

# Random Forest Model
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor(n_estimators=100)

# Support Vector Machine
from sklearn.svm import SVC

svm = SVC(kernel="rbf", gamma="scale")

# Evaluation
training_data = pd.read_csv("./data/test.csv")
training_data = training_data.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
training_prepared = pipeline.transform(training_data)

training_labels = pd.read_csv("./data/gender_submission.csv")[["Survived"]].copy()

def evaluate_model(model_name, model, labels, data, training_labels, training_data):
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import cross_val_score

    print("Fitting model: " + model_name)
    model.fit(data, labels)
    print("Evaluating", model_name, "on training data")
    data_predict = model.predict(data)
    rmse = np.sqrt(mean_squared_error(labels, data_predict))
    print("RMSE:", rmse)
    scores = cross_val_score(
        model, data, labels, scoring="neg_mean_squared_error", cv=10
    )
    scores = np.sqrt(-scores)
    print("RMSE CV:\n", "Mean", scores.mean(), "\n STD:", scores.std())

    print("Evaluating", model_name, "on test data")
    data_predict = model.predict(training_data)
    rmse = np.sqrt(mean_squared_error(training_labels, data_predict))
    print("RMSE:", rmse)
    print("---------\n")

models = (("Linear_Model",linear_reg),("Log_Model",log_reg),("Random_Forest",forest_reg),("SVM",svm))

for name,model in models:
    evaluate_model(
        name,
        model,
        titanic_labels,
        titanic_prepared,
        training_labels,
        training_prepared,
    )
