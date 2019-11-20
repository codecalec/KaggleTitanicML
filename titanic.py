import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_path = "data"
train_data_path = os.path.join(data_path,"train.csv")
test_data_path = os.path.join(data_path,"test.csv")

train_set = pd.read_csv(train_data_path)

titanic = train_set.copy()

print(titanic.head())

survived = titanic[titanic["Survived"] != 0]

def plot_hist_feature(feature,BINS=10):
    plt.hist(titanic[feature],bins=BINS)
    plt.hist(survived[feature],bins=BINS)
    plt.show()

plot_hist_feature("Age",BINS=30)
plot_hist_feature("Fare",BINS=30)

titanic = train_set.drop("Survived", axis =1)
titanic_labels = train_set["Survived"].copy()
