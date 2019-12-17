import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_path = "data"
train_data_path = os.path.join(data_path, "train.csv")
test_data_path = os.path.join(data_path, "test.csv")
test_labels_data_path = os.path.join(data_path, "gender_submission.csv")

train_set = pd.read_csv(train_data_path)
test_set = pd.read_csv(test_data_path)

# Preprocessing
train_set = train_set.dropna(subset=["Embarked"])
train_labels = train_set["Survived"].copy()
train_set = train_set.drop(["PassengerId", "Name", "Ticket", "Cabin","Survived"], axis=1)

test_set = test_set.drop(["PassengerId", "Name", "Ticket", "Cabin"],axis=1)
test_labels = pd.read_csv(test_labels_data_path)[["Survived"]].copy()

titanic_num = train_set[["Pclass", "Age", "SibSp", "Parch", "Fare"]]
titanic_cat = train_set[["Sex", "Embarked"]]

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

pipeline_cat = Pipeline([("onehot", OneHotEncoder())])

pipeline = ColumnTransformer(
    [("num", pipeline_num, list(titanic_num)), ("cat", pipeline_cat, list(titanic_cat))]
)

train_prepared = pipeline.fit_transform(train_set)
test_prepared = pipeline.fit_transform(test_set)


###
#Define Model
###
import torch
import torch.nn.functional as F

useCUDA = True if torch.cuda.is_available() else False
device = torch.device("cuda:0") if useCUDA else torch.device("cpu")

class TitanicNet(torch.nn.Module):
    def __init__(self):
        super(TitanicNet,self).__init__()
        self.lin1 = torch.nn.Linear(10,30)
        self.lin2 = torch.nn.Linear(30,30)
        self.lin3 = torch.nn.Linear(30,30)
        self.out = torch.nn.Linear(30,1)

    def forward(self,x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        return torch.sigmoid(self.out(x))

#Convert data to torch tensors
train_prepared = torch.from_numpy(train_prepared)
train_labels = torch.from_numpy(np.asarray(train_labels,dtype=np.float64))

x = train_prepared
y = train_labels

x = x.type(torch.FloatTensor)
y = y.type(torch.FloatTensor)

x = x.to(device)
y = y.to(device)

#Initialise model
model = TitanicNet().cuda(device=device) if useCUDA else TitanicNet()
loss_fn = torch.nn.MSELoss(reduction='mean')
learning_rate = 1e-4

import time
start = time.time()

#Train Model
for t in range(2000):
    y_pred = model(x)
    y_pred = torch.reshape(y_pred,(-1,))
    loss = loss_fn(y_pred,y)

    if t % 100 == 99:
        print(t, loss.item())

    model.zero_grad()
    loss.backward()

    #Using SGD
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

end = time.time()
print(end - start, useCUDA)

#Test model
eval_fn = loss_fn

test_prepared = torch.from_numpy(test_prepared)
test_labels = torch.from_numpy(np.asarray(test_labels,dtype=np.float64))

x = test_prepared
y = test_labels

x = x.type(torch.FloatTensor)
y = y.type(torch.FloatTensor)

x = x.to(device)
y = y.to(device)

y_pred = model(x)

score = torch.sqrt(eval_fn(y_pred,y))

print("Score:",score.item())
