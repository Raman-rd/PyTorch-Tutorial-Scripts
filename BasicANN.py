import torch 
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

## Init Model

class Model(nn.Module):


    def __init__(self,in_features=4,h1=8,h2=9,out=3):
        super().__init__()

        self.fc1 = nn.Linear(in_features,h1)
        self.fc2 = nn.Linear(h1,h2)
        self.out = nn.Linear(h2,out)

    def forward(self,x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.out(x))
        return x


torch.manual_seed(43)
model = Model()

df = pd.read_csv("iris.csv")

print("Printing head of the dataframe...")
print(df.head())

X = df.drop("target",axis=1).values
y = df["target"].values

X_train , X_test , y_train , y_test = train_test_split( X , y , test_size=0.20 , random_state=42)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

epochs = 1000
losses = []

for i in range(epochs):

    i = i+1
    y_pred = model.forward(X_train)
    loss = criterion(y_pred,y_train)

    losses.append(loss)

    if i % 10 == 0 :
        print(f"Epoch is {i} and loss is {loss}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    

plt.plot(range(epochs),losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


## Prediction Time

with torch.no_grad():

    y_eval = model.forward(X_test)

    loss = criterion(y_eval,y_test)
 
correct = 0
with torch.no_grad():

     for i , data in enumerate(X_test):
         my_val = model.forward(data)

         print(f"{data} {str(my_val)} {y_test[i]}")

         if my_val.argmax().item() == y_test[i]:

             correct = correct +1

print(correct)

print(len(y_test))