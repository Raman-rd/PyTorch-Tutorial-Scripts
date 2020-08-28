import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt



transform = transforms.ToTensor()

train_data = datasets.MNIST(root="../Data",download=True,train=True,transform=transform)
test_data = datasets.MNIST(root="../Data",download=True,train=False,transform=transform)

train_loader = DataLoader(train_data,batch_size=10,shuffle=True)
test_loader = DataLoader(test_data,batch_size=10,shuffle=False)


class covnet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1,6,3,1)
        self.conv2 = nn.Conv2d(6,16,3,1)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,2,2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2,2)
        x = x.view(-1,5*5*16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.log_softmax(x,dim=1)
torch.manual_seed(101)    
model = covnet()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

import time 
start_time = time.time()

epochs = 5
train_losses= []
test_losses =[]
train_correct =[]
test_correct = []

for i in range(epochs):
    trn_corr = 0
    tst_corr = 0

    for b,(X_train,y_train) in enumerate(train_loader):

        b+=1

        y_pred = model(X_train)
        loss = criterion(y_pred,y_train)

        predicted = torch.max(y_pred.data,1)[1]
        batch_corr = (predicted == y_train).sum()
        trn_corr += batch_corr

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if b%600 == 0:
            print(f'epoch: {i:2}  batch: {b:4} [{10*b:6}/60000]  loss: {loss.item():10.8f}  \
accuracy: {trn_corr.item()*100/(10*b):7.3f}%')

    train_losses.append(loss)
    train_correct.append(trn_corr)


    with torch.no_grad():
        for b,(X_test,y_test) in enumerate(test_loader):

            y_val = model(X_test)

            predicted = torch.max(y_val.data,1)[1]
            tst_corr+= (predicted == y_test).sum()

    
    loss= criterion(y_val,y_test)
    test_losses.append(loss)
    test_correct.append(tst_corr)


        
print(f'\nDuration: {time.time() - start_time:.0f} seconds') # print the time elapsed            


plt.plot(train_losses)
plt.plot(test_losses)
plt.legend()
plt.title("loss")
plt.show()


test_load_all = DataLoader(test_data,batch_size=10000,shuffle=False)

with torch.no_grad():

    correct = 0
    for X_test,y_test in test_load_all:
        y_val = model(X_test)
        predicted = torch.max(y_val,1)[1]
        correct += (predicted==y_test).sum()

print(correct.item()/len(test_data))