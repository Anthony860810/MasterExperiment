from inspect import Parameter
import os
from typing import MutableSequence
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


import torch
import torch.nn as nn
from torch.nn.modules.linear import Linear
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import TensorDataset

Folder_PATH = "../YahooBenchmark/A4Benchmark/"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 10
lr = 0.001
epochs = 200

class TrendDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x)
        self.y = torch.tensor(y)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        return self.x[index], self.y[index]

class ConvNet(nn.Module):
    def __init__(self):  # override __init__
        super(ConvNet, self).__init__() # 使用父class的__init__()初始化網路
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 5, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv1d(5, 10, kernel_size=5, stride=2),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(4180,1680)
        )
    def forward(self, x):
        out = self.layer1(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out




def ReadFile(FolderPath):
    x=[]
    y=[]
    for fileName in os.listdir(Folder_PATH):
        if(fileName[0:14]=="A4Benchmark-TS"):
            file = os.path.join(Folder_PATH,fileName)
            df = pd.read_csv(file)
            x.append(df["value"])
            y.append(df["trend"])
    x = np.array(x)
    y = np.array(y)
    return x, y

def TrainTestSplit(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    return x_train, x_test, y_train, y_test


def Evaluation(y, y_hat):
    print("MSE: "+ str(nn.MSELoss(y, y_hat)))
    print("MAE: "+ str(nn.L1Loss(y, y_hat)))
    #print("R2_score: "+ str(r2_score(y, y_hat)))

if __name__ == '__main__':
    x,y = ReadFile(Folder_PATH)
    x_train, x_test, y_train, y_test = TrainTestSplit(x,y)

    torch_train_dataset = TrendDataset(x_train, y_train)
    torch_test_dataset = TrendDataset(x_test, y_test)
    loader_train = DataLoader(
            dataset=torch_train_dataset,
            batch_size=batch_size,
            num_workers=4,
            shuffle=True)
    loader_test = DataLoader(
            dataset=torch_test_dataset,
            batch_size=batch_size,
            num_workers=4,
    )
    model = ConvNet()
    model.to(device, dtype=torch.double)
    loss_func = nn.L1Loss()
    optimizer = torch.optim.Adamax(model.parameters(), lr=lr)
    ## Train
    loss_list=[]
    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss=0.0
        for batch_idx, (data, target) in enumerate(loader_train):
            data, target = data.to(device), target.to(device)
            data = data.view(batch_size, 1 , 1680)
            y_hat = model(data)
            target = target.to(dtype=float)
            loss = loss_func(y_hat, target)
            running_loss+=loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("Epoch_" + str(epoch) + "_loss: " + str(running_loss) )
        loss_list.append(running_loss)
    ## Test
    predict = []
    mse=0
    mae=0
    for batch_idx, (data, target) in tqdm(enumerate(loader_test)):
            data, target = data.to(device), target.to(device)
            data = data.view(batch_size, 1 , 1680)
            y_hat=model(data)
            mse += nn.MSELoss()(target, y_hat)
            mae += nn.L1Loss()(target, y_hat)
    print("MSE: "+str(mse.item()))
    print("MAE: "+str(mae.item()))
    ##Evaluation(target, predict)
    plt.plot(loss_list)
    plt.show()