import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader,Dataset
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
import os

curr_dir = os.getcwd()

data = pd.read_csv(os.path.join(curr_dir,"T12//near-earth-comets.csv"))
input = 9
output = 1
# 搭建dataset,nn。本质上是利用pytorch中提供的工具，加上那你自己的一些组合，创建出自己的数据加载模块/训练网络。
# 这就相当于搭积木，只是底层的积木都是完备的，例如把数据转化为tensor，或者是损失函数，单层的神经元等等
list_drop = ["A1","A2","A3","DT","P","ref","Object_name","Object"]
class comet_dataset(Dataset):
    def __init__(self,data):
        y = data["P"]
        x = data.drop(list_drop,axis = 1)
        self.features = torch.tensor(x.values,dtype=torch.float32)
        self.lables = torch.tensor(y.values,dtype=torch.float32)
    def __len__(self):
        return len(self.features)
    def __getitem__(self,idx):
        return self.features[idx],self.lables[idx]

n_num = 128
class Comet_prediction(nn.Module):
    def __init__(self,input_shape,output_shape):
        super().__init__()
        self.w1 = nn.Linear(input_shape,n_num)
        self.relu = nn.ReLU()
        self.w2 = nn.Linear(n_num,output_shape)
    def forward(self,x):
        x = self.w1(x)
        x = self.relu(x)
        x = self.w2(x)
        return x

        
dataset = comet_dataset(data)
comet_loader = DataLoader(dataset,batch_size=10)
cost = torch.nn.MSELoss()
Modul = Comet_prediction(input,output)
optimizer = optim.Adam(Modul.parameters(),lr=0.001)
loss_list = []
for i in range(10000):
    lost = 0
    for idx,(f,l) in enumerate(comet_loader):
        x = Modul(f)
        loss = cost(x,l)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lost = lost + loss/16
    loss_list.append(lost)
    if(i%1000==0):
        print(f"In {i} epoch,loss : {lost}")

# 以下是一次训练过程，为什么不准确呢？（提示，数据是什么样子的？）
# In 0 epoch,loss : 20554106880.0
# In 1000 epoch,loss : 645815.5
# In 2000 epoch,loss : 29064.9609375
# In 3000 epoch,loss : 206992.21875
# In 4000 epoch,loss : 213947.125
# In 5000 epoch,loss : 57800.22265625
# In 6000 epoch,loss : 177341.8125
# In 7000 epoch,loss : 134329.75
# In 8000 epoch,loss : 176590.984375
# In 9000 epoch,loss : 20240.654296875