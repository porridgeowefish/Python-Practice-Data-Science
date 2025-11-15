from sklearn.model_selection import train_test_split # 这是新东西，这个代码主要就是引入了对数据的切分和标准化处理
from sklearn.preprocessing import StandardScaler
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader,Dataset
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
import os

curr_dir = os.getcwd()

data = pd.read_csv(os.path.join(curr_dir,"Data//near-earth-comets.csv"))
input = 9
output = 1
list_drop = ["A1","A2","A3","DT","ref","Object_name","Object"]
data = data.drop(list_drop,axis = 1)
Y = data["P"]
X = data.drop("P",axis=1)
print(f"Shape of Y:{Y.shape}")
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state=42)
trans = StandardScaler()
X_train = trans.fit_transform(X_train) # 特别注意，这个函数相当于fit+transfrom，fit代表用X_train计算均值和标准差
X_test = trans.transform(X_test)       # 这个时候不能再fit，而是用保存在类内部的参数进行标准化
# Q:那么为什么不能让X_test计算自己的标准差呢？
# 测试集应该是完全未知的，标准化也是训练流程的一部分。这个标准化过程泄露，也是泄露了总体分布。容易导致对模型预测能力做出过分乐观估计。


class comet_dataset(Dataset):
    def __init__(self,x,y):
        self.features = torch.tensor(x,dtype=torch.float32)
        lable = y.values
        lable = lable.reshape(-1,1) # -1 代表 自行推断行的数目，1代表转换成一列。
        self.lables = torch.tensor(lable,dtype=torch.float32)
    def __len__(self):
        return len(self.features)
    def __getitem__(self,idx):
        return self.features[idx],self.lables[idx]

n_num = 128
class Comet_prediction(nn.Module): 
    def __init__(self,input_shape,output_shape):
        super().__init__()
        self.w1 = nn.Linear(input_shape,n_num)
        self.relu1 = nn.ReLU()
        self.w2 = nn.Linear(n_num,256)
        self.relu2 = nn.ReLU()
        self.w3 = nn.Linear(256,output_shape)
    def forward(self,x):
        x = self.w1(x)
        x = self.relu1(x)
        x = self.w2(x)
        x = self.relu2(x)
        x = self.w3(x)
        return x

        
dataset = comet_dataset(X_train,Y_train)
testset = comet_dataset(X_test,Y_test)
comet_loader = DataLoader(dataset,batch_size=20)
comet_tester = DataLoader(testset,batch_size=20) 
cost = torch.nn.MSELoss()
Modul = Comet_prediction(input,output)
optimizer = optim.Adam(Modul.parameters(),lr=0.001,weight_decay=1e-5)
loss_list = []
for i in range(10001):
    lost_record_per_epoch = []
    for idx,(f,l) in enumerate(comet_loader):
        x = Modul(f)
        loss = cost(x,l)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lost_record_per_epoch.append(loss.item()) # 使用item()获取具体数值。
    loss_list.append(np.mean(lost_record_per_epoch))
    if(i%1000==0):
        Modul.eval() # 评估模式，非常重要！
        with torch.no_grad(): # 上下文管理器，一个非常重要的Python机制，这里主要是不记录梯度。
            test_loss = []
            for idx,(f,l) in enumerate(comet_tester):
                x = Modul(f)
                loss = cost(x,l)
                test_loss.append(loss.item())
            print(f"In {i} epoch,Training loss : {np.mean(lost_record_per_epoch)}")
            print(f"In {i} epoch,Testing loss:{np.mean(test_loss)}")
        Modul.train()

# 这是训练时候的一次报错，本质上是因为在计算cost的时候，解释器发现lables和预测值的维度不匹配。Lable还是1维，输出却是二维。最好的处理方法是在
# Dataset 的数据预处理部分就用reshape(-1,1)扩张一个维度，这样子就能够避免这个错误！
# 后续记录，当我做出修改后，模型的损失暴跌，训练效果极佳，我怀疑前面错误对损失的计算产生了干扰。

# C:\Users\xmz14_ugn3mh4\anaconda3\envs\pytorch_cpu\Lib\site-packages\torch\nn\modules\loss.py:616: UserWarning: Using a target size (torch.Size([20])) that is different to the input size (torch.Size([20, 1])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
#   return F.mse_loss(input, target, reduction=self.reduction)
# C:\Users\xmz14_ugn3mh4\anaconda3\envs\pytorch_cpu\Lib\site-packages\torch\nn\modules\loss.py:616: UserWarning: Using a target size (torch.Size([8])) that is different to the input size (torch.Size([8, 1])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
#   return F.mse_loss(input, target, reduction=self.reduction)
# C:\Users\xmz14_ugn3mh4\anaconda3\envs\pytorch_cpu\Lib\site-packages\torch\nn\modules\loss.py:616: UserWarning: Using a target size (torch.Size( )) that is different to the input size (torch.Size([12, 1])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
#   return F.mse_loss(input, target, reduction=self.reduction)