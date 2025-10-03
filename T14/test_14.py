import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


iris = load_iris()
print(type(iris))
X = iris.data
Y = iris.target
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
class iris_set(Dataset):
    def __init__(self,X,Y):
        self.features = torch.tensor(X,dtype = torch.float32)
        self.lables = torch.tensor(Y,dtype = torch.long)
    def __len__(self):
        return len(self.features)
    def __getitem__(self,idx):
        return self.features[idx],self.lables[idx]

hidden = 64
in_s = 4
out_s = 3
class iris_trainer(nn.Module):
    def __init__(self,in_s,out_s):
        super().__init__()
        self.forword_func = nn.Sequential(
            nn.Linear(in_s,hidden),
            nn.ReLU(),
            nn.Linear(hidden,hidden),
            nn.ReLU(),
            nn.Linear(hidden,out_s)
        )
    def forward(self,x):
        y = self.forword_func(x)
        return y

iris_dataset = iris_set(X_train,Y_train)
iris_Loader = DataLoader(iris_dataset,shuffle=True,batch_size = 10)

cost = nn.CrossEntropyLoss()
Modul = iris_trainer(in_s,out_s)

optimizer = optim.Adam(Modul.parameters(),lr = 0.001)
epoch = 1000
loss_record = []
for i in range(epoch):
    Modul.train()
    losses = []
    for idx,(f,l) in enumerate(iris_Loader):
        y = Modul(f)
        loss = cost(y,l)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    loss_record.append(np.mean(losses))
    if i%100==0:
        print(f"Loss in {i} epoch: {np.mean(losses)}")

Modul.eval()
train_load = iris_set(X_test,Y_test)
train_loader = DataLoader(train_load,batch_size = 30)
correct = 0
false = 0
with torch.no_grad():
    for idx,(f,l) in enumerate(train_loader):
        result = Modul(f)
        softmax = nn.Softmax(dim = 1)
        result = softmax(result)
        predict = torch.max(result,dim=1)
        label_predict = predict.indices
        correct_prediction = torch.sum(label_predict==l)
        correct = correct_prediction.item()
print(f"从{30}个测试集中，预测有{correct}个正确")
print(f"准确率:{correct/30.0}")



        
    
    