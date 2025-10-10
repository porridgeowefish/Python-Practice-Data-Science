import torchvision
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F # 方便使用relu和最大池化。
import matplotlib.pyplot as plt

# 数据增强和归一化 
transform_train = transforms.Compose([
 transforms.RandomHorizontalFlip(p=0.5),
 transforms.RandomRotation(degrees=10),
 transforms.ToTensor(),
 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)) # CIFAR-10均值标准差
])
transform_test = transforms.Compose([
 transforms.ToTensor(),
 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])
# 加载数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
transform=transform_test)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
testloader = DataLoader(testset, batch_size=128, shuffle=False)
# 3 x 32 x 32 的结构定义卷积神经网络
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size=(3,3),padding=1)
        self.conv2 = nn.Conv2d(64,128,kernel_size=(3,3),padding=1)
        self.conv3 = nn.Conv2d(128,128,kernel_size=(3,3),padding=1)
        self.GAP = nn.AdaptiveAvgPool2d((1,1))
        self.w1 = nn.Linear(128,128) 
        self.w2 = nn.Linear(128,10) 
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x),2)) 
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = self.GAP(x)
        x = x.view(-1, 128) 
        x = F.relu(self.w1(x))
        x = self.w2(x)
        return x
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA:",torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("CUDA not available,using CPU")
cost = nn.CrossEntropyLoss()
CNN_module = CNN().to(device)
optimizer = optim.Adam(CNN_module.parameters(),lr=0.001)
epoch = 100
loss_avg = []
for i in range(epoch):
    losses = []
    CNN_module.train()
    for f,l in trainloader:
        f=f.to(device) # 数据放进GPU
        l=l.to(device) 
        y = CNN_module(f)
        loss = cost(y,l)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    loss_avg.append(np.mean(losses))
    if(i%5==0):
        print(f"In {i} epoch: the loss is:{np.mean(losses)}")
    if(i%25==0):
        print("展示测试结果:")
        loss_tem = []
        CNN_module.eval()
        with torch.no_grad():
            for f,l in testloader:
                f=f.to(device)
                l=l.to(device)
                y = CNN_module(f)
                loss = cost(y,l)
                loss_tem.append(loss.item())
        print(f"Loss in {i} epoch(Test set):{np.mean(loss_tem)}")
        CNN_module.train()

CNN_module.eval()
correct = 0
with torch.no_grad():
    for f,l in testloader:
        f = f.to(device)
        l = l.to(device)
        y = CNN_module(f)
        softmax = nn.Softmax(dim=1)
        result = softmax(y)
        predict = torch.max(result,dim=1)
        label_predict = predict.indices
        correct_pre = torch.sum(label_predict==l)
        correct += correct_pre.item()

print(f"准确率为:{correct/len(testset)}")

epochs = range(epoch) # 把epoch变成list，使得能够画图
plt.plot(epochs,loss_avg)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.titel("Loss Curve")
plt.grid(True)
plt.legend()
plt.savefig(".//T15//loss.png")
plt.show()

"""
Using CUDA: NVIDIA GeForce RTX 5060 Laptop GPU
In 0 epoch: the loss is:1.7744844423230652
展示测试结果:
Loss in 0 epoch(Test set):1.5624310653420943
In 5 epoch: the loss is:1.0974683964343936
In 10 epoch: the loss is:0.8739456287430375
In 15 epoch: the loss is:0.7512791310734761
In 20 epoch: the loss is:0.6623474264236362
In 25 epoch: the loss is:0.5889437819838219
展示测试结果:
Loss in 25 epoch(Test set):0.6364969294282454
In 30 epoch: the loss is:0.5391957747661854
In 35 epoch: the loss is:0.49692814894344495
In 40 epoch: the loss is:0.4616471085402057
In 45 epoch: the loss is:0.4360313632375444
In 50 epoch: the loss is:0.41206282910788455
展示测试结果:
Loss in 50 epoch(Test set):0.5859638535523717
In 55 epoch: the loss is:0.3861588184790843
In 60 epoch: the loss is:0.3658104290056716
In 65 epoch: the loss is:0.3480990528298156
In 70 epoch: the loss is:0.33938816155466583
In 75 epoch: the loss is:0.31899331243294277
展示测试结果:
Loss in 75 epoch(Test set):0.6389940633803983
In 80 epoch: the loss is:0.3153849905332946
In 85 epoch: the loss is:0.29806044934045933
In 90 epoch: the loss is:0.28697903214208303
In 95 epoch: the loss is:0.27864288639686907
准确率为:0.8114
"""