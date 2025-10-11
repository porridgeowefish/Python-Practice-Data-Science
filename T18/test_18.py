# 计算机视觉实验二源代码在此：

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
# 定义一个VGG blokc，相当于把卷积操作模块化，打包了！
def vgg_block(conv_num,input,output):
    vgg = []
    for i in range(conv_num):
        if(i==0):
            vgg.append(nn.Conv2d(input,output,3,padding=1))
        else:
            vgg.append(nn.Conv2d(output,output,3,padding=1))
        vgg.append(nn.BatchNorm2d(num_features=output))
        vgg.append(nn.ReLU())
    vgg.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*vgg)


class CNN(nn.Module): 
    def __init__(self):
        super().__init__()
        self.block1 = vgg_block(2,3,128) # 两层卷积层
        self.block2 = vgg_block(2,128,128) # 两层卷积层
        self.GAP = nn.AdaptiveAvgPool2d((1,1))  
        self.w1 = nn.Linear(128,512) 
        self.bn4 = nn.BatchNorm1d(num_features=512) 
        self.w2 = nn.Linear(512,10)  
    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.GAP(x) 
        x = x.view(-1, 128) 
        x = self.w1(x)
        x = self.bn4(x) # 全连接层也使用BatchNorm，加快收敛速度！
        x = F.leaky_relu(x)
        x = F.dropout(x,p=0.5) # 使用dropout，可能导致收敛速度变慢！
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
optimizer = optim.Adam(CNN_module.parameters(),lr=0.001,weight_decay=0.00001) # L2正则化，使用交叉损失熵计算的同时加上L2范数，对模型复杂度的惩罚。
epoch = 60 
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = epoch)
loss_avg = []
Accuracy = []
test_loss = []
def test_eval(i):
    print(f"第{i+1}epoch展示测试结果:")
    loss_tem = []
    CNN_module.eval()
    correct = 0
    with torch.no_grad():
        for f,l in testloader:
            f=f.to(device)
            l=l.to(device)
            y = CNN_module(f)
            loss = cost(y,l)
            loss_tem.append(loss.item())
            # --下面计算准确率
            softmax = nn.Softmax(dim=1)
            result = softmax(y)
            predict = torch.max(result,dim=1)
            label_predict = predict.indices
            correct_pre = torch.sum(label_predict==l) 
            correct += correct_pre.item()
    print(f"Loss in {i+1}个 epoch(Test set):{np.mean(loss_tem)}")
    test_loss.append(np.mean(loss_tem))
    tem = correct/len(testset)
    tem = tem*100
    print(f"第{i+1}个epoch准确率为:{tem}%")
    Accuracy.append(tem)
    CNN_module.train()

for i in range(epoch):
    losses = []
    CNN_module.train()
    for f,l in trainloader:
        f=f.to(device) # 数据放进GPU
        l=l.to(device) 
        y = CNN_module(f) # 前向传播
        loss = cost(y,l) # 算损失
        optimizer.zero_grad() # 梯度清零
        loss.backward() # 反向传播
        optimizer.step() # 更新参数，由优化器自己做
        losses.append(loss.item())
    scheduler.step() # 使用调度器更新参数
    loss_avg.append(np.mean(losses)) 
    if((i+1)%5==0):
        print(f"In {i+1} epoch: the loss is:{np.mean(losses)}")
    if((i+1)%10==0):
        test_eval(i)
        

torch.save(CNN_module,".//T18//model.pth")

loss_y = [10,20,30,40,50,60] 
epochs = range(epoch) # 把epoch变成list，使得能够画图
plt.plot(epochs,loss_avg,label='Train_loss',color = "blue")
plt.plot(loss_y,test_loss,label='Test_loss',color = "red")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.grid(True)
plt.legend() # 这个legend是用来显示图例的，上面用到了label，这里就需要这个了。
plt.savefig(".//T18//loss.png")
plt.show()
plt.plot(loss_y,Accuracy)
plt.xlabel("Epochs")
plt.ylabel("Accuracy(%)")
plt.title("Prediction Accuracy")
plt.grid(True)
plt.savefig(".//T18//Ac.png")
plt.show()

