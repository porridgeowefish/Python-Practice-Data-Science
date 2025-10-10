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
class CNN(nn.Module): # 模型总参数量为1792+78356+147584+16512+1290+3*256+128 = 246430个参数，少于50w
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size=(3,3),padding=1) # 第一卷积层 参数个数是：64*3*3*3+64 = 1792个参数
        self.bn1 = nn.BatchNorm2d(num_features=64) # gamma+beta 参数有128
        self.conv2 = nn.Conv2d(64,128,kernel_size=(3,3),padding=1) # 第二卷积层 参数个数是：128*64*3*3 + 128 = 73856 个参数
        self.bn2 = nn.BatchNorm2d(num_features=128) # gamma+beta 参数有256
        self.conv3 = nn.Conv2d(128,128,kernel_size=(3,3),padding=2,dilation=2) # 加入空洞卷积,相当于扩大了卷积核，为了维持图片是原大小,填充是2
        self.bn3 = nn.BatchNorm2d(num_features=128) # gamma+beta 参数有256      # 第三层卷积层，参数个数是128*128*3*3+128 = 147584 个参数
        self.GAP = nn.AdaptiveAvgPool2d((1,1))  
        self.w1 = nn.Linear(128,128) # 参数个数 128*128 + 128 = 16512
        self.bn4 = nn.BatchNorm1d(num_features=128) # gamma+beta 参数有256
        self.w2 = nn.Linear(128,10)  # 参数个数 128*10 + 10 = 1290
    def forward(self,x):
        x = self.conv1(x) # 卷积
        x = self.bn1(x)  # 归一化
        x = F.leaky_relu(x) # 激活
        x = F.avg_pool2d(x,2,stride=2) # 池化 由32 x 32 -> 16*16

        x = self.conv2(x)
        x = self.bn2(x) 
        x = F.leaky_relu(x)
        x = F.avg_pool2d(x,2,stride=2) # 16 x 16 -> 8 x 8

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x)
        x = F.avg_pool2d(x,2,stride=2) # 8 x 8 -> 4 x 4
        
        x = self.GAP(x) # 全局平均池化 
        x = x.view(-1, 128)  # 128个输入特征
        x = self.w1(x)
        x = self.bn4(x) # 全连接层也使用BatchNorm，加快收敛速度！
        x = F.leaky_relu(x)
        x = F.dropout(x,p=0.3) # 使用dropout，可能导致收敛速度变慢！
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
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.5) # 引入学习率调度器。
epoch = 60 
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
        

torch.save(CNN_module,".//T17//model.pth")

loss_y = [10,20,30,40,50,60] 
epochs = range(epoch) # 把epoch变成list，使得能够画图
plt.plot(epochs,loss_avg,label='Train_loss',color = "blue")
plt.plot(loss_y,test_loss,label='Test_loss',color = "red")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.grid(True)
plt.legend() # 这个legend是用来显示图例的，上面用到了label，这里就需要这个了。
plt.savefig(".//T17//loss.png")
plt.show()
plt.plot(loss_y,Accuracy)
plt.xlabel("Epochs")
plt.ylabel("Accuracy(%)")
plt.title("Prediction Accuracy")
plt.grid(True)
plt.savefig(".//T17//Ac.png")
plt.show()


'''
Using CUDA: NVIDIA GeForce RTX 5060 Laptop GPU
In 5 epoch: the loss is:0.8039234285159489
In 10 epoch: the loss is:0.6188474852410729
In 15 epoch: the loss is:0.517682204389816
In 20 epoch: the loss is:0.4472316009614169
第20epoch展示测试结果:
Loss in 20 epoch(Test set):0.6068094451970691
In 25 epoch: the loss is:0.34906699563688637
In 30 epoch: the loss is:0.3197782166931025
In 35 epoch: the loss is:0.284204633568254
In 40 epoch: the loss is:0.26484448948632117
第40epoch展示测试结果:
Loss in 40 epoch(Test set):0.5562257031096688
In 45 epoch: the loss is:0.21580256895183603
In 50 epoch: the loss is:0.19719089036021392
In 55 epoch: the loss is:0.18862867310566975
In 60 epoch: the loss is:0.18239814521330397
第60epoch展示测试结果:
Loss in 60 epoch(Test set):0.5508445331567451
准确率为:0.8514
'''