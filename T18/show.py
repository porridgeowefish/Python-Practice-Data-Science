import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import torchvision
from torchvision import datasets,transforms,models
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
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
    cudnn.benchmark = True
else:
    device = torch.device("cpu")
    print("Cuda不存在,使用CPU")
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

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
transform=transform_test)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
testloader = DataLoader(testset, batch_size=128, shuffle=False)
class_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def showimg(inp,title = None):
    inp = inp.numpy().transpose((1,2,0))
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])
    inp = std*inp + mean # 标准化的逆操作
    inp = np.clip(inp,0,1) 
    plt.imshow(inp)
    if(title!=None):
        plt.title(title)
    plt.savefig(".//T19//data_display.png")

def show_model(model,test_num):
    is_training = model.training
    model.eval()
    image_count = 0
    fig = plt.figure() # 创建底层画布
    with torch.no_grad():
        for i,(feature,lable) in enumerate(testloader):
            input = feature.to(device)
            output = lable.to(device)
            result = model(input)
            # softmax = nn.Softmax(dim=1)
            # result = softmax(predict)  没必要，选最大即可。
            _,result = torch.max(result,dim=1)# result 接受的参数是indices
            for j in range(feature.size()[0]):
                image_count+=1
                ax = plt.subplot(test_num//2,2,image_count)
                ax.axis("off")
                ax.set_title(f"Predict:{class_name[result[j]]} Answer:{class_name[lable[j]]}")
                showimg(input.detach().cpu()[j]) # 使用detach是一个更好的方法，移出计算图
                if(test_num==image_count):
                    model.train(mode=is_training)
                    return
        model.train(mode=is_training)


CNN = torch.load(".//T18//model.pth",weights_only=False)

show_model(CNN,10)

plt.show()
