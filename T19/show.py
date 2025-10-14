import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import torchvision
from torchvision import datasets,transforms,models
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn



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
picture_view = DataLoader(trainset,batch_size = 10, shuffle = True) # 提取数据
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


model = torch.load(".//T19//model.pth",weights_only=False)

show_model(model,10)

plt.show()
