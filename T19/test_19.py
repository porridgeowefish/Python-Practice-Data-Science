import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import torchvision
from torchvision import datasets,transforms,models
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn


Size = 224
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
    transforms.Resize(Size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)) # CIFAR-10均值标准差
])
transform_test = transforms.Compose([
    transforms.Resize(Size),
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
    

def show_model(model,test_num = 6):
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
                ax.set_title(f"Predict:{class_name[result[j]]}  Answer:{class_name[lable[j]]}")
                showimg(input.detach().cpu()[j]) # 使用detach是一个更好的方法，移出计算图
                if(test_num==image_count):
                    model.train(mode=is_training)
                    return
        model.train(mode=is_training)


inputs,classes = next(iter(picture_view))
out = torchvision.utils.make_grid(inputs)
showimg(out,title = [class_name[x] for x in classes])
plt.show()

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) 
for param in model.parameters():
    param.requires_grad = False

num = model.fc.in_features# 这是在resnet的model中定义的
model.fc = nn.Linear(num,10)
model = model.to(device)
cost = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(),lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=15)
epoch = 20
loss_avg = []
Accuracy = []
test_loss = []
def test_eval(i):
    print(f"第{i+1}epoch展示测试结果:")
    loss_tem = []
    model.eval()
    correct = 0
    with torch.no_grad():
        for f,l in testloader:
            f=f.to(device)
            l=l.to(device)
            y = model(f)
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
    model.train()

for i in range(epoch):
    losses = []
    model.train()
    for f,l in trainloader:
        f=f.to(device) # 数据放进GPU
        l=l.to(device) 
        y = model(f) # 前向传播
        loss = cost(y,l) # 算损失
        optimizer.zero_grad() # 梯度清零
        loss.backward() # 反向传播
        optimizer.step() # 更新参数，由优化器自己做
        losses.append(loss.item())
    scheduler.step() # 使用调度器更新参数
    loss_avg.append(np.mean(losses)) 
    print(f"In {i+1} epoch: the loss is:{np.mean(losses)}")
    if((i+1)%5==0):
        test_eval(i)

torch.save(model,".//T19//model.pth")

loss_y = [5,10,15]
epochs = range(epoch) # 把epoch变成list，使得能够画图
show_model(model,test_num=10)
plt.show()
plt.plot(epochs,loss_avg,label='Train_loss',color = "blue")
plt.plot(loss_y,test_loss,label='Test_loss',color = "red")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.grid(True)
plt.legend() # 这个legend是用来显示图例的，上面用到了label，这里就需要这个了。
plt.savefig(".//T19//loss.png")
plt.show()
plt.plot(loss_y,Accuracy)
plt.xlabel("Epochs")
plt.ylabel("Accuracy(%)")
plt.title("Prediction Accuracy")
plt.grid(True)
plt.savefig(".//T19//Ac.png")
plt.show()  

# In 1 epoch: the loss is:1.128518447089378
# In 2 epoch: the loss is:0.8388138305195763
# In 3 epoch: the loss is:0.7959261896360256
# In 4 epoch: the loss is:0.7731772202360051
# In 5 epoch: the loss is:0.7577929662926423
# 第5epoch展示测试结果:
# Loss in 5个 epoch(Test set):0.6318638611443436
# 第5个epoch准确率为:78.61%
# In 6 epoch: the loss is:0.7546959168008526
# In 7 epoch: the loss is:0.7448171393950577
# In 8 epoch: the loss is:0.7365307790391585
# In 9 epoch: the loss is:0.7337617107364528
# In 10 epoch: the loss is:0.7308984091672142
# 第10epoch展示测试结果:
# Loss in 10个 epoch(Test set):0.6098719129833994
# 第10个epoch准确率为:78.94%
# In 11 epoch: the loss is:0.7217906925379468
# In 12 epoch: the loss is:0.7158803442859893
# In 13 epoch: the loss is:0.723057499017252
# In 14 epoch: the loss is:0.7114191809883508
# In 15 epoch: the loss is:0.7135936009609486
# 第15epoch展示测试结果:
# Loss in 15个 epoch(Test set):0.6049525360517864
# 第15个epoch准确率为:79.10000000000001%
# In 16 epoch: the loss is:0.7132633945826069
# In 17 epoch: the loss is:0.7147559068544441
# In 18 epoch: the loss is:0.7187649482656318
# In 19 epoch: the loss is:0.7173347323751815
# In 20 epoch: the loss is:0.719469146198019
# 第20epoch展示测试结果:
# Loss in 20个 epoch(Test set):0.6101234464705745
# 第20个epoch准确率为:78.77%