import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader,Dataset 
import torch.nn

def func(x):
    return (x/3)* np.sqrt(abs(np.sin(x)))

class Sets(Dataset): # 定义数据装载器
    def __init__(self,x,y):
        self.x = torch.tensor(x,requires_grad=True,dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y,requires_grad=True,dtype=torch.float32).unsqueeze(1)
    def __len__(self):
        return len(self.x)
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx] # 确保返回是张量


x_origin = np.linspace(-15,15,1000) # 生成数据
x_test = np.linspace(-15,15,200) # 生成测试数据

y_origin = func(x_origin)
Set = Sets(x_origin,y_origin)
Loader = DataLoader(Set,shuffle=True,batch_size=50) # 使用随机梯度下降，可以更好解决非凸函数拟合问题(跳出局部最优解)
hidden_size = 30
net = torch.nn.Sequential(    # 这是一种流水线模型，其实也可以通过更加简单基础方法定义（详细可以看12题目）
    torch.nn.Linear(1,hidden_size),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_size,1)
)
losses = []
cost = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(),lr=0.001) # 使用一个优化器，这个东西帮我做梯度更新

for i in range(100000): # epoch 为 500
    batch_loss = []
    for idx,(x,y) in enumerate(Loader):
        result = net(x)
        loss = cost(result,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.data.numpy())
    if(i%10==0):
        losses.append(np.mean(batch_loss))
        print(f"In epoch {i}: loss is: {np.mean(batch_loss)}")


# 注意这里没有进入eval 模式
x_test = torch.tensor(x_test,requires_grad=True,dtype=torch.float32).unsqueeze(1)
y_test = net(x_test)
x_test = x_test.detach().numpy() # 从计算图上剥离，然后转换成ndarray
y_test = y_test.detach().numpy()
plt.scatter(x_origin,y_origin,s=10,c="blue",marker='o',label = "F(x)") # 散点图组件
plt.scatter(x_test,y_test,s=10,c="red",marker='x',label='G(x)')
plt.plot(x_origin,y_origin) # 直线图组件
plt.xlabel = "x"
plt.ylabel = "y"
plt.grid(True)
plt.show()
