import torch
import matplotlib.pyplot as plt
import numpy as np

def func(x):
    return 10*np.sin(x)+x**2


x = np.linspace(-15,15,100)
x_test = np.linspace(-15,15,50)

y = func(x)
x = torch.tensor(x,requires_grad=True)
y = torch.tensor(y,requires_grad=True)
print(x.shape)
print(y.shape)
n = 128
w1 = torch.randn((1,n),dtype = torch.double,requires_grad = True)  # 定义神经元权重
bias = torch.randn(n,dtype=torch.double,requires_grad=True) # 定义神经元偏置
w2 = torch.randn((n,1),dtype = torch.double,requires_grad = True) # 定义输出权重

lr = 0.0001 # 定义学习率
loss = []
x = x.unsqueeze(1)
y = y.unsqueeze(1)
for _ in range(200000):
    mid = x*w1+bias
    hidden = torch.relu(mid) 
    pre = hidden@w2
    losses = torch.mean((pre-y)**2)
    if(_%10000==0):
        print(f"Losses:{losses}")
        loss.append(losses)
    losses.backward()
    w1.data.add_(-lr*w1.grad.data) # 梯度更新，如果对值进行直接操作，一定要使用data属性
    bias.data.add_(-lr*bias.grad.data)
    w2.data.add_(-lr*w2.grad.data)
    w1.grad.data.zero_()
    w2.grad.data.zero_()
    bias.grad.data.zero_()

x_test = torch.tensor(x_test,requires_grad=True)
x_test = x_test.unsqueeze(1)
mid = x_test*w1+bias
hidden = torch.relu(mid)
y_test = hidden@w2
x_test = x_test.detach().numpy() # 从计算图上剥离，然后转换成ndarray
y_test = y_test.detach().numpy()
x = x.detach().numpy()
y = y.detach().numpy()
plt.scatter(x,y,s=10,c="blue",marker='o',label = "F(x)") # 散点图组件
plt.scatter(x_test,y_test,s=10,c="red",marker='x',label='G(x)')
plt.plot(x,y) # 直线图组件
plt.xlabel = "x"
plt.ylabel = "y"
plt.grid(True)
plt.show()
