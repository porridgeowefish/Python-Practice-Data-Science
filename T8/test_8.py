import torch

x = torch.tensor(3.0,requires_grad=True)
y = 2*x**2+2*x
y.backward()
print(f"手动梯度计算值为：{4*x.item()+2}")
print(f"自动梯度计算值为：{x.grad}")