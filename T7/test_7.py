import torch

a = torch.ones(2,3)
b = torch.rand(3,4)

c = a@b

print(c)