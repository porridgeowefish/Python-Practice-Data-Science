import torch
import numpy as np

def convert(array):
    tensors = torch.from_numpy(array)
    tensors = tensors * 2
    result = tensors.numpy()
    return result

a = np.array([1,2,3,4],dtype=float)
ans = convert(a)
print(ans)

# 后续，会接触detach，clone。注意，detach是从计算图中分离，如果在进行梯度运算，中途需要进行某些处理，可以使用detach。
# 注意共享内存机制，这里面numpy和tensor是共享内存的。