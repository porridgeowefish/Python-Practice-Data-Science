import numpy as np

test = [
    [1,2,3],
    [4,5,6],
    [7,8,9],
    [9,10,11]
]

def func(test):
    ans = test[::2,::]
    return ans

test = np.array(test)
answer = func(test)
print(answer)

# 同样考察dataframe 切片操作！