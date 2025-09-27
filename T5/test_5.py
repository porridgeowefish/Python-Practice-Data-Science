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
# 补充pandas的数据切片操作：
# 1. 使用[]运算符，语法和numpy一致，只能用来提取行。
# 2. 使用iloc[],可以提取行列，语法和上面相同。
# 3. 使用loc[]，可以使用标签进行提取。