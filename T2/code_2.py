import numpy as np

a_one = np.array((1,2,2,2,2))  # np的创建只用列表样子的东西就行了,可以是元组，也可以是列表！
a_two = np.array(([3,3,3,3,3]))
a_three = a_one*a_two # 运算都是重载过的，不用担心
print(a_three)