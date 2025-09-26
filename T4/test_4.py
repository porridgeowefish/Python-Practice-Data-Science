# import numpy as np
# def checkers(N):
#     b = np.zeros((N,N))
#     marker = 0;
#     for i in range(N):
#         if(marker==0):
#             b[0][i] = 0
#             marker = 1
#         else:
#             b[0][i] = 1
#             marker = 0
#     for i in range(1,N):
#         for j in range(N):
#             if(b[i-1][j]==1):
#                 b[i][j] = 0
#             else:
#                 b[i][j] = 1
#     return b

# number = int(input())
# ans = checkers(number)
# print(ans)


# ------------------ 请你对上面代码做出评价

# 上面这个代码，属于典型的能跑，但是完全不是学过numpy的人应该写出的代码。Numpy的核心优势是向量化操作，即一次操作多个数。这是一个更加完美的实践！

import numpy as np

def func(num):
    a = np.zeros((num,num))
    a[1::2,::2] = 1 
    a[::2,1::2] = 1
    return a

number = int(input())
ans = func(number)
print(ans)

# 重点在于掌握切片的语法
# 针对每一个行/列，都有三个操作数 x:y:z
# x ：start 切片的启示点，默认是0
# y : end 切片的终点，默认到结束
# z : step 步长，这里步长是2。
# 由此，可以看懂上面操作了！