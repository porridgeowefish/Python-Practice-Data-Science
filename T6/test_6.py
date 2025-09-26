import pandas as pd

def func_one(Dframe,age):
    return Dframe.query(f"Age>{age}") # query可以使用一个字符串，进行一些类似数据库的查询。

def func_two(Df,N):
    return Df[Df["Name"]==N]


frame = pd.DataFrame(
    {
        "City": ["深圳","广州","阳朔"],
        "Name": ["Andrew","Andrew_one","Andrew_two"],
        "Age":[21,22,23]
    }
,index = ["P1","P2","P3"])

ans_one = func_one(frame,21)
ans_two = func_two(frame,"Andrew")
print(ans_one)
print(ans_two)