import pandas as pd

def func(Dframe,pram):
    return Dframe[pram] # 返回行信息，使用.loc[pram],如果是想用数组样的索引，可以使用iloc[]


frame = pd.DataFrame(
    {
        "City": ["深圳","广州","阳朔"],
        "Name": ["Andrew","Andrew_one","Andrew_two"],
        "Age":[21,22,23]
    }
,index = ["P1","P2","P3"])

tmp = func(frame,"Name")
print(tmp)
print(f"其类型是{type(tmp)}")
print(frame.head(3)) # 获取前n列，n传入head

# 了解了Pandas 最基础的两种数据结构，序列和dataframe。
# Pandas最重要的特征是索引，我们可以通过json，csv，或者字典生成一个dataframe（本质是一个表），这个表会有各种属性。可以指定索引，详细可以运行本代码，探究一下