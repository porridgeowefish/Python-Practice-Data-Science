import torch 
from torch.utils.data import Dataset,DataLoader # 导入关键的包
import os 
import pandas as pd
filename = os.getcwd()
print(filename)
file = pd.read_csv(os.path.join(filename,"Data\\near-earth-comets.csv"))
# 一个非常重要的实践，在init中完成数据的所有转换。
# 传入dataset的可以是任何类型，但是返回一定要是tensor,一般来说就是lable和features
class Comet_dataset(Dataset):
    def __init__(self,dataframe):
        lable_df = dataframe["P"]
        feature_df = dataframe.drop(["A1","A2","A3","DT","P","ref","Object_name","Object"],axis = 1)
        self.lable = torch.tensor(lable_df.values,dtype = torch.float32)
        self.feature = torch.tensor(feature_df.values,dtype=torch.float32)
    def __len__(self):
        return len(self.lable)
    def __getitem__(self,idx):
        lable_idx = self.lable[idx]
        feature_idx = self.feature[idx]
        return feature_idx,lable_idx
    
test = Comet_dataset(file)
Comet_loader = DataLoader(test,batch_size=10,shuffle=False)  #batch_size不是除数
for idx,(f,l) in enumerate(Comet_loader): # enumerate 遍历可迭代对象的同时返回一个索引，同时还可以用参数start = 指定计数器从哪个数字开始。
    print(f"第{idx}个数据集:{f}:{l}")
