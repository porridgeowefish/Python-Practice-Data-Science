import torch 
from helper import to_Embedding_input,to_vocab,to_word_list,read_file,to_list_sentence
# to_Embedding_input 能够把一个字符串（经过处理后）转化为可输入词嵌入张量
# to_vocab           从一个空格分割的大字符串中提取词汇表
# to_word_list       可以把一个单词转换成单词列表
# read_file          读取文件，获取所有的单词
# to_list_sentence   用于实现词语分离
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence # 用于填充
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


PAD = 0
UNK = 1 # 定义填充符
# 读取句子，去除标点，换行，使用打包好的函数，转换成一个词汇表。
path1 = './data/emotion_prediction/good.txt'
path2 = './data/emotion_prediction/bad.txt'
result1 = read_file(path1)
result2 = read_file(path2)
sentence1 = to_list_sentence(path1)
sentence2 = to_list_sentence(path2)
# -- 下面一步的作用是打标签
lengtha = len(sentence1)
lengthb = len(sentence2)
lable_1 = torch.ones(lengtha,dtype=torch.float32) # 相当于，积极是1，消极是0
lable_2 = torch.zeros(lengthb,dtype=torch.float32)
lable = torch.cat([lable_1,lable_2],dim = 0)
print(lable)
print(lable_1.shape)
print(lable.shape)
# -- 创建词汇表
result1 = result1 +' '+ result2 # 注意这里细节
vocab = to_vocab(result1)
print(len(vocab))
# 数据特征处理
features = [] 
for sentence in sentence1:
    features.append((to_Embedding_input(sentence,vocab,0))) # 这里传入0，详细见实现
for sentence in sentence2:
    features.append((to_Embedding_input(sentence,vocab,0)))

# 这里函数设计还是有很多优化空间的，有很多函数功能重叠。

# 定义模型
class RNN_emotion(nn.Module):
    def __init__(self,input_state,hidden_state):
        super().__init__()
        self.embeddeding = nn.Embedding(num_embeddings=len(vocab),embedding_dim=64,padding_idx=PAD) # 给出padding_idx，有助于简化模型训练。不训练填充值。
        self.rnn = nn.RNN(input_size = input_state,hidden_size=hidden_state,batch_first=True)
        self.fc = nn.Linear(hidden_state,hidden_num)
        self.fc1 = nn.Linear(hidden_num,1)
        self.tanh = nn.Tanh() # 这里选了Tanh，目前没有理由吧:D
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x = self.embeddeding(x)
        x,hidden = self.rnn(x)
        hidden = hidden.squeeze(0) # 这里要压缩掉一个维度！要不然batch之间匹配不上，隐藏层是最后一个维度 1 x batch_size x 1。要把前面的压掉。
        x = self.fc(hidden)
        x = self.tanh(x)
        x = self.fc1(x)
        x = self.sigmoid(x) # 二元分类问题最后用sigmoid压缩到0，1区间是最方便的
        return x.squeeze(1) # 便于进行比较，待会可以试试影响！
    

# 接下来，我们要定义dataset，定义我们的数据集

class Emotion_set(Dataset): 
    def __init__(self,input,output):
        self.feature = input
        self.lable = output
    def __len__(self):
        return len(self.feature)
    def __getitem__(self, index):
        return self.feature[index],self.lable[index]
# 为了dataloader打包过程正常，我们需要一个辅助函数：
def helper_func(batch):
    lable = []
    sequence = []
    for f,l in batch:
        sequence.append(f)
        lable.append(l)
    padded_sequences = pad_sequence(sequence, batch_first=True, padding_value=PAD) # 相当于，我们回用PAD这个值对长度不够的
    labels = torch.stack(lable, 0)

    return padded_sequences, labels 

# 使用加速
device = torch.device("cuda") # 使用cuda加速
# 然后，创建dataloader

dataset = Emotion_set(features,lable)
emotion_loader = DataLoader(dataset,shuffle=True,batch_size=32,collate_fn=helper_func)
# 定义参数们
hidden_state = 256
hidden_num = 128
lr_rate = 0.001
loss = nn.BCELoss() # 二元分类
modul = RNN_emotion(64,hidden_state) # 前面这个参数是词嵌入的维度
optimizer = optim.Adam(modul.parameters(),lr = lr_rate)
schduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max = 500)
modul.to(device)
lost_all = []
epoch = 500
for i in range(epoch):
    modul.train()
    lost_tem = []
    for f,l in emotion_loader:
        f = f.to(device)
        l = l.to(device)
        result = modul(f)
        lost = loss(result,l)
        optimizer.zero_grad()
        lost.backward()
        optimizer.step()
        lost_tem.append(lost.item())
    schduler.step()
    lost_all.append(np.mean(lost_tem))
    if(i%50==0):
        print(f"Loss in {i} epoch is:{np.mean(lost_tem)}")

def test(num):
    modul.eval()
    modul.to(torch.device("cpu"))
    for i in range(num):
        text = input("输入一个句子,我来预测其是乐观/消极:")
        text = to_word_list(text)
        text = to_Embedding_input(text,vocab,1)
        text = text.unsqueeze(0)
        with torch.no_grad():
            l = modul(text)
            result = l.item()
            print(f"result:{result}")
            if(result==0.5):
                print("中性词")
            elif(result>0.5):
                print("积极预期!Good!")
            else:
                print("有点悲观哦,Hakuna Matata!")

x_axis = range(epoch)
plt.plot(x_axis,lost_all)
plt.xlabel("loss")
plt.ylabel("Epoch")
plt.title("Loss curve for emotion prediction")
plt.grid()
plt.savefig("./T23/loss.png")
plt.show()
torch.save(modul,".//T23//modul.pth")
test_number = 10
test(test_number)








