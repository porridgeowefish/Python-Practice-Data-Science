import torch
from torch import nn

class vocab_rnn(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.rnn = nn.RNN(input_size,16,batch_first=False)
    def forward(self,x):
        x,hidden = self.rnn(x)
        return x,hidden


word_list = ['i',"love","her","but",'she',"loves","someone","else","who","was","her","highschool","classmate",'i','give','up','but','the','seed','of','sadness','was','placed','within','my','heart','i','want','to','feel','the','warmth','of','the','world','nothing','for','me','except','few','of','it','from','mom','dad','perhaps']

word = list(dict.fromkeys(word_list))
print(word)
word_number = len(word)
print(f"共有{word_number}个单词")

embedding = nn.Embedding(num_embeddings=word_number,embedding_dim=10)

vocabulary = {word:index for index,word in enumerate(word)} # 创建词汇表。

my_word = ['she','loves','me']

input_value = [vocabulary[i] for i in my_word]
input_value = torch.tensor(input_value,dtype=torch.long)

out = embedding(input_value)

print(out) # 这里会输出三个单词的矢量矩阵，一般来说特征会更多的。

out = out.unsqueeze(1)

rnn = vocab_rnn(10)
result,hidden = rnn(out)
print(hidden)
print(f"shape:{hidden.shape}")
print(result)
print(f"shape:{result.shape}")

