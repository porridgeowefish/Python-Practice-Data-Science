import torch
from torch import nn

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

