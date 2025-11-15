import string
import torch
UNK = 1
def read_file(path):
    with open(path,'r',encoding = 'utf-8') as f:
        result = f.read()
    translator = str.maketrans('','',string.punctuation)
    result = result.translate(translator)
    result = result.lower()
    result = result.replace('\n',' ')
    return result

def to_vocab(text):
    text = text.split()
    word_list = list(dict.fromkeys(text))
    word_fill = ['<pad>','unk']
    word_list = word_fill + word_list
    vocabulary = {word:index for index,word in enumerate(word_list)}
    return vocabulary

def to_Embedding_input(text,vocab,param):
    if(param == 0):
        text = text.split()
    # print(text)
    text = [vocab.get(i,UNK) for i in text]
    text = torch.tensor(text,dtype = torch.long)
    return text

def to_word_list(text):
    translator = str.maketrans('','',string.punctuation)
    text = text.translate(translator)
    text = text.lower()
    text = text.replace('\n',' ')
    text = text.split()
    return text

def to_list_sentence(path):
    with open(path,'r',encoding = 'utf-8') as f:
        result = f.read()
    translator = str.maketrans('','',string.punctuation)
    result = result.translate(translator)
    result = result.lower()
    result = result.splitlines()
    result = [a for a in result if a!='']
    return(result)
