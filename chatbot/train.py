import json
from nltk_utils import tokenize,stem,bag_of_words;
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset , DataLoader  


with open('intents.json','r') as f:
    intents=json.load(f)
#print(intents)
all_words=[]
tags=[]
xy=[]
for intent in intents['intents']:
    tag=intent['tag']
    tags.append(tag)
    for pattren in intent['patterns']:
        w=tokenize(pattren)
        all_words.extend(w)
        xy.append((w, tag))
ignore_words=['?','!','.',',']
all_words=[stem(w) for w in all_words if w not in ignore_words]
all_words=sorted(set(all_words))
tags=sorted(set(tags)   )
print(tags)

x_train=[]
y_train=[]
for(pattern_sentece, tag) in xy:
    bag=bag_of_words(pattern_sentece,all_words)
    x_train.append(bag)

    label=tags.index(tags)
    y_train.append(x_train)
    
x_train=np.array(x_train)
y_train=np.array(y_train)

class ChatDataSet(Dataset):
    def _init_(self):
        self.n_samples=len(x_train)
        self.x_data=x_train
        self.y_data=y_train
    def _getitem_(self, index):
        return self.x_data[index],self.y_data[index]
    def _len_(self):
        return self.n_samples
 #HyperPrameters   
 batch_size=8   

Dataset = ChatDataSet()
train_loader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=2)