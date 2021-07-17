# -*- coding: utf-8 -*-
"""
Implementation of binary neural network classifier for binary classification 
of IMDB movie reviews (chapter 3.4 of Chollet's "Deep learning with Python")
using Pytorch
"""

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%% Load training and testset
print("Load datasets")
from torchtext.datasets import IMDB

train_set = IMDB(root="./data/IMDB",split="train")

#%% Generate vocabulary from dataset
print("Generate vocabulary")

from collections import Counter
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from tqdm import tqdm

counter = Counter()
tokenizer = get_tokenizer("basic_english")
for label, text in tqdm(train_set):
    tokens = tokenizer(text)
    counter.update(tokens)
    
MAX_TOKENS = 10000

vocabulary = Vocab(counter=counter,max_size=(MAX_TOKENS-2),min_freq=1)   
    
#%% translate text to one hot encoding

def collate_one_hot(batch, vocabulary, tokenizer, max_tokens, label_dict, device):
    """One hot encode texts in batch"""
    one_hot_text_list = []
    one_hot_label_list = []
    for label, text in batch:
        one_hot_text = torch.zeros((max_tokens))
        idcs = [vocabulary[token] for token in tokenizer(text)]
        one_hot_text[idcs] = 1
        one_hot_text_list.append(one_hot_text)
        
        idx = label_dict[label]
        one_hot_label_list.append(idx)

    one_hot_label_list= torch.tensor(one_hot_label_list,dtype=torch.float32)
    one_hot_text_list = torch.cat(one_hot_text_list)

    return one_hot_text_list.to(device), one_hot_label_list.to(device)


#%% Train NN
print("Prepare dataset")

from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

BATCH_SIZE = 64

train_set, test_set = IMDB(root="./data/IMDB",split=("train","test"))
train_set_list = list(train_set)
test_set_list = list(test_set)

NUM_VALID = int(len(test_set)/2)
test_set_list, valid_set_list = random_split(test_set_list, 
                                             [len(test_set_list)- 
                                              NUM_VALID, NUM_VALID])

label_dict = {"neg":0,"pos":1}
collate_fn = lambda batch: collate_one_hot(batch, vocabulary, tokenizer, 
                                           MAX_TOKENS, label_dict, device)

train_loader = DataLoader(dataset=train_set_list, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(dataset=valid_set_list, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(dataset=test_set_list, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=collate_fn)


#%% Linear neural network for binary classification
print("Define neural network")

from torch import nn
import torch.nn.functional as F

class BinaryTextClassifier(nn.Module):
    
    def __init__(self, max_tokens):
        super(BinaryTextClassifier,self).__init__()
        self.max_tokens = max_tokens
        self.fc1 = nn.Linear(in_features=max_tokens,out_features=16)
        self.do1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=16,out_features=16)
        self.do2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(in_features=16,out_features=1)
        
    def forward(self,x):
        x = x.view(-1,self.max_tokens)
        x = F.relu(self.fc1(x))
        x = self.do1(x)
        x = F.relu(self.fc2(x))
        x = self.do2(x)
        x = torch.sigmoid(self.fc3(x))
        
        return x
    
net = BinaryTextClassifier(MAX_TOKENS).to(device)


#%% Train model
print("Train model")

from torch import optim as optim
from helpers import train_model

NUM_EPOCHS = 5
LEARNING_RATE = 0.001

optimizer = optim.RMSprop(net.parameters(),lr=LEARNING_RATE)
criterion = nn.BCELoss()

net, history_dict = train_model(net, device, NUM_EPOCHS, optimizer, criterion, 
                                train_loader, valid_loader, BATCH_SIZE, 
                                classifier="binary")

#%% Plot results
print("Plot results")
from helpers import plot_results

plot_results(history_dict)
