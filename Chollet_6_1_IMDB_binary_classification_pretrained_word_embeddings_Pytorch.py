# -*- coding: utf-8 -*-
"""
Implementation of neural network classifier for binary classification of
movie reviews with and without pretrained word embeddings (chapter 6.1.3 of 
Chollet's "Deep learning with Python") using Pytorch

Download GloVe from: https://nlp.stanford.edu/projects/glove
and place it in the folder "./data/GloVe"
"""

import torch
from torchtext.datasets import IMDB

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%% import dataset
print("Load IMDB data set")

MAX_LEN = 100   # max number of tokens from each text in IMDB data set
MAX_TOKENS = 10000 # max number of tokens in vocabulary

NUM_TRAIN_SAMPLES = 200 
NUM_VALID_SAMPLES = 10000
DATA_SET_LEN = NUM_TRAIN_SAMPLES + NUM_VALID_SAMPLES

data_set, test_set = IMDB(root="./data/IMDB",split=("train","test"))

#%% generate vocabulary
print("Generate vocabulary")

from helpers import gen_vocab
from torchtext.data.utils import get_tokenizer

tokenizer = get_tokenizer(tokenizer="basic_english")

vocabulary = gen_vocab(dataset=data_set, tokenizer=tokenizer,
                       max_tokens=MAX_TOKENS, max_len=MAX_LEN)
    
#%% translate texts to index list for embedding layer
print("Preprocess data set")

from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from helpers import collate_emb

data_set, test_set = IMDB(root="./data/IMDB",split=("train","test"))
data_set_list = list(data_set)

BATCH_SIZE = 32

label_dict = {"neg":0,"pos":1}

collate_emb_batch = lambda batch : collate_emb(batch, vocabulary, tokenizer, 
                                               MAX_LEN, label_dict, device)

train_set_list, split_set_list = random_split(data_set_list, 
                                             [NUM_TRAIN_SAMPLES,
                                              len(data_set_list)- 
                                              NUM_TRAIN_SAMPLES])
valid_set_list, _ = random_split(split_set_list, [NUM_VALID_SAMPLES,
                                                 len(split_set_list)- 
                                                 NUM_VALID_SAMPLES])

train_loader = DataLoader(dataset=train_set_list,batch_size=BATCH_SIZE,
                          drop_last=True, shuffle=True,
                          collate_fn=collate_emb_batch)
valid_loader = DataLoader(dataset=valid_set_list, batch_size=BATCH_SIZE,
                          shuffle=True, collate_fn=collate_emb_batch)

#%% Load and preprocess GloVe word-embeddings
print("Load GloVe from .txt-file")

import os

glove_dir = "./data/GloVe/"
glove_txt  = "glove.6B.100d.txt"
emb_index = {}

with open(os.path.join(glove_dir, glove_txt), encoding="utf8") as f:
    lines = f.readlines()

for line in lines:
    values = line.split()
    token = values[0]
    coefs_f = [float(val) for val in values[1:]]
    coefs = torch.tensor(coefs_f, dtype=torch.float32)
    emb_index[token] = coefs

#%% Preparing GloVe word embadding matrix
print("Define embedding matrix from GloVe")

EMB_DIM = 100
emd_mat = torch.zeros((MAX_TOKENS,EMB_DIM))

token_list = vocabulary.__dict__["stoi"].keys()

for count_token, token in enumerate(token_list):
    try:
        emd_mat[count_token, :] = emb_index[token]
    except:
        pass

#%% define neural net with embedding layer
print("Define neural network with pretrained embedding layer")

from torch import nn
import torch.nn.functional as F

class SeqNet(nn.Module):

    def __init__(self, max_tokens, max_len, emb_dim):
        super(SeqNet,self).__init__()
        self.max_len =  max_len
        self.max_tokens = max_tokens
        self.EmbLayer = nn.Embedding(num_embeddings=max_tokens,
                                     embedding_dim=emb_dim)
        self.flatten = nn.Flatten(start_dim=1,end_dim=-1)
        self.fcl1 = nn.Linear(in_features=max_len*emb_dim,out_features=32)
        self.fcl2 = nn.Linear(in_features=32,out_features=1)
                
    def forward(self, x):
        x = x.view(-1,self.max_len)
        x = self.EmbLayer(x)
        x = self.flatten(x)
        x = F.relu(self.fcl1(x))
        x = self.fcl2(x)
        x = torch.sigmoid(x)
        
        return x

net  = SeqNet(MAX_TOKENS, MAX_LEN, EMB_DIM)

#%% set embedding layer to GloVe
print("Set embedding layer")

net.EmbLayer.weight = nn.Parameter(emd_mat)
net.EmbLayer.requires_grad_ = False
net.to(device)

#%% Train model
print("Train model")
import torch.optim as optim
from helpers import train_model

NUM_EPOCHS = 10

optimizer = optim.RMSprop(params=net.parameters(),lr=1e-3)
criterion = torch.nn.BCELoss()

net, history_dict = train_model(net, device, NUM_EPOCHS, optimizer, criterion, 
                                train_loader, valid_loader, BATCH_SIZE,
                                classifier="binary", calc_acc = True)

#%% Plot results
print("Plot results")
from helpers import plot_results

plot_results(history_dict)

##############################################################################
#%% -------------- Training without pretrained embedding ---------------------
##############################################################################

print("Define neural network with embedding layer")

class BinaryClassifier(nn.Module):

    def __init__(self, max_tokens, max_len, emb_dim):
        super(BinaryClassifier,self).__init__()
        self.max_tokens = max_tokens
        self.max_len = max_len
        self.EmbLayer = nn.Embedding(num_embeddings=max_tokens,
                                     embedding_dim= emb_dim)
        self.flatten = nn.Flatten(start_dim=1,end_dim=-1)
        self.fcl1 = nn.Linear(in_features=max_len*emb_dim,out_features=32)
        self.fcl2 = nn.Linear(in_features=32,out_features=1)
        
    def forward(self, x):
        x = x.view(-1,self.max_len)
        x = self.EmbLayer(x)
        x = self.flatten(x)
        x = F.relu(self.fcl1(x))
        x = torch.sigmoid(self.fcl2(x))
        
        return x
    
#%% Train model
print("Train model")
net  = BinaryClassifier(MAX_TOKENS, MAX_LEN, EMB_DIM).to(device)
optimizer = optim.RMSprop(params=net.parameters(),lr=1e-3)
criterion = torch.nn.BCELoss()

net, history_dict = train_model(net, device, NUM_EPOCHS, optimizer, criterion, 
                                train_loader, valid_loader, BATCH_SIZE,
                                classifier="binary", calc_acc = True)

#%% Plot results
print("Plot results")
from helpers import plot_results

plot_results(history_dict)
