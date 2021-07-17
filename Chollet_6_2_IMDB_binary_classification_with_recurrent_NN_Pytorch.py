# -*- coding: utf-8 -*-
"""
Implementation of recurrent neural network classifier for binary 
classification of movie reviews with recurrent RNNs and LSTMs (chapter 6.2 
of Chollet's "Deep learning with Python") using Pytorch
"""

import torch 
from torchtext.datasets import IMDB

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#%% import dataset
print("Load IMDB data set")

MAX_TOKENS = 10000
MAX_LEN = 500
BATCH_SIZE = 128
validation_split=0.2

train_set, test_set = IMDB(root="./data/IMDB",split=("train","test"))

#%% generate vocabulary
print("Generate vocabulary")

from helpers import gen_vocab
from torchtext.data.utils import get_tokenizer

tokenizer = get_tokenizer(tokenizer="basic_english")

vocabulary = gen_vocab(dataset=train_set, tokenizer=tokenizer,
                       max_tokens=MAX_TOKENS, max_len=MAX_LEN)
    
#%% translate texts to index list for embedding layer
print("Preprocess data set")

from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from helpers import collate_emb

train_set, test_set = IMDB(root="./data/IMDB",split=("train","test"))
train_set_list = list(train_set)

NUM_TRAIN_SAMPLES = len(train_set_list)
NUM_VALID_SAMPLES = int(NUM_TRAIN_SAMPLES*validation_split)
label_dict = {"neg":0,"pos":1}

collate_batch = lambda batch: collate_emb(batch, vocabulary, tokenizer, 
                                          MAX_LEN, label_dict,  device)

train_set_list, valid_set_list = random_split(train_set_list, 
                                             [NUM_TRAIN_SAMPLES - 
                                              NUM_VALID_SAMPLES,
                                              NUM_VALID_SAMPLES])

train_loader = DataLoader(dataset=train_set_list,batch_size=BATCH_SIZE,
                          drop_last=True, shuffle=True, 
                          collate_fn=collate_batch)
valid_loader = DataLoader(dataset=valid_set_list, batch_size=BATCH_SIZE,
                          shuffle=True, collate_fn=collate_batch)

#%% Recurrent neural net
print("Define neural network with recurrent layer")
import torch.nn as nn

class RecNN(nn.Module):
    "Neural network with recurrent layer"
    def __init__(self,  max_tokens, max_len, emb_dim, hidden_size):

        super(RecNN,self).__init__()
        self.max_len =  max_len
        self.hidden_size = hidden_size
        self.EmbLayer = nn.Embedding(num_embeddings=max_tokens,
                                     embedding_dim=emb_dim)
        self.RecLayer = nn.RNN(input_size=emb_dim,hidden_size=hidden_size)
        self.flatten = nn.Flatten(start_dim=1,end_dim=-1)
        self.fcl = nn.Linear(in_features=self.max_len*hidden_size
                             ,out_features=1)

    def forward(self,x):
        x = x.view(-1,self.max_len)
        x = self.EmbLayer(x)
        x, xh = self.RecLayer(x)
        x = self.flatten(x)
        x = self.fcl(x)
        x = torch.sigmoid(x)
        
        return x

#%% Train model
print("Train model")

import torch.optim as optim
from helpers import train_model

EMB_DIM = 32
HIDDEN_SIZE = 1
LEARNING_RATE = 1e-2

net = RecNN(MAX_TOKENS,MAX_LEN, EMB_DIM,HIDDEN_SIZE).to(device)
optimizer = optim.RMSprop(params=net.parameters(),lr=LEARNING_RATE)
criterion = torch.nn.BCELoss()

NUM_EPOCHS = 10

net, history_dict = train_model(net, device, NUM_EPOCHS, optimizer, criterion, 
                                train_loader, valid_loader, BATCH_SIZE,
                                classifier="binary", calc_acc = True)

#%% Plot results
print("Plot results")
from helpers import plot_results

plot_results(history_dict)

##############################################################################
#%% ------------------------ Using LSTM's ------------------------------------
##############################################################################
print("Define neural network with LSTM Layer")

import torch.nn as nn

class LSTM_NN(nn.Module):
    "Neural network with LSTM Layer"
    def __init__(self,  max_tokens, max_len, emb_dim, hidden_size):

        super(LSTM_NN,self).__init__()
        self.max_len =  max_len
        self.EmbLayer = nn.Embedding(num_embeddings=max_tokens,
                                      embedding_dim=emb_dim)
        self.LSTMLayer = nn.LSTM(input_size=emb_dim,hidden_size=hidden_size)
        self.flatten = nn.Flatten(start_dim=1,end_dim=-1)
        self.fcl = nn.Linear(in_features=self.max_len*hidden_size,out_features=1)

    def forward(self,x):
        x = x.view(-1,self.max_len)
        x = self.EmbLayer(x)
        x, xh = self.LSTMLayer(x)
        x = self.flatten(x)
        x = self.fcl(x)
        x = torch.sigmoid(x)
        
        return x

#%% Train model
print("Train model")

import torch.optim as optim
from helpers import train_model

EMB_DIM = 32

net = LSTM_NN(MAX_TOKENS,MAX_LEN, EMB_DIM, HIDDEN_SIZE).to(device)
optimizer = optim.RMSprop(params=net.parameters(),lr=LEARNING_RATE)
criterion = torch.nn.BCELoss()

NUM_EPOCHS = 10

net, history_dict = train_model(net, device, NUM_EPOCHS, optimizer, criterion, 
                                train_loader, valid_loader, BATCH_SIZE,
                                classifier="binary", calc_acc = True)

#%% Plot results
print("Plot results")
from helpers import plot_results

plot_results(history_dict)