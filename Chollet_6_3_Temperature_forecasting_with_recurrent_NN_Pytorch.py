# -*- coding: utf-8 -*-
"""
Implementation of recurrent neural network classifier for temperature-
forecasting (chapter 6.3 of Chollet's "Deep learning with Python") using 
Pytorch
"""

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%% Loading and preparing the dataset
print("Load and prepare dataset")

from custom_datasets import JenaClimate
from custom_datasets import ToTensor
from torch.utils.data import DataLoader

LOOK_BACK = 1440
STEP = 6
DELAY = 144
BATCH_SIZE = 128

transform = ToTensor()
path = "./data"

dataset = JenaClimate(root=path, download = True, transform=transform)
LEN_DATASET = len(dataset)

train_set = dataset.subset(normalize=True,  lookback=LOOK_BACK, delay=DELAY, 
                        step=STEP, min_index=0, max_index=20000)
valid_set = dataset.subset(normalize=True, lookback=LOOK_BACK, delay=DELAY, 
                        step=STEP, min_index=200001, max_index=300000)
test_set = dataset.subset(normalize=True, lookback=LOOK_BACK, delay=DELAY, 
                        step=STEP, min_index=300001, max_index=LEN_DATASET)

train_loader = DataLoader(dataset=train_set,batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_set,batch_size=BATCH_SIZE)


#%% Define neural network
print("Define neural network")
import torch.nn as nn
import torch.nn.functional as F

class DenseNN(nn.Module):
    
    def __init__(self, feature_size, lookback, step):
        super(DenseNN,self).__init__()
        self.feature_size = feature_size
        self.lookback = lookback
        self.step = step
        
        self.flatten = nn.Flatten(start_dim=1,end_dim=-1)
        self.fcl1 = nn.Linear(in_features=int(lookback/step*feature_size),
                              out_features=32)
        self.fcl2 = nn.Linear(in_features=32,out_features=1)
        
    def forward(self,x):
        x= x.view(-1,int(self.lookback/self.step),self.feature_size)
        x = self.flatten(x)
        x = self.fcl1(x)
        x = F.relu(x)
        x = self.fcl2(x)
        
        return x
FEATURE_SIZE = 13    
HIDDEN_SIZE = 32  
net = DenseNN(FEATURE_SIZE, lookback=LOOK_BACK,step=STEP).to(device)

#%% Train model
print("Train model")
from helpers import train_model
import torch.optim as optim


NUM_EPOCHS = 10
STEPS_PER_EPOCH = 500
LEARNING_RATE = 1e-3

optimizer = optim.RMSprop(params=net.parameters(),lr=LEARNING_RATE)
criterion = torch.nn.L1Loss()# MAE loss

net, history_dict = train_model(
    net, device, NUM_EPOCHS, optimizer, criterion, train_loader, valid_loader, 
    BATCH_SIZE, STEPS_PER_EPOCH, calc_acc = False)


#%% Plot results
print("Plot results")
from helpers import plot_results

plot_results(history_dict, calc_acc = False)

##############################################################################
#%% ------------------------------ GRU Model ---------------------------------
##############################################################################
print("Define neural network with GRU layer")
class GRUNet(nn.Module):
    "Neural network with GRU layer"
    def __init__(self, feature_size, hidden_size, lookback, step):
        super(GRUNet,self).__init__()
        self.feature_size = feature_size
        self.lookback = lookback
        self.step = step

        self.gru = nn.GRU(input_size=feature_size, hidden_size=hidden_size)
        self.fcl1 = nn.Linear(in_features=hidden_size,out_features=1)
        
    def forward(self,x):
        x= x.view(-1,int(self.lookback/self.step),self.feature_size)
        x, xh = self.gru(x)
        x = self.fcl1(x)
        x = x[:,-1,:]
        return x
    
FEATURE_SIZE = 13    
HIDDEN_SIZE = 32
net = GRUNet(FEATURE_SIZE, HIDDEN_SIZE,lookback=LOOK_BACK,step=STEP).to(device)


#%% Train model
print("Train model")
from helpers import train_model
import torch.optim as optim

NUM_EPOCHS = 10
STEPS_PER_EPOCH = 500

optimizer = optim.RMSprop(params=net.parameters(),lr=LEARNING_RATE)
criterion = torch.nn.L1Loss()# MAE loss

net, history_dict = train_model(
    net, device, NUM_EPOCHS, optimizer, criterion, train_loader, valid_loader, 
    BATCH_SIZE, STEPS_PER_EPOCH, calc_acc = False)

#%% Plot results
print("Plot results")

from helpers import plot_results

plot_results(history_dict, calc_acc = False)

