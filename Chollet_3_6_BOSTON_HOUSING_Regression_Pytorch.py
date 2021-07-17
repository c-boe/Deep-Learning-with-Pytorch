# -*- coding: utf-8 -*-
"""
Implementation of neural network regression model for Bosting housing dataset
(chapter 3.6 of Chollet's "Deep learning with Python") using Pytorch
(The length of the train and test set differ from the book)

dataset:
    https://people.sc.fsu.edu/~jburkardt/datasets/boston_housing/boston_housing.html
"""
import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
#%%
print("Define Dataset")

from custom_datasets import BH
from custom_datasets import ToTensor
    
trainset = BH(root="./data/BH", train=True, transform=ToTensor(), download=True)

#%% NN Model
print("Define neural network regression model")
import torch.nn as nn
import torch.nn.functional as F

class LinearRegressionModel(nn.Module):

    def __init__(self,len_features):
        super(LinearRegressionModel,self).__init__()
        self.len_features = len_features
        self.fc1 = nn.Linear(self.len_features, 64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,1)
        
    def forward(self,x):
        x.view(-1,self.len_features)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x

#%%  accuracy helper

def calcualte_mae(net, data_loader):
    "Calculate mean absolute error "
    net.eval()
    mae = 0
    length= 0
    for data in data_loader:
        length += len(data[0])
        features, labels = data
        features = features.to(device)
        labels = labels.to(device)
        pred_labels = net(features)
        mae+=torch.sum(torch.abs(pred_labels-labels.view(labels.shape[0],1)))
    mae=mae/length
    return mae

#%% train model
print("K-fold validation")
import torch.optim as optim
from tqdm import tqdm

LEN_FEATURES = 13
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 100
K_FOLD = 3

net = LinearRegressionModel(LEN_FEATURES).to(device)
optimizer = optim.RMSprop(net.parameters(),lr=LEARNING_RATE)
criterion = nn.MSELoss()

all_scores = []
scores_k = []
for i in range(K_FOLD): # K-fold cross validation
    print("\n Processing fold: {} \n".format(i + 1))
    folds = list(range(K_FOLD))
    folds.pop(i)
    
    trainset = BH(root="./data/BH", train=True, transform=ToTensor(), normalize=True)#
    train = trainset.partition(K_FOLD, curr_parts=folds)
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=BATCH_SIZE)#, shuffle=True,drop_last=True
    
    validset = BH(root="./data/BH", train=True, transform=ToTensor(), normalize=True)#
    valid = validset.partition(K_FOLD, curr_parts=[i])
    valid_loader = torch.utils.data.DataLoader(
        valid, batch_size=BATCH_SIZE)

    net.train()
    score_per_ep = []
    for epoch in tqdm(range(EPOCHS)):
        for data in train_loader:
            optimizer.zero_grad()
            
            features, labels = data
            features = features.to(device)
            labels = labels.to(device)

            output = net(features)
        
            loss = criterion(output,labels.view(labels.shape[0],1))  
            
            loss.backward()
            optimizer.step()
        score_per_ep.append(calcualte_mae(net, valid_loader))
    scores_k.append(score_per_ep)
    val_score = calcualte_mae(net, valid_loader)

    all_scores.append(val_score)        

print("Scores: {}".format(all_scores))

#%%
print("Plot results")
import matplotlib.pyplot as plt

scores_k = torch.tensor(scores_k)
score_per_episode = torch.sum(scores_k,axis=0)/K_FOLD

plt.figure()
plt.plot(range(EPOCHS), score_per_episode)
plt.ylabel("Validation MAE")
plt.xlabel("Epochs")

