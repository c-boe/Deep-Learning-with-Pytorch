# -*- coding: utf-8 -*-
"""
Implementation Convolutional neural network classifier for multiclass
classification of MNIST data set (chapter 5.1 of Chollet's 
"Deep learning with Python") using Pytorch
"""

import torch
import torchvision
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%% Import and preprocess MNIST dataset
print("Define Dataset")

BATCH_SIZE = 64

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5,),std=(0.5))])

train_set = torchvision.datasets.MNIST(root="./data/",transform=transform,
                                      download=False,train=True)
test_set = torchvision.datasets.MNIST(root="./data/",transform=transform,
                                      download=False,train=False)

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE,
                                             shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE,
                                            shuffle=True)

#%% Neural network for multiclass classification
print("Define neural network for multiclass classification")
from torch import nn
import torch.nn.functional as F

class MultiClassifier(nn.Module):
    
    def __init__(self):
        super(MultiClassifier,self).__init__()
        self.cl1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=(3,3))
        self.mp1 = nn.MaxPool2d(kernel_size=(2,2))
        self.cl2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3))
        self.mp2 = nn.MaxPool2d(kernel_size=(2,2))
        self.cl3 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3))
        self.flatten = nn.Flatten(start_dim=1,end_dim=-1)
        self.fc1 = nn.Linear(in_features=3*3*64, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=10)
        
    def forward(self,x):
        x = F.relu(self.cl1(x)) 
        x = self.mp1(x)
        x = F.relu(self.cl2(x)) 
        x = self.mp2(x)
        x = F.relu(self.cl3(x)) 
        x = self.flatten(x)
        x = F.relu(self.fc1(x)) 
        x = self.fc2(x)
        return x
 
net = MultiClassifier().to(device) 

#%% Train model
print("Train model")

from torch import optim as optim
from helpers import train_model

NUM_EPOCHS = 5
LEARNING_RATE = 0.001

optimizer = optim.RMSprop(net.parameters(),lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

net, history_dict = train_model(net, device, NUM_EPOCHS, optimizer, criterion, 
                                train_loader, test_loader, BATCH_SIZE,
                                classifier="multi")

#%% Plot results
print("Plot results")
from helpers import plot_results

plot_results(history_dict)
