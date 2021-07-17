# -*- coding: utf-8 -*-
"""
Implementation of basic neural network classifier for multiclass classification 
of MNIST data set (chapter 2.1 of Chollet's "Deep learning with Python")
using Pytorch
"""

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%% Import MNIST dataset from torchvision
print("Load and prepare MNIST dataset")
import torchvision
from torchvision import transforms

BATCH_SIZE =  128

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0,),std=(1,))])#
train_set = torchvision.datasets.MNIST(root="./data/",train=True,
                                       transform=transform,download=True)
test_set = torchvision.datasets.MNIST(root="./data/",train=False,
                                       transform=transform,download=False)

train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                         batch_size=BATCH_SIZE,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                        batch_size=BATCH_SIZE,shuffle=True)

#%% NN model
print("Define neural network for multiclass classification")
import torch.nn.functional as F
import torch.nn as nn

class MultiClassifier(nn.Module):
    """Linear NN classifier"""
    def __init__(self, num_inp_feat, num_classes, num_neurons):
        super(MultiClassifier,self).__init__()
        self.num_inp_feat = num_inp_feat
        self.fcl1=nn.Linear(in_features=num_inp_feat,out_features=num_neurons)
        self.fcl2=nn.Linear(in_features=num_neurons,out_features=num_classes)
        
    def forward(self,x):
        x=x.view(-1,self.num_inp_feat)
        x=F.relu(self.fcl1(x)) 
        x=F.softmax(self.fcl2(x),dim=1)
    
        return x
    
NUM_INP_FEAT = len(train_set[0][0].flatten())
NUM_CLASSES =  len(test_set.classes)
NUM_NEURONS =  512
net = MultiClassifier(NUM_INP_FEAT, NUM_CLASSES, NUM_NEURONS).to(device) 

#%% Accuracy helper
def calculate_accuracy(NN, data_set_iter):
    """Calculate accuracy of Neural net predictions for given data set"""
    
    NN.eval()
    with torch.no_grad():
        correct = 0
        total=0
        for data in data_set_iter:
            features, labels = data
            features=features.to(device)
            labels=labels.to(device)
            
            output = NN(features)
            prediction = torch.argmax(output, axis=1)
                
            correct+=torch.sum(prediction == labels)
            total+=len(prediction)
        
        accuracy = float(correct/total)
    return accuracy

#%% Train model
print("Train model for multiclass classification of MNIST datasets")
from torch import optim as optim
from tqdm import tqdm

NUM_EPOCHS = 5
LEARNING_RATE = 0.001

optimizer = optim.RMSprop(net.parameters(),lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

accuracy_train=calculate_accuracy(net, train_loader)  
accuracy_test=calculate_accuracy(net, test_loader) 
  
print("\n")
print("Train Accuracy:" + str(accuracy_train) + "%")
print("Test Accuracy:" + str(accuracy_test) + "%")

for epoch in range(NUM_EPOCHS):
    print("\n")
    print("epoch:" +str(epoch))
    net.train()
    for data in tqdm(train_loader,unit= "samples",unit_scale=BATCH_SIZE):
        optimizer.zero_grad()
        
        features, labels = data
        features=features.to(device)
        labels=labels.to(device)
       
        outputs = net(features)
        loss = criterion(outputs,labels)
        loss.backward()

        optimizer.step()

    accuracy_train=calculate_accuracy(net, train_loader)  
    accuracy_test=calculate_accuracy(net, test_loader) 
  
    print("\n")
    print("Train Accuracy:" + str(accuracy_train) + "%")
    print("Test Accuracy:" + str(accuracy_test) + "%")
    