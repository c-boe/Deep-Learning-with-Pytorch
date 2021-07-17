# -*- coding: utf-8 -*-
"""
Implementation convolutional neural network classifier with VGG16 convolutional
base for binary classification of "Cats vs. Dogs" data set (chapter 5.3 of 
Chollet's "Deep learning with Python") using Pytorch

Download dataset:
www.kaggle.com/c/dogs-vs-cats/data
"""

import os
import shutil
import torch
import torchvision

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%% Instantiating the VGG16 convolutional base
print("Load VGG16")

model = torchvision.models.vgg16(pretrained=True,progress=False)
vgg16_conv_base = model.features
#%% import datasets
print("Prepare data")
NUM_TRAIN_IMGS = 2000
NUM_VALID_IMGS = 1000
NUM_TEST_IMGS = 1000
NUM_IMGS = NUM_TRAIN_IMGS + NUM_VALID_IMGS + NUM_TEST_IMGS

path = "./data/CatsVsDogs"
path_labels = os.path.join(path, "labels")

if os.path.exists(path_labels) is not True:
    os.mkdir(path_labels)
    
path_train = os.path.join(path, "train_original") 

def gen_label_folder(label, path_labels, path_train):
    """copy image to from folder to new folder"""
    new_path = os.path.join(path_labels, label)  
    if os.path.exists(new_path) is not True:
        os.mkdir(new_path)
    
    list_img = os.listdir(path=path_train)
    img_list = [img for img in list_img if label in img]
    for img in img_list:
        shutil.copy(src= os.path.join(path_train, img),
                    dst= os.path.join(new_path, img))

    return

labels = ["dog", "cat"]
for label in labels:
    gen_label_folder(label, path_labels, path_train)
    
#%% import datasets
print("Prepare data")
import torchvision
from torchvision import transforms
from torch.utils.data.dataset import random_split

NUM_TRAIN_IMGS = 2000
NUM_VALID_IMGS = 1000
NUM_TEST_IMGS = 1000
NUM_IMGS = NUM_TRAIN_IMGS + NUM_VALID_IMGS + NUM_TEST_IMGS

BATCH_SIZE = 16

transform = torchvision.transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize(size=(150,150)), 
                                            transforms.Normalize(mean=(0,0,0),
                                                                 std=(1,1,1))])
data_set = torchvision.datasets.ImageFolder(root=path_labels, transform=transform,)

sub_set_1, sub_set_2 = random_split(data_set,[NUM_IMGS, len(data_set) - NUM_IMGS])
train_set, sub_set_3 = random_split(sub_set_1, [NUM_TRAIN_IMGS, 
                                                 len(sub_set_1) - NUM_TRAIN_IMGS])
valid_set, test_set = random_split(sub_set_3, [NUM_VALID_IMGS, NUM_TEST_IMGS])

train_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size=BATCH_SIZE,
                                         shuffle=True,)
valid_loader = torch.utils.data.DataLoader(dataset=valid_set,batch_size=BATCH_SIZE,
                                         shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set,batch_size=BATCH_SIZE,
                                        shuffle=True,)

#%% Neural net with VGG16 conv base
print("Define neural net with VGG16 conv base")
from torch import nn
import torch.nn.functional as F

class BinaryImageClassifier(nn.Module):
    
    def __init__(self, conv_base):
        super(BinaryImageClassifier, self).__init__()
        self.conv_base = conv_base
        self.flatten = nn.Flatten(start_dim=1,end_dim=-1)
        self.fc1 = nn.Linear(in_features = 4 * 4 *512, out_features = 256)
        self.DO = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features =  256, out_features = 1)
        
    def forward(self,x):
        x = self.conv_base(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.DO(x)
        x = torch.sigmoid(self.fc2(x))
        return x

#%% Train model
print("Train model")
from helpers import train_model
from torch import optim as optim

LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
STEPS_PER_EPOCH = 100

net = BinaryImageClassifier(vgg16_conv_base).to(device)
net.conv_base.requires_grad_ = False

optimizer = optim.RMSprop(net.parameters(),lr=LEARNING_RATE)
criterion = nn.BCELoss()

net, history_dict = train_model(net, device, NUM_EPOCHS, optimizer, criterion, 
                                train_loader, valid_loader, BATCH_SIZE,
                                steps_per_epoch=STEPS_PER_EPOCH, 
                                classifier="binary", calc_acc = True)

#%% Plot results
print("Plot results")
from helpers import plot_results

plot_results(history_dict)
    
        