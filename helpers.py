# -*- coding: utf-8 -*-
"""
Collection of some helper functions which are used throughout the examples
"""

import torch
import math

#%%  Train model
from tqdm import tqdm

def train_model(net, device, num_epochs, optimizer, criterion, train_loader, 
                valid_loader, batch_size, steps_per_epoch=math.inf, 
                classifier = "binary", calc_acc = True):
    """
    Train neural network model on training data set

    Parameters
    ----------
    net : 
        torch neural net with initial weights.
    device : 
        torch device on which training is done.
    num_epochs : int
        number of training epochs.
    optimizer :
        training optimizer.
    criterion :
        loss function.
    train_loader : 
        data loader of training set.
    valid_loader : TYPE
        data loader of validation set.
    batch_size : int
        batch size for training of model.
    steps_per_epoch : TYPE, optional
        maximum number of steps per epoch. The default is math.inf.
    classifier : string, optional
        choose "binary" or multi "classifier". The default is "binary".
    calc_acc : bool, optional
        calculate accuracy of prediction for training and validation data. The 
        default is True.

    Returns
    -------
    net :
         torch neural net with optimized weights.
    history_dict : dict
        Dictionary with loss and accuracy of training data, validation for 
        every epoch.

    """
    history_dict = {"loss":[],"valid_loss": [], "accuracy": [], "valid_accuracy": []}

    # Initial loss and accuracy
    train_loss = calculate_loss(net, device, criterion, train_loader, 
                                steps_per_epoch, classifier = classifier)
    valid_loss = calculate_loss(net, device, criterion, valid_loader, 
                                steps_per_epoch, classifier = classifier)
    print("Train loss: {}".format(train_loss)) 
    print("Valid loss: {}".format(valid_loss)) 

    if calc_acc:
        train_acc =  calculate_accuracy(net, device, train_loader, batch_size, 
                                        steps_per_epoch, classifier = classifier)
        valid_acc =  calculate_accuracy(net, device, valid_loader, batch_size, 
                                        steps_per_epoch, classifier = classifier)
        
        print("train accuracy: " + str(train_acc))
        print("valid accuracy: " + str(valid_acc))   
        print("\n")
        history_dict["accuracy"].append(train_acc) 
        history_dict["valid_accuracy"].append(valid_acc)
                                              
                                          
    history_dict["loss"].append(train_loss)
    history_dict["valid_loss"].append(valid_loss)

    total_steps = (steps_per_epoch if steps_per_epoch < math.inf else len(train_loader))

    for epoch in range(1,num_epochs+1):
        print("Epoch: {}".format(epoch))
        net.train()
        step = 0
        for data in tqdm(train_loader,unit= "samples",unit_scale=batch_size,
                         total=total_steps):
            step += 1
            if step <= steps_per_epoch:#
                optimizer.zero_grad()
                features, labels = data
                features = features.to(device)
                labels = labels.to(device)
                outputs  = net(features)
                
                if classifier == "multi":
                    loss = criterion(outputs, labels) 
                elif  classifier == "binary":
                    try:  # 3.4
                        loss = criterion(outputs, labels.view(labels.shape[0],1))
                    except: # 5.2
                        labels = labels.type_as(outputs)
                        loss = criterion(outputs, labels.view(labels.shape[0],1))
                
                loss.backward()
                optimizer.step()
            else:
                break
        train_loss = calculate_loss(net, device, criterion, train_loader,
                                    steps_per_epoch, classifier = classifier)
        valid_loss = calculate_loss(net, device, criterion, valid_loader,
                                    classifier = classifier)
        print("\n")
        print("Train loss: {}".format(train_loss)) 
        print("Valid loss: {}".format(valid_loss)) 
        history_dict["loss"].append(train_loss)
        history_dict["valid_loss"].append(valid_loss)

        if calc_acc:    
            train_acc =  calculate_accuracy(net, device, train_loader, batch_size,
                                            steps_per_epoch, classifier = classifier)
            valid_acc =  calculate_accuracy(net, device, valid_loader, batch_size, 
                                            classifier = classifier)
              
            print("train accuracy: " + str(train_acc))
            print("valid accuracy: " + str(valid_acc))    
            print("\n")
            history_dict["accuracy"].append(train_acc) 
            history_dict["valid_accuracy"].append(valid_acc)

    return net, history_dict

#%% Accuray helpers


def calculate_accuracy(net, device, data_loader, batch_size, 
                       steps_per_epoch=math.inf, classifier = "binary"):
    """
    Calculate accuracy of neural net predictions for given data set

    Parameters
    ----------
    net : 
        torch neural net with initial weights.
    device : 
        torch device on which training is done.
    data_loader : TYPE
        torch data loader of data set..
    batch_size : int
        batch size for training of model.
    steps_per_epoch : TYPE, optional
        maximum number of steps per epoch. The default is math.inf.
    classifier : string, optional
        choose "binary" or multi "classifier". The default is "binary".

    Returns
    -------
    accuracy
        accuracy of predictions.

    """

    net.eval()
    with torch.no_grad():
        correct_pred = 0
        for step, data in enumerate(data_loader):
            if step < steps_per_epoch:#
                features, labels = data
                features = features.to(device)
                labels = labels.to(device)
                
                output = net(features)
                
                if classifier == "multi":
                    predictions = torch.argmax(output, axis=1)
                    correct_pred += torch.sum(predictions == labels)
                elif classifier == "binary":
                    correct_pred += torch.sum(torch.round(output) == 
                                              labels.view(labels.shape[0],1))
            else:
                break
        accuracy = correct_pred/(len(data_loader)*batch_size)
    return float(accuracy)

def calculate_loss(net, device, criterion, data_loader, 
                   steps_per_epoch=math.inf, classifier = "binary"):
    """
    Calculate loss of Neural net predictions for given data set

    Parameters
    ----------
    net : 
        torch neural net with initial weights.
    device : 
        torch device on which training is done.
    criterion : 
        loss function.
    data_loader :
        torch data loader of data set.
    steps_per_epoch : TYPE, optional
        maximum number of steps per epoch. The default is math.inf.
    classifier : string, optional
        choose "binary" or multi "classifier". The default is "binary".

    Returns
    -------
    loss_ep
        loss .

    """

    net.eval()
    with torch.no_grad():
        loss_ep = 0
        for step, data in enumerate(data_loader):
            if step < steps_per_epoch:#
                features, labels = data
                features = features.to(device)
                labels=labels.to(device)
                
                outputs  = net(features)
                 
                if classifier == "multi":
                    loss = criterion(outputs, labels)
                elif  classifier == "binary":
                    try:  # 3.4
                        loss = criterion(outputs, labels.view(labels.shape[0],1))
                    except: # 5.2
                        labels = labels.type_as(outputs)
                        loss = criterion(outputs, labels.view(labels.shape[0],1))
                loss_ep += loss
            else:
                break
        loss_ep /= len(data_loader)
    return float(loss_ep)

#%%

import matplotlib.pyplot as plt

def plot_results(history_dict, calc_acc = True):
    """
    Plot training and validation losses (and accuracies)

    Parameters
    ----------
    history_dict : dict
        Dictionary with loss and accuracy of training data, validation for 
        every epoch.
    calc_acc : bool, optional
        Plot accuracy of prediction for training and validation data. The 
        default is True.

    Returns
    -------
    None.

    """
    NUM_EPOCHS =  len(history_dict["loss"])
    
    plt.figure()
    plt.plot(range(0,NUM_EPOCHS), history_dict["loss"], "o")
    plt.plot(range(0,NUM_EPOCHS), history_dict["valid_loss"])
    plt.legend(["Training loss", "Valdition loss"])
    plt.title("Training and validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    if calc_acc:
        plt.figure()
        plt.plot(range(0,NUM_EPOCHS), history_dict["accuracy"], "o")
        plt.plot(range(0,NUM_EPOCHS), history_dict["valid_accuracy"])
        plt.legend(["Training accuracy", "Valdition accuracy"])
        plt.title("Training and validation accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        
    return

#%%

from collections import Counter
from torchtext.vocab import Vocab

def gen_vocab(dataset, tokenizer, max_tokens, max_len):
    """
    Generate mapping of token to index for given dataset

    Parameters
    ----------
    dataset : 
        Dataset from which  the vocabulary is build.
    tokenizer : 
        Torchtext tokenizer.
    max_tokens : int
        Maximum number of tokens of vocabulary.
    max_len : int
        Maximum length of each individual text which is considered for building
        the vocabulary.

    Returns
    -------
    vocabulary : dict
        Dictionary which maps tokens to index.

    """
    counter = Counter()

    for label, text in dataset:
        tokenized_text = tokenizer(text)[:max_len]
        counter.update(tokenized_text)
    
    vocabulary = Vocab(counter=counter,max_size=max_tokens-2,min_freq=1)

    return vocabulary

def collate_emb(batch, vocabulary, tokenizer, max_len, label_dict, device):
    """
    collate function to encode IMDB texts to list of indices for embedding layer

    Parameters
    ----------
    batch : TYPE
        batch of texts from dataset.
    vocabulary : dict
        Dictionary which maps tokens to index.
    tokenizer : 
        Torchtext tokenizer.
    max_len : int
        Maximum length of each individual text which is considered for building
        the vocabulary.
    label_dict : dict
        Dictionary mapping labels onto integers.
    device : 
        torch device on which training is done.

    Returns
    -------
    index_list: torch tensor
        tensor with embedding indices
    label_list: torch tensor
        tensor with labels
    """

    index_list = []
    label_list = []
    for label, text in batch:
        words = tokenizer(text)[:max_len]
        idcs = [vocabulary[token] for token in words]
        if len(idcs)<max_len: # pad if less than max_len words in text
             idcs+= [0]*(max_len-len(idcs))
       
        idcs_tensor = torch.tensor(idcs)
        index_list.append(idcs_tensor)
        
        lbl = int(label_dict[label])
        label_list.append(lbl)

    label_list = torch.tensor(label_list,dtype=torch.float32)
    index_list = torch.cat(index_list)
    return index_list.to(device), label_list.to(device)






