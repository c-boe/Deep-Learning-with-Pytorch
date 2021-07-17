# -*- coding: utf-8 -*-
"""
Implementation of datasets from "Deep learning with Python" by Chollet as 
torch datasets: 
-) Boston housing dataset (chapter 3.6 of Chollet's "Deep learning with Python")
-) Jena climate dataset (chapter 6.3 of Chollet's "Deep learning with Python")
"""

#%%Boston housing dataset
import numpy as np
import os
import pandas as pd
import requests
import zipfile

import torch
from torch.utils.data import Dataset


class BH(Dataset):

    url_trainset = "https://people.sc.fsu.edu/~jburkardt/datasets/boston_housing/boston_housing_train.csv"
    url_testset = "https://people.sc.fsu.edu/~jburkardt/datasets/boston_housing/boston_housing_test.csv"
    test_filename = "boston_housing_test.csv"
    train_filename = "boston_housing_train.csv"
    
    len_features = 13
    
    def __init__(self, root: str, train: bool = True, normalize: bool = False,
                 transform=None, download: bool = False):
        """

        Parameters
        ----------
        root : str
            root directory of where train and test data are saved.
        train : bool, optional
            Create data from train or test set (csv file without targets!). 
            The default is True.
        normalize : bool, optional
            normalize features by mean and standard deviation. 
            The default is False.
        transform : callable, optional
            Apply transform to dataset. The default is None.
        download : bool, optional
            download train and test data to root. The default is False.

        Returns
        -------
        None.

        """
        self.root = root
        self.train = train
        self.normalize = normalize
        self.transform = transform
        self.file_test =  os.path.join(self.root, self.test_filename)
        self.file_train =  os.path.join(self.root, self.train_filename)
        
        if download:
            self._download()

        if self.train:
            self.dataset = pd.read_csv(self.file_train)
        elif not self.train:
            self.dataset = pd.read_csv(self.file_test) # targets missing!
            
        if self.normalize:
            self._normalize()
            
    def __len__(self):
        return len(self.dataset)
    
    
    def __getitem__(self, index):
        """
        

        Parameters
        ----------
        index : int
            index

        Returns
        -------
        sample:
            features for prediction of house prices.
        target : 
            median price.

        """
        sample = self.dataset.iloc[index,1:self.len_features+1]
        target = self.dataset.iloc[index,self.len_features+1]
        sample = np.array([sample],dtype=np.float32)
        target = np.float32(target)
        
        if self.transform is not None:
            sample, target = self.transform(sample, target)

        return sample[0], target            
    
    def _download(self):
        """Dowload data"""
        file_test_exists = os.path.exists(self.file_test)
        file_train_exists = os.path.exists(self.file_train)
        
        if  os.path.exists(self.root) is not True:
            os.makedirs(self.root)
        
        if file_test_exists is not True:
            r= requests.get(self.url_testset)
            with open(self.file_test,"wb") as f:
                f.write(r.content)
                
        if file_train_exists is not True: 
            r= requests.get(self.url_trainset)
            with open(self.file_train,"wb") as f:
                f.write(r.content)                
        return
    
    def _normalize(self):
        """Normalizing of features in pandas dataframe by mean and standard 
        deviation"""
        self.dataset_std = self.dataset.iloc[:,:self.len_features+1].std()
        self.dataset_std[self.dataset_std==0]=1
        self.dataset_mean = self.dataset.iloc[:,:self.len_features+1].mean()
        self.dataset.iloc[:,:self.len_features+1] = (self.dataset.iloc[:,:self.len_features+1] - 
                                                      self.dataset_mean)/self.dataset_std
        return  
    
    def partition(self, num_parts: int, curr_parts: list, normalize=False):
        """
        Select partitions for k-fold cross validation

        Parameters
        ----------
        num_parts : int
            number of partitions for K-Fold validation.
        curr_parts : list
            list of partitions from which subset of dataset is created.
        normalize : TYPE, optional
            normalize subset. The default is False.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        num_samples = int(len(self.dataset)/num_parts)
        idcs = list(range(len(self.dataset)))
        
        fold_idcs = []
        for fold in curr_parts:
            fold_idcs.extend(idcs[fold*num_samples:(fold+1)*num_samples])
        
        self.dataset =  self.dataset.iloc[fold_idcs,:]
        if normalize:
            self._normalize()
        self.__len__()
        
        return self



#%% Jena climate dataset

"""
Climate dataset from Max-Planck Institute for biogeochemistry in Jena,
Germany: https://www.bgc-jena.mpg.de/index.php/Main/HomePage
dataset: https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip
"""

class JenaClimate(Dataset):
    
    url = "https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip"
    filename = "jena_climate_2009_2016.csv"
    zipname = "jena_climate_2009_2016.csv.zip"
    
    def __init__(self, root: str, download = False, transform=None, 
                 normalize=False):
        """
        

        Parameters
        ----------
        root : str
            folder of dataset "jena_climate_2009_2016.csv".
        download : str, optional
            download dataset from url. The default is False.
        transform : callable, optional
            Function that transforms dataset, e.g. to tensors. The default is 
            None.
        normalize : bool, optional
            normalize columns of input dataset by mean and standard deviation. 
            The default is True.
        lookback : int, optional
            Number of past observations. The default is 720.
        delay : int, optional
            delayed timestep in the future at which target temperature is taken. 
            The default is 144.
        step : int, optional
            Defines intervall between timesteps at which observation are sampled. 
            E.g. step = 1 corresponds to 10 min
            The default is 6.
        min_index : int, optional
            starting index of dataset from which samples are taken. The default 
            is 0.
        max_index : int, optional
            maximum index of dataset from which samples are taken. The default 
            is None.

        Returns
        -------
        None.

        """
        self.root = root
        self.file =  os.path.join(self.root,self.filename)
        
        if self.download:
            self.download()
        
        self.climate_data = pd.read_csv(self.file)
        self.transform = transform

        if normalize:
            self._normalize()

        self.dataset =  self.climate_data.iloc[:, 1:]
   
        self.delay = 144
        self.step = 6
        self.lookback =720

        self.min_index = 0
        self.max_index = len(self.dataset) - self.delay - 1
      
    def __len__(self):
        """Define possible indices which __getitem__ can utilize"""
        len_dataset = self.max_index - self.min_index  - self.lookback - self.delay
        return len_dataset
    
    def __getitem__(self, index: int):

        row = index + self.lookback# self.min_index + 
        past_indices = range(row - self.lookback, row, self.step)
        #print(row)
        sample = self.dataset.iloc[past_indices, 1:]
        
        target = self.dataset.iloc[row + self.delay, 1]
        sample = np.array(sample,dtype=np.float32)#np.array([sample],dtype=np.float32)
        target = np.float32(target)
        
        if self.transform is not None:
            sample, target = self.transform(sample, target)
            
        return sample, target

    def _normalize(self):
        """normalizing columns of pandas dataframe by mean and standard 
        deviation"""
        self.dataset_std = self.dataset.std()
        self.dataset_mean = self.dataset.mean()

        self.dataset = (self.dataset - self.dataset_mean)/self.dataset_std
        return
    
    def download(self):
        """Dowload and unzip climate"""
        folder = os.path.join(self.root, self.zipname)
        file =  os.path.join(self.root, self.filename)
        folder_exists = os.path.exists(folder)
        file_exists = os.path.exists(file)
        
        if  os.path.exists(self.root) is not True:
            os.makedirs(self.root)
        
        if folder_exists is not True:
            r= requests.get(self.url)
            with open(folder,"wb") as f:
                f.write(r.content)
                
        if file_exists is not True:
            with zipfile.ZipFile(folder,"r") as zip_file:
                zip_file.extractall(path=self.root)

        return
    
    def subset(self, normalize=True, 
                 lookback=720, delay=144, step=6, min_index=0, max_index=None):
        """Split dataset"""
        self.normalize = normalize
    
        self.lookback = lookback                          
        self.delay = delay
        self.step = step
        self.min_index = min_index
        self.max_index = max_index

        self.dataset =  self.climate_data.iloc[self.min_index:self.max_index, 1:]
       
        if self.normalize:
            self._normalize()
        
        self.__len__()
        
        return self
    
    
#%% Transforms  
        
class ToTensor():

    def __init__(self): 
        pass
    
    def __call__(self, sample, target):
        sample = torch.from_numpy(sample)
        target = target
        return sample, target