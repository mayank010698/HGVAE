from __future__ import print_function, division
import os
import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

import tensorflow as tf
from torchsummary import summary
import pickle
LS=16
'''class LatticeDataset(Dataset):
    def __init__(self, file_path):
        f = open(file_path, 'rb')
        if (f.read(2) == '\x1f\x8b'):
            f.seek(0)
            gzip.GzipFile(fileobj=f)
        else:
            f.seek(0)
        training_inputs = pickle.load(f, encoding="latin1")
        self.data = np.array(training_inputs).reshape(190000, LS, LS)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].reshape(1,8,8)
'''
class LatticeDataset(Dataset):
  def __init__(self,file_path):
    data_npz=np.load(file_path)
    self.data=data_npz['lattices']
    #self.beta=np.repeat(np.arange(0.1,2,0.1),10000)
    #print(self.data.shape)
    
   
  def __len__(self):
    return self.data.shape[0]
  
  def __getitem__(self,idx):
    return self.data[idx].reshape(1,16,16)




