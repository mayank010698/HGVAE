from __future__ import print_function, division
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

class LatticeDataset(Dataset):
  def __init__(self,file_path,ls):
    data_npz=np.load(file_path)
    self.ls = ls
    self.data=data_npz['lattices']

  def __len__(self):
    return self.data.shape[0]
  
  def __getitem__(self,idx):
    return self.data[idx].reshape(1,self.ls,self.ls)




