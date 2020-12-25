import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

class LatticeDataset(Dataset):
  def __init__(self,file_path,ls):
    data=np.load(file_path)["lattices"]
    self.ls = ls
    self.classes = data.shape[0]
    self.spc = data.shape[1]
    self.size = self.classes*self.spc

    self.data=data.reshape(self.size,-1)

  def __len__(self):
    return self.data.shape[0]
  
  def __getitem__(self,idx):
    return self.data[idx].reshape(1,self.ls,self.ls)




