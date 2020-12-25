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

device=torch.device("cuda")
def calculate_energy(lattice):
    lattice = 2 *np.pi*lattice
    xp =torch.Tensor(lattice.size()).to(device)
    yp =torch.Tensor(lattice.size()).to(device)


    for i in range(lattice.size(2)):
        xp[: ,: ,i ,: ] =torch.cos(lattice[: ,: ,( i +1 ) %lattice.size(2) ,: ] -lattice[: ,: ,i ,:])


    for i in range(lattice.size(3)):
        yp[: ,: ,: ,i ] =torch.cos(lattice[: ,: ,: ,( i +1 ) %lattice.size(3) ] -lattice[: ,: ,: ,i])

    return ((torch.sum(torch.sum(xp ,dim=3) ,dim=2 ) +torch.sum(torch.sum(yp ,dim=3) ,dim=2) ))/(LS*LS) 



def calculate_magnetisation(lattice):

    mag_x =torch.sum(torch.sum(torch.cos( 2*np.pi *lattice) ,dim=1) ,dim=1)
    mag_y =torch.sum(torch.sum(torch.sin( 2*np.pi *lattice) ,dim=1) ,dim=1)

    return (mag_x**2 +mag_y**2 )**0.5/64


def loss_function(recon_x, x, mean, var, mu, logvar):
#    LK = (torch.sum((torch.cos(2 * np.pi * x.view(-1,LS*LS)) -torch.cos(2 * np.pi * mean.view(-1,LS*LS)))**2)/x.size(0))#,reduction='mean')+F.mse_loss(torch.sin(2*np.pi*x),torch.sin(2*np.pi*mean),reduction='mean'))
    EN = LS*LS*torch.sum((calculate_energy(recon_x)- calculate_energy(x))**2)/x.size(0)#, reduction='mean')
    MAG = F.mse_loss(calculate_magnetisation(recon_x), calculate_magnetisation(x), reduction='sum') / x.size(0)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    LK=torch.sum((x-mean)**2)/x.size(0)
    return LK+0.001*KLD+EN , LK, KLD, EN