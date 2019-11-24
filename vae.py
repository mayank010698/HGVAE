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
device = torch.device("cuda")

class VAE(nn.Module):
    def __init__(self, image_channels=1, h_dim=16 * LS * LS , z_dim=2):
        super(VAE, self).__init__()



        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.fc_e = nn.Sequential(nn.Linear(h_dim, 32), nn.ReLU(), nn.Linear(32, 10), nn.ReLU())
        self.mean_n = nn.Linear(10, z_dim)
        self.logvar_n = nn.Linear(10, z_dim)



        self.fc_d = nn.Sequential(nn.Linear(z_dim, 10), nn.ReLU(), nn.Linear(10, 32), nn.ReLU(), nn.Linear(32,h_dim),
                                  nn.ReLU())

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
	    nn.ConvTranspose2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.mean_decoder = nn.Sequential(nn.ConvTranspose2d(16, image_channels, kernel_size=3, stride=1, padding=1))
        self.logvar_decoder = nn.Sequential(nn.ConvTranspose2d(16, image_channels, kernel_size=3, stride=1, padding=1))

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(mu.size()).to(device)

        z = mu + std * esp.float()
        return z

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc_e(x)

        mu = self.mean_n(x)
        logvar = self.logvar_n(x)
        z = self.reparameterize(mu, logvar)
        #z=mu

        z = self.fc_d(z)
        z = z.view(z.size(0), 16, LS, LS)

        z = self.decoder(z)
        z_mean = self.mean_decoder(z)
        z_logvar = self.logvar_decoder(z)

        #z = self.reparameterize(z_mean, z_logvar)
        z_std = z_logvar.mul(0.5).exp_()/(2*3.14)
        esp = torch.randn(z_mean.size()).to(device)

        z = z_mean + z_logvar * esp.float()

        return z, z_mean, z_logvar, mu, logvar
