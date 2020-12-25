import os
import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, ls, channels = 1, z_dim=2):
        super(VAE, self).__init__()
        self.h_dim = 16*ls*ls
        self.ls = ls
        '''encoder '''
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.fc_encoder = nn.Sequential(nn.Linear(self.h_dim, 32), nn.ReLU(), nn.Linear(32, 10), nn.ReLU())
        self.zmean = nn.Linear(10, z_dim)
        self.zlogvar = nn.Linear(10, z_dim)

        ''' decoder '''
        self.conv_decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
	        nn.ConvTranspose2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.fc_decoder = nn.Sequential(nn.Linear(z_dim, 10), nn.ReLU(), nn.Linear(10, 32), nn.ReLU(), nn.Linear(32,self.h_dim),
                                  nn.ReLU())
        self.decoder = nn.Sequential(nn.ConvTranspose2d(16, channels, kernel_size=3, stride=1, padding=1))


    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(mu.size())

        z = mu + std * esp.float()
        return z

    def forward(self, x):
        x = self.conv_encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc_encoder(x)

        mu = self.zmean(x)
        logvar = self.zlogvar(x)
        z = self.reparameterize(mu, logvar)

        z = self.fc_decoder(z)
        z = z.view(z.size(0), 16, self.ls, self.ls)
        recon = self.decoder(z)
        return recon, mu, logvar
