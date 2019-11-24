
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


from vae import VAE
from LatticeDataset import LatticeDataset
from loss_function import loss_function
from plot_observables import get_energy,get_magnetisation

epochs = 200
lr=0.0001
LS=16
batch_size=100
is_train=True



device = torch.device("cuda")
model = VAE().to(device)

print(model)

print(summary(model,input_size=(1,16,16)))

LD = LatticeDataset('../japneet/MCMC/all_normalized_lattices.npz')
dataloader = DataLoader(LD, batch_size=batch_size, num_workers=4)


if(is_train):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    model.zero_grad()

    epoch=0


    LS=[]
    KLD=[]
    E=[]


    while(epoch<epochs):
        epoch+=1

        total_loss = 0
        recon_loss = 0
        KL_loss = 0
        EN_loss = 0

        for batch_idx, data in enumerate(dataloader):
            data = torch.Tensor(data.float()).to(device)
            optimizer.zero_grad()

            recon_batch, z_mean, z_logvar, mu, logvar = model(data)
            loss, LK, KL, EN= loss_function(recon_batch, data, z_mean, z_logvar, mu, logvar)

            total_loss += loss.item()/1900
            recon_loss += LK.item()/1900
            KL_loss += KL.item()/1900
            EN_loss += EN.item()/1900

            loss.backward()
            optimizer.step()
            #break
            
        LS.append(total_loss)
        KLD.append(KL_loss)
        E.append(EN_loss)

        print('Epoch: {} Average loss: {:.4f}  Recon Loss {:.4f}  KL Divergence {:.4f} Energy loss {:.4f} '.format(
        epoch, total_loss , recon_loss , KL_loss, EN_loss))



    print("Model trained\n")
    torch.save(model,"CNN_VAE")

    #LOSS_PLOTS:

    plt.plot(LS)
    plt.title("total_loss")
    plt.savefig("LOSS")
    plt.close()

    plt.plot(KLD)
    plt.title("KL-Divergence")
    plt.savefig("KL")
    plt.close()

    plt.plot(E)
    plt.title("Energy Loss")
    plt.savefig("Energy")
    plt.close()



else:
    print("Loading Saved Model\n")
    model.load_state_dict(torch.load("CNN_VAE"))
    model.eval()

    ip_mag = []
    op_mag = []
    ip_en = []
    op_en = []
    z = []

    for i in range(len(training_inputs)):
        ip = np.array(training_inputs[i][0:1000]).reshape(1000, 1, 8, 8)
        ip = torch.Tensor(ip).to(device)
        op = model(ip)
        loss.append(loss_function(op[0], ip, op[1], op[2], op[3], op[4])[1].item())
        z.append(op[3].cpu())
        ip_mag.append(np.mean(np.array([get_magnetisation(lattice) for lattice in training_inputs[i]])))
        op_mag.append(np.mean(np.array([get_magnetisation(lattice) for lattice in np.array(op[1].cpu())])))
        ip_en.append(np.mean(np.array([get_energy(lattice) for lattice in training_inputs[i]])))
        op_en.append(np.mean(np.array([get_energy(lattice) for lattice in np.array(op[0].cpu()).reshape(-1, 8, 8)])))






    plt.plot(ip_mag)
    plt.plot(op_mag)
    plt.savefig("Magnetisation")

    plt.plot(ip_en)
    plt.plot(op_en)
    plt.savefig("Energy")






