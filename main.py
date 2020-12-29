import yaml
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

from matplotlib import pyplot as plt

from vae import VAE
from LatticeDataset import LatticeDataset
from loss_function import loss_function
from plot_observables import get_energy,get_magnetisation


def train(model,lr,bs,ls,data_path,epochs):
    LD = LatticeDataset(data_path,ls)
    dataloader = DataLoader(LD, batch_size=bs, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    model.zero_grad()

    for epoch in range(epochs):
        total_loss = 0
        recon_loss = 0
        KL_loss = 0
        EN_loss = 0

        for batch_idx, data in enumerate(dataloader):
            optimizer.zero_grad()
            data = torch.Tensor(data.float())
        
            recon_batch, mu, logvar = model(data)
            loss, L2_loss, KL= loss_function(recon_batch, data, mu, logvar)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            recon_loss += L2_loss.item()
            KL_loss += KL.item()
        
            
        print('Epoch: {} Average loss: {:.4f}  Recon Loss {:.4f}  KL Divergence {:.4f} '.format(
            epoch, total_loss , recon_loss , KL_loss))

        

    print("Model trained\n")
    torch.save(model,"cnn_vae.pt")


def test(data_path,ls):
    model = torch.load("cnn_vae.pt")
    model.eval()
    lattices = np.load(data_path)["lattices"]
    fig,ax = plt.subplots(1)
    for beta in np.arange(1,19):
        samples = lattices[beta]
        indices = np.random.choice(1000,20)

        recon_x, mu, logvar = model(torch.tensor(samples[indices].reshape(20,1,ls,ls)).float())
        mean = mu.detach().numpy()
        ax.scatter(mean[:,0],mean[:,1])
    
    plt.savefig("Latent Variables.png")
    
    

    
    
    

    
    



def main(config_file):
    with open(config_file,"r" ) as f:
        config = yaml.load(f)
    
    epochs = config["epochs"]
    lr = config["lr"]
    ls = config["lattice_size"]
    bs = config["bs"]
    is_train = config["is_train"]
    is_test = config["is_test"]
    data_path = config["data_path"]
    model = VAE(ls)

    if(is_train):
        train(model,lr,bs,ls,data_path,epochs)
    if(is_test):
        test(data_path,ls)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    args = parser.parse_args()
    main(args.config_file)
