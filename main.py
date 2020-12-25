import yaml
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

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
            loss, recon_loss, KL= loss_function(recon_batch, data, mu, logvar)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            recon_loss += recon_loss.item()
            KL_loss += KL.item()
        
            
        print('Epoch: {} Average loss: {:.4f}  Recon Loss {:.4f}  KL Divergence {:.4f} '.format(
            epoch, total_loss , recon_loss , KL_loss))

        

    print("Model trained\n")
    torch.save(model,"cnn_vae.pt")


def main(config_file):
    with open(config_file,"r" ) as f:
        config = yaml.load(f)
    
    epochs = config["epochs"]
    lr = config["lr"]
    ls = config["lattice_size"]
    bs = config["bs"]
    is_train = config["bs"]
    data_path = config["data_path"]
    model = VAE(ls)

    if(is_train):
        train(model,lr,bs,ls,data_path,epochs)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    args = parser.parse_args()
    main(args.config_file)