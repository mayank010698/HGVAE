import torch
import torch.nn as nn
from torch.nn import functional as F

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


def loss_function(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x,x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + KLD,recon_loss,KLD
