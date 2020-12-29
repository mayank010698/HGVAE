import sys
import numpy as np
import pickle, pprint
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import math
import time
import random
import os
import shutil

from lib.generate_lattice_fns import generate_lattices
from lib.observable_fns import calculate_observables
from lib.plot_graphs import plot_graphs


start_time = time.time()
#lattice size 
ls  = (8,8)
J = 1

random_state = 45
samples = 1000
steps = samples*100

rso = np.random.RandomState(seed=random_state)
addr = 'lattices'


lat_out_file = "/lattices.npz"
normalized_lattices = "/normalized_lattices.npz"
observables_file = "/observables.npz"
dat_file = addr+"lattice_dataset.npz"
betas= np.arange(1,20,1)/10

plot_observables = True
normalize_angles = False

if os.path.exists(addr):
    shutil.rmtree(addr)

os.mkdir(addr)
''' Generate lattices using MCMC '''

all_lattices=[]
for beta in betas:
    os.mkdir(addr+"/beta_"+str(beta))
    print('Generating lattices for beta = ',beta)
    lattices = generate_lattices(beta,ls,steps,rso,J)
    lat_outfile = addr + "/beta_"+str(beta) + lat_out_file
    
    if plot_observables:
        ''' Calculate Observables '''
        print('Calculating observables for beta = ',beta)
        energies,magnetizations,magnetic_suscep = calculate_observables(lattices,beta)
        observables_outfile = addr + "/beta_"+str(beta) + observables_file
        np.savez(observables_outfile, energies=energies, magnetizations=magnetizations, magnetic_suscep=magnetic_suscep)

    np.savez(lat_outfile, lattices=lattices)
    all_lattices.append(lattices)
    
np.savez('../MCMC/dataset.npz', lattices=all_lattices)
plot_graphs(addr,betas)


if normalize_angles:
    ''' Normalize angles '''
    all_lattices=[]
    for beta in betas:
        print('Normalizing angles for beta = ',beta)
        D = np.load(addr+"/beta_"+str(beta)+'/lattices.npz')['lattices']
        for i in range(lattices.shape[0]):
            lattice = lattices[i]
            mag_x=np.mean(np.cos(2*np.pi*lattice))
            mag_y=np.mean(np.sin(2*np.pi*lattice))
            u = math.atan2(mag_y,mag_x)/(2*np.pi)
            if u<0:
                u = u + 1
            A = lattice - u + 0.5
            b = A < 0
            A[b] = A[b] + 1
            c = A > 1
            A[c] = A[c] - 1
            lattices[i] = A
        outfile = addr + "/beta_"+str(beta) + normalized_lattices
        np.savez(outfile, lattices=lattices)
        np.concatenate((all_lattices,lattices),axis=0)
    
    np.savez('../MCMC/normalized_dataset.npz', lattices=all_lattices)
print("--- %s seconds ---" % (time.time() - start_time))



