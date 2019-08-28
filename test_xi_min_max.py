#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 12:27:34 2019

@author: bohrer
"""

import numpy as np

my_OS = "Mac"

if(my_OS == "Linux_desk"):
    home_path = '/home/jdesk/'
    simdata_path = "/mnt/D/sim_data_cloudMP/"
#    sim_data_path = home_path + "OneDrive/python/sim_data/"
#    fig_path = home_path + 'Onedrive/Uni/Masterthesis/latex/Report/Figures/'
elif (my_OS == "Mac"):
    simdata_path = "/Users/bohrer/sim_data_cloudMP/"
elif (my_OS == "TROPOS_server"):
    simdata_path = "/vols/fs1/work/bohrer/sim_data_cloudMP/"


#no_cells = np.array((10,10))
no_cells = np.array((75,75))

no_spcm = np.array((26,38))

solute_type = "AS"
#seed_SIP_gen = 3711
#seed_SIP_gen = 9711
seed_SIP_gen_list = [9711,9713,9715,9717]

t = 0

for seed_SIP_gen in seed_SIP_gen_list:

    grid_folder =\
        f"{solute_type}/" \
        + f"grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm[0]}_{no_spcm[1]}/" \
        + f"{seed_SIP_gen}/"
    
    load_path = simdata_path + grid_folder
    
    xi = np.load(load_path + f"multiplicity_{int(t)}.npy")
    cells = np.load(load_path + f"particle_cells_{int(t)}.npy")
    
    
    print (len(xi))
    print (xi.min())

#print("hello")

#%%

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6,6))

ax.semilogy(xi, ".")

#%% 
no_cells_tot = 75*75

xi_min = []
for i in range(no_cells[0]):
    mask_i = cells[0] == i
    for j in range(no_cells[1]):
        mask = mask_i & (cells[1] == j)
        xi_min.append(xi[mask].min())
#    print(np.min(xi[i*64:(i+1)*64]))
    
fig, ax = plt.subplots(figsize=(6,6))
ax.plot(xi_min, ".")    

print(np.min(xi_min))
