#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 15:17:00 2019

@author: jdesk
"""

import numpy as np

#load_path = "/mnt/D/sim_data_cloudMP_col/grid_75_75_spcm_20_20/no_spin_up_col/"
#
#xi_1800 = np.load(load_path + "particle_xi_data_1800.npy")
#xi_all_1800 = np.load(load_path + "particle_xi_data_all_1800.npy")


#%%
#load_path = "/mnt/D/sim_data_cloudMP_col/grid_10_10_spcm_12_12/3711/"
#load_path = "/mnt/D/sim_data_cloudMP_col/grid_10_10_spcm_0_4/3711/"




#%% CHECK INITIAL DISTRIBUTIONS

### STORAGE DIRECTORIES
my_OS = "Linux_desk"
#my_OS = "Mac"

if(my_OS == "Linux_desk"):
    home_path = '/home/jdesk/'
    simdata_path = "/mnt/D/sim_data_cloudMP_col/"
#    simdata_path = "/mnt/D/sim_data_cloudMP/test_gen_grid_and_pt/"
#    sim_data_path = home_path + "OneDrive/python/sim_data/"
#    fig_path = home_path + 'Onedrive/Uni/Masterthesis/latex/Report/Figures/'
elif (my_OS == "Mac"):
#    home_path = "/Users/bohrer/sim_data_cloudMP/test_gen_grid_and_pt/"
    simdata_path = "/Users/bohrer/sim_data_cloudMP/test_gen_grid_and_pt/"
#    simdata_path = home_path + "OneDrive - bwedu/python/sim_data/"
#    fig_path = home_path \
#               + 'OneDrive - bwedu/Uni/Masterthesis/latex/Report/Figures/'

no_spcm = np.array([16, 24])
#no_spcm = np.array([4, 4])

no_cells = (75, 75)
#no_cells = (3, 3)

# seed of the SIP generation -> needed for the right grid folder
seed_SIP_gen = 3713

grid_folder =\
    f"grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm[0]}_{no_spcm[1]}/" \
    + f"{seed_SIP_gen}/"

load_path = simdata_path + grid_folder

xi = np.load(load_path + "multiplicity_0.npy")
scalars = np.load(load_path + "particle_scalars_0.npy")
m_w = scalars[0]
m_s = scalars[1]

cells = np.load(load_path + "particle_cells_0.npy")
modes = np.load(load_path + "modes_0.npy")

with open(load_path + "grid_paras.txt") as gp_file:
    lines = gp_file.readlines()
    grid_para_names = lines[0].split()
    grid_paras = lines[1].split()


dx = float(grid_paras[4])
dy = float(grid_paras[5])
dz = float(grid_paras[6])
dV = dx*dy*dz

DNC0 = np.array(( grid_paras[11],  grid_paras[12]) ).astype(float)
no_spcm = np.array(( grid_paras[13],  grid_paras[14]) ).astype(int)
mu_m_log = np.array(( grid_paras[16],  grid_paras[17]) ).astype(float)
sigma_m_log = np.array(( grid_paras[18],  grid_paras[19]) ).astype(float)
kappa = np.array(( grid_paras[20],  grid_paras[21]) ).astype(float)
eta = float(grid_paras[22])

print(DNC0, no_spcm, mu_m_log, sigma_m_log, kappa, eta)

no_rpcm_scale_factors_lvl_wise =\
    np.load(load_path + "no_rpcm_scale_factors_lvl_wise.npy" )

def moments_analytical_lognormal_m(n, DNC, mu_m_log, sigma_m_log):
    if n == 0:
        return DNC
    else:
        return DNC * np.exp(n * mu_m_log + 0.5 * n*n * sigma_m_log*sigma_m_log)

#%%
no_modes = 2
no_moments = 4
#for mode_n in range(no_modes):
moments_an = []
for j in range(no_cells[1]):
    moments_an_lvl = []
    for mom_n in range(no_moments):
        moments_an_lvl.append(
            moments_analytical_lognormal_m(
                mom_n, DNC0 * no_rpcm_scale_factors_lvl_wise[j],
                mu_m_log, sigma_m_log))
    moments_an.append(moments_an_lvl)
        
moments_an = np.array(moments_an)    
    

def compute_moments_num(xi, m_s, cells, no_modes, no_moments):
    moments_num = np.zeros( (no_cells[0], no_cells[1], no_moments, no_modes) )    
    for i in range(no_cells[0]):
        mask_i = cells[0] == i
        for j in range(no_cells[1]):
            mask_ij = np.logical_and(mask_i , (cells[1] == j))
            
            for mode_n in range(no_modes):
                mask_mode = modes == mode_n
                mask_ij = np.logical_and(mask_mode, mask_ij)
                moments_num[i,j,0,mode_n] = np.sum( xi[mask_mode] )
                for mom_n in range(1,no_moments):
                    moments_num[i,j,mom_n,mode_n] =\
                        np.sum( xi[mask_mode] * m_s[mask_mode]**mom_n ) / dV
    return moments_num                    
    
moments_num = compute_moments_num(xi, m_s, cells, no_modes, no_moments)  

save_path = simdata_path + grid_folder + "init_eval/"

#%%
import os
if not os.path.exists(save_path):
    os.makedirs(save_path)

np.save(save_path + "moments_an", moments_an)
np.save(save_path + "moments_num", moments_num)


#%% PLOTTING

#import matplotlib.pyplot as plt
#fig, ax = plt.subplots(nrows=no_modes)
#
#for n in range(4):
#    ax.plot(n*np.ones_like(moments_sampled[n]),
#            moments_sampled[n]/moments_an[n], "o")
#ax.errorbar(np.arange(4), moments_sampled_avg_norm, moments_sampled_std_norm,
#            fmt = "x" , c = "k", markersize = 20.0, linewidth =5.0,
#            capsize=10, elinewidth=5, markeredgewidth=2,
#            zorder=99)
#ax.plot(np.arange(4), np.ones_like(np.arange(4)))
#ax.xaxis.set_ticks([0,1,2,3])
#ax.set_xlabel("$k$")
## ax.set_ylabel(r"($k$-th moment of $f_m$)/(analytic value)")
#ax.set_ylabel(r"$\lambda_k / \lambda_{k,analytic}$")

    
