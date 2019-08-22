#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 13:59:23 2019

@author: jdesk
"""

import numpy as np
import matplotlib.pyplot as plt

from collision.kernel import compute_E_col_Hall_Bott, generate_E_col_grid_R
from collision.kernel import generate_E_col_grid_R_from_R_grid

Hall_Bott_E_col = np.load("collision/kernel_data/Hall/Hall_Bott_collision_efficiency.npy")
Hall_Bott_E_col_raw_mod = np.loadtxt("collision/kernel_data/Hall/Hall_1980_Collision_eff_Bott2_my_corr.txt",
                                 delimiter=",")

Hall_Bott_E_col_raw = np.loadtxt("collision/kernel_data/Hall/Hall_1980_Collision_eff_Bott_raw_Unt.txt",
                                 delimiter=",").flatten()[:-5]

Hall_Bott_R_col = np.load("collision/kernel_data/Hall/Hall_Bott_collector_radius.npy")
Hall_Bott_R_col_ratio = np.load("collision/kernel_data/Hall/Hall_Bott_radius_ratio.npy")  

Hall_Bott_R_col_Unt = np.loadtxt("collision/kernel_data/Hall/Radius_Hallkernel_Unt_orig.out").flatten()[:-2]
Hall_Bott_E_col_Unt = np.loadtxt("collision/kernel_data/Hall/Efficiency_Hallkernel_Unt_orig.out").flatten()[:-2]


Hall_Bott_E_col_raw_mod = np.reshape(Hall_Bott_E_col_raw_mod, (21,15))
Hall_Bott_E_col_raw = np.reshape(Hall_Bott_E_col_raw, (21,15))
Hall_Bott_E_col_Unt = np.reshape(Hall_Bott_E_col_Unt, (400,400))

np.save("collision/kernel_data/Hall/Hall_Bott_E_col_Unt.npy", Hall_Bott_E_col_Unt)
np.save("collision/kernel_data/Hall/Hall_Bott_R_col_Unt.npy", Hall_Bott_R_col_Unt)

#%%
#idx_ = 592
#
#R1 = 547.206503613
#R2 = 566.43652577
#
#print(compute_E_col_Hall_Bott(R1, 0.6))  
#print(compute_E_col_Hall_Bott(0.6, 0.6))  

print("Hall_Bott_E_col_raw-Hall_Bott_E_col_raw_mod")
print(Hall_Bott_E_col_raw-Hall_Bott_E_col_raw_mod)


#%%
kernel_name = "Hall_Bott"
R_low_kernel, R_high_kernel, no_bins_10_kernel = 0.6, 6E3, 200

E_col_grid1, radius_grid1 = \
    generate_E_col_grid_R(R_low_kernel, R_high_kernel, no_bins_10_kernel,
                          kernel_name)    
E_col_grid2, radius_grid2 = \
    generate_E_col_grid_R_from_R_grid (Hall_Bott_R_col_Unt,
                          kernel_name)    

E_col_grid3 = np.minimum(E_col_grid2, np.ones_like(E_col_grid2))

print((E_col_grid3 - Hall_Bott_E_col_Unt)/Hall_Bott_E_col_Unt)

#E_col_grid3[np.abs((E_col_grid3 - Hall_Bott_E_col_Unt)/Hall_Bott_E_col_Unt) > 0.5]
print(np.where(np.abs((E_col_grid3 - Hall_Bott_E_col_Unt)/Hall_Bott_E_col_Unt) > 0.5)[0].shape)

E_col_rel_dev = \
    np.maximum(np.abs((E_col_grid3 - Hall_Bott_E_col_Unt)/Hall_Bott_E_col_Unt),
           np.ones_like(E_col_grid3)*1E-5)

#%%


fig, ax = plt.subplots()    
ax.plot(Hall_Bott_R_col_Unt)  
ax.set_yscale("log")
  