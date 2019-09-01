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


Hall_Bott_E_col_raw_mod = np.reshape(Hall_Bott_E_col_raw_mod, (21,15)).T
Hall_Bott_E_col_raw = np.reshape(Hall_Bott_E_col_raw, (21,15)).T
Hall_Bott_E_col_Unt = np.reshape(Hall_Bott_E_col_Unt, (400,400))

#np.save("collision/kernel_data/Hall/Hall_Bott_E_col_Unt.npy", Hall_Bott_E_col_Unt)
#np.save("collision/kernel_data/Hall/Hall_Bott_R_col_Unt.npy", Hall_Bott_R_col_Unt)

#%%

#Hall_Bott_R_col_raw = np.array([6.,8.,10.,15.,20.,25.,30.,40.,
#                                50.,60.,70.,100.,150.,200.,300.])
#
##Hall_Bott_R_ratio_raw2 = np.array([ 0.,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,
##                                  0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,
##                                  0.9,0.95,1.0])
#Hall_Bott_R_ratio_raw = np.arange(0.,1.01, 0.05)
#
#np.save("collision/kernel_data/Hall/Hall_Bott_E_col_table_raw_corr.npy",
#        Hall_Bott_E_col_raw_mod)
#np.save("collision/kernel_data/Hall/Hall_Bott_E_col_table_raw.npy",
#        Hall_Bott_E_col_raw)
#np.save("collision/kernel_data/Hall/Hall_Bott_R_col_table_raw.npy",
#        Hall_Bott_R_col_raw)
#np.save("collision/kernel_data/Hall/Hall_Bott_R_ratio_table_raw.npy",
#        Hall_Bott_R_ratio_raw)

#%%

import math
from numba import njit
# i = ind 1, p = weight for ind 1
# j = ind 2, 1 = weight for ind 2
@njit()
def interpol_bilin(i,j,p,q,f):
    return (1.-p)*(1.-q)*f[i,j] + (1.-p)*q*f[i,j+1]\
           + p*(1.-q)*f[i+1,j] + p*q*f[i+1,j+1]

Hall_Bott_E_col_raw_corr = np.load("collision/kernel_data/Hall/Hall_Bott_E_col_table_raw_corr.npy")
#Hall_Bott_E_col_raw_corr = np.load("collision/kernel_data/Hall/Hall_Bott_E_col_table_raw.npy")
Hall_Bott_R_col_raw = np.load("collision/kernel_data/Hall/Hall_Bott_R_col_table_raw.npy")
Hall_Bott_R_ratio_raw = np.load("collision/kernel_data/Hall/Hall_Bott_R_ratio_table_raw.npy") 
# R_low must be > 0 !
def generate_E_col_grid_R_Hall_Bott_corr_np(R_low, R_high, no_bins_10,
                                            radius_grid_in=None):
    
    no_R_table = Hall_Bott_R_col_raw.shape[0]
    no_rat_table = Hall_Bott_R_ratio_raw.shape[0]
    
    if radius_grid_in is None:
        bin_factor = 10**(1.0/no_bins_10)
        no_bins = int(math.ceil( no_bins_10 * math.log10(R_high/R_low) ) ) + 1
        radius_grid = np.zeros( no_bins, dtype = np.float64 )
        
        radius_grid[0] = R_low
        for bin_n in range(1,no_bins):
            radius_grid[bin_n] = radius_grid[bin_n-1] * bin_factor   
    else:
        radius_grid = radius_grid_in
        no_bins = radius_grid_in.shape[0]
    
    kernel_grid = np.zeros( (no_bins, no_bins), dtype = np.float64 )
    
    # R = larger radius
    # r = smaller radius
    for ind_R_out, R_col in enumerate(radius_grid):
        # get index of coll. radius lower boundary (floor)
        if R_col <= Hall_Bott_R_col_raw[0]:
            ind_R_table = -1
        elif R_col > Hall_Bott_R_col_raw[-1]:
            ind_R_table = no_R_table - 1 # = 14
        else:
            for ind_R_table_ in range(0,no_R_table-1):
                # we want drops larger 300 mu to have Ecol=1
                # thus drops with R = 300 mu (exact) go to another category
                if (Hall_Bott_R_col_raw[ind_R_table_] < R_col) \
                   and (R_col <= Hall_Bott_R_col_raw[ind_R_table_+1]):
                    ind_R_table = ind_R_table_
                    break
        for ind_r_out in range(ind_R_out+1):
            ratio = radius_grid[ind_r_out] / R_col
            # get index of radius ratio lower boundary (floor)
            # index of ratio should be at least one smaller
            # than the max. possible index. The case ratio = 1 is also covered
            # by the bilinear interpolation
            if ratio >= 1.0: ind_ratio = no_rat_table-2
            else:
                for ind_ratio_ in range(no_rat_table-1):
                    if (Hall_Bott_R_ratio_raw[ind_ratio_] <= ratio) \
                       and (ratio < Hall_Bott_R_ratio_raw[ind_ratio_+1]):
                        ind_ratio = ind_ratio_
                        break
            # bilinear interpolation:
            # linear interpol in ratio, if R >= R_max
            if ind_R_table == no_R_table - 1:
                q = ( ratio - Hall_Bott_R_ratio_raw[ind_ratio] ) \
                    / ( Hall_Bott_R_ratio_raw[ind_ratio+1] 
                        - Hall_Bott_R_ratio_raw[ind_ratio] )
                E_col = (1.-q) * Hall_Bott_E_col_raw_corr[ind_R_table,ind_ratio]\
                        + q * Hall_Bott_E_col_raw_corr[ind_R_table,ind_ratio+1]
                if E_col <= 1.0:
                    kernel_grid[ind_R_out, ind_r_out] = E_col
                else:    
                    kernel_grid[ind_R_out, ind_r_out] = 1.0
#                kernel_grid[ind_R_out, ind_r_out] = min(E_col, 1.0)
            elif ind_R_table == -1:
                q = ( ratio - Hall_Bott_R_ratio_raw[ind_ratio] ) \
                    / ( Hall_Bott_R_ratio_raw[ind_ratio+1] 
                        - Hall_Bott_R_ratio_raw[ind_ratio] )
                kernel_grid[ind_R_out, ind_r_out] =\
                        (1.-q) * Hall_Bott_E_col_raw_corr[0,ind_ratio]\
                        + q * Hall_Bott_E_col_raw_corr[0,ind_ratio+1] 
            else:
                p = ( R_col - Hall_Bott_R_col_raw[ind_R_table] ) \
                    / ( Hall_Bott_R_col_raw[ind_R_table+1] 
                        - Hall_Bott_R_col_raw[ind_R_table] )                
                q = ( ratio - Hall_Bott_R_ratio_raw[ind_ratio] ) \
                    / ( Hall_Bott_R_ratio_raw[ind_ratio+1] 
                        - Hall_Bott_R_ratio_raw[ind_ratio] )                
                kernel_grid[ind_R_out, ind_r_out] = \
                    interpol_bilin(ind_R_table, ind_ratio, p, q,
                                   Hall_Bott_E_col_raw_corr)
            kernel_grid[ind_r_out, ind_R_out] =\
                kernel_grid[ind_R_out, ind_r_out]
    return kernel_grid, radius_grid
generate_E_col_grid_R_Hall_Bott_corr = \
    njit()(generate_E_col_grid_R_Hall_Bott_corr_np)

#R_low = 1E-2
#R_high = 301.
#no_bins_10 = 100
#
##R_low = Hall_Bott_R_col_Unt[0]
##R_high = Hall_Bott_R_col_Unt[-1]
##no_bins_10 = 99
#
#my_E_col_HB, my_R_col = \
#    generate_E_col_grid_R_Hall_Bott_corr(R_low, R_high, no_bins_10)
#%%
#path = "/mnt/D/sim_data_cloudMP/Ecol_grid_data/Long_Bott/"

kernel_type = "Long_Bott"
#path = "collision/Ecol_grid_data/" + kernel_name + "/"

save_dir_Ecol_grid =  f"Ecol_grid_data/{kernel_type}/"

R_Long_c = np.load(save_dir_Ecol_grid + "radius_grid_out.npy")
E_Long_c = np.load(save_dir_Ecol_grid + "E_col_grid.npy")

kernel_type = "Hall_Bott"
#kernel_name = "Long_Bott"

#path = "collision/Ecol_grid_data/" + kernel_name + "/"
save_dir_Ecol_grid =  f"Ecol_grid_data/{kernel_type}/"

R_Hall_c = np.load(save_dir_Ecol_grid + "radius_grid_out.npy")
E_Hall_c = np.load(save_dir_Ecol_grid + "E_col_grid.npy")



#%%
#my_E_col_HB, my_R_col = \
#    generate_E_col_grid_R_Hall_Bott_corr(R_low, R_high, no_bins_10,
#                                            Hall_Bott_R_col_Unt)
#dev =  np.abs((my_E_col_HB-Hall_Bott_E_col_Unt))
#rel_dev = np.abs((my_E_col_HB-Hall_Bott_E_col_Unt)/Hall_Bott_E_col_Unt)    
#lim = 1E-7
#mask = rel_dev > lim
##print( )
#inds = np.where(rel_dev > lim)
#
#print((my_E_col_HB[mask]).shape)
#
#print(rel_dev.max())

#%%
#idx_ = 592
#
#R1 = 547.206503613
#R2 = 566.43652577
#
#print(compute_E_col_Hall_Bott(R1, 0.6))  
#print(compute_E_col_Hall_Bott(0.6, 0.6))  

#print("Hall_Bott_E_col_raw-Hall_Bott_E_col_raw_mod")
#print(Hall_Bott_E_col_raw-Hall_Bott_E_col_raw_mod)
#
#
##%%
#kernel_name = "Hall_Bott"
#R_low_kernel, R_high_kernel, no_bins_10_kernel = 0.6, 6E3, 200
#
#E_col_grid1, radius_grid1 = \
#    generate_E_col_grid_R(R_low_kernel, R_high_kernel, no_bins_10_kernel,
#                          kernel_name)    
#E_col_grid2, radius_grid2 = \
#    generate_E_col_grid_R_from_R_grid (Hall_Bott_R_col_Unt,
#                          kernel_name)    
#
#E_col_grid3 = np.minimum(E_col_grid2, np.ones_like(E_col_grid2))
#
#print((E_col_grid3 - Hall_Bott_E_col_Unt)/Hall_Bott_E_col_Unt)
#
##E_col_grid3[np.abs((E_col_grid3 - Hall_Bott_E_col_Unt)/Hall_Bott_E_col_Unt) > 0.5]
#print(np.where(np.abs((E_col_grid3 - Hall_Bott_E_col_Unt)/Hall_Bott_E_col_Unt) > 0.5)[0].shape)
#
#E_col_rel_dev = \
#    np.maximum(np.abs((E_col_grid3 - Hall_Bott_E_col_Unt)/Hall_Bott_E_col_Unt),
#           np.ones_like(E_col_grid3)*1E-5)

#%%


#fig, ax = plt.subplots()    
#ax.plot(Hall_Bott_R_col_Unt)  
#ax.set_yscale("log")
  