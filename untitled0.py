#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 16:17:24 2019

@author: bohrer
"""

def load_kernel_data(kernel_method, save_dir_Ecol_grid, E_col_const):
    if kernel_method == "Ecol_grid_R":
        radius_grid = \
            np.load(save_dir_Ecol_grid + "radius_grid_out.npy")
        E_col_grid = \
            np.load(save_dir_Ecol_grid + "E_col_grid.npy" )        
        R_kernel_low = radius_grid[0]
        bin_factor_R = radius_grid[1] / radius_grid[0]
        R_kernel_low_log = math.log(R_kernel_low)
        bin_factor_R_log = math.log(bin_factor_R)
        no_kernel_bins = len(radius_grid)
    elif kernel_method == "Ecol_const":
        E_col_grid = E_col_const
        radius_grid = None
        R_kernel_low = None
        R_kernel_low_log = None
        bin_factor_R_log = None
        no_kernel_bins = None    
    return E_col_grid, radius_grid, \
           R_kernel_low, bin_factor_R, \
           R_kernel_low_log, bin_factor_R_log, \
           no_kernel_bins