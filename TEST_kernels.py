#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 13:32:44 2020

@author: bohrer
"""

import numpy as np

data_dir = "/Users/bohrer/sim_data_box_mod_test/expo/Ecol_grid_data"
data_dir2 = "/Users/bohrer/sim_data_box_mod_test/expo/kernel_grid_data"

kernel_name = "Long_Bott"
kernel_name2 = "Hall_Bott"

Ecol = np.load(f"{data_dir}/{kernel_name}/E_col_grid.npy")
radius = np.load(f"{data_dir}/{kernel_name}/radius_grid_out.npy")

kernel_grid_L = np.load(f"{data_dir2}/{kernel_name}/kernel_grid.npy")
kernel_grid_H = np.load(f"{data_dir2}/{kernel_name2}/kernel_grid.npy")

mass_L = np.load(f"{data_dir2}/{kernel_name}/mass_grid_out.npy")
mass_H = np.load(f"{data_dir2}/{kernel_name2}/mass_grid_out.npy")

radius_k_L = np.load(f"{data_dir2}/{kernel_name}/radius_grid_out.npy")
radius_k_H = np.load(f"{data_dir2}/{kernel_name2}/radius_grid_out.npy")

vel_L = np.load(f"{data_dir2}/{kernel_name}/velocity_grid.npy")
vel_H = np.load(f"{data_dir2}/{kernel_name2}/velocity_grid.npy")


print(kernel_grid_H-kernel_grid_L)
print(mass_H-mass_L)
print(radius_k_H-radius_k_L)
print(vel_H-vel_L)

#Hall_Bott_E_col_raw_orig =\
#    np.loadtxt("/Users/bohrer/CloudMP/SAVES/Efficiency_Hallkernel_Bott_orig_test.out")
#    
#Hall_Bott_E_col_raw_orig2 = np.reshape(Hall_Bott_E_col_raw_orig.flatten()[:-2],(400,400))    
#    
#print(np.where(Hall_Bott_E_col_raw_orig2 > 1.0))
#
#Hall_Bott_E_col_raw_corr =\
#    np.load("collision/kernel_data/Hall/Hall_Bott_E_col_table_raw_corr.npy")
#Hall_Bott_R_col_raw =\
#    np.load("collision/kernel_data/Hall/Hall_Bott_R_col_table_raw.npy")
#Hall_Bott_R_ratio_raw =\
#    np.load("collision/kernel_data/Hall/Hall_Bott_R_ratio_table_raw.npy")
