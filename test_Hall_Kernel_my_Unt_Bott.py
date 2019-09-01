#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 11:47:14 2019

@author: jdesk
"""

import numpy as np

kernel_path = "collision/kernel_data/Hall/"

R_B_my = np.load(  kernel_path +  "Hall_Bott_collector_radius.npy" )
r_B_my = np.load(  kernel_path +  "Hall_Bott_radius_ratio.npy" )
E_B_my = np.load(  kernel_path +  "Hall_Bott_collision_efficiency.npy" )

R_Hall = np.load(  kernel_path +  "Hall_collector_radius.npy" )
r_Hall = np.load(  kernel_path +  "Hall_radius_ratio.npy" )
E_Hall = np.load(  kernel_path +  "Hall_collision_efficiency.npy" )


E_Unt = np.load(  kernel_path +  "Hall_Bott_E_col_Unt.npy" )


print(E_Unt.max())

#%%


