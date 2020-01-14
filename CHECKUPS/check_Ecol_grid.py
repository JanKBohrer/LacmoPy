#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 11:20:41 2019

@author: bohrer
"""

import numpy as np

path = "/Users/bohrer/sim_data_cloudMP/Ecol_grid_data/Long_Bott/"
path = "/Users/bohrer/sim_data_cloudMP/Ecol_grid_data/Long_Bott/"
path = "/Users/bohrer/CloudMP/collision/kernel_data/Hall/"

Ecol_grid = np.load(path + "Hall_Bott_E_col_Unt.npy")
R_grid = np.load(path + "Hall_Bott_R_col_Unt.npy")

Ecol_grid2 = np.load(path + "new/Hall_Bott_E_col_Unt.npy")
R_grid2 = np.load(path + "new/Hall_Bott_R_col_Unt.npy")


