#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 17:54:00 2019

@author: jdesk
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt

import constants as c
from collision import compute_terminal_velocity_Beard

filename =\
"Re__Box_model_simulations_of_Lagrangian_cloud_microphysics/VBeard.dat"

data_Beard = np.loadtxt(filename).transpose()
R = data_Beard[0]
# R2 = np.arange(0.1,6330,0.1)
R2 = R
v_Be = data_Beard[2]

v_sed = []

for R_ in R2:
    v_sed.append(compute_terminal_velocity_Beard(R_)*100.0)

v_sed = np.array(v_sed)

fig,ax = plt.subplots(figsize=(6,6) )

ax.plot(data_Beard[0], data_Beard[2], "o-")
ax.plot(R2, v_sed, "o-")

# ax.set_xlim([534.9,535.1])
# ax.set_xlim([520,560])
# ax.set_ylim([410,440])

# ax.plot(R, (v_sed-v_Be) / v_sed)

ax.grid()



#%%

filename =\
"Re__Box_model_simulations_of_Lagrangian_cloud_microphysics/Rad_Longkernel.out"
data = np.loadtxt(filename).flatten()

R_Bott = np.append(data, 6330.53121399468)

filename =\
"Re__Box_model_simulations_of_Lagrangian_cloud_microphysics/E_Longkernel.out"
data = np.loadtxt(filename).flatten()
data = np.append(data, 1.00)
# E_Bott

