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
from collision import compute_terminal_velocity_Beard2
from collision import compute_E_col_Long_Bott
from collision import kernel_Long_Bott
from collision import kernel_Long_Bott_R

#%% TERMINAL VELOCITY COMPARE WITH BEARD

filename =\
"Re__Box_model_simulations_of_Lagrangian_cloud_microphysics/VBeard.dat"

data = np.loadtxt(filename).transpose()
R = data[0]
R2 = R
# R2 = np.arange(0.1,6330,0.1)
v_Be = data[2]

v_sed = []
v_sed2 = []

for R_ in R2:
    v_sed.append(compute_terminal_velocity_Beard(R_)*100.0)
    v_sed2.append(compute_terminal_velocity_Beard2(R_)*100.0)

v_sed = np.array(v_sed)
v_sed2 = np.array(v_sed2)

fig,ax = plt.subplots(figsize=(6,6) )

ax.plot(R, v_Be, "o-")
# ax.plot(R2, v_sed, "o-")
ax.plot(R2, v_sed2, "o-")

# ax.set_xlim([534.9,535.1])
# ax.set_xlim([520,560])
# ax.set_ylim([410,440])

# ax.plot(R, (v_sed-v_Be) / v_sed)

ax.grid()

# i_max = 20
# for i in range(i_max):
#     print(v_sed[i]/v_Be[i])

# print((v_sed2-v_Be)/v_Be)
# 118, 117

i_max = 400
for i in range(i_max):
    strng = " "
    for j in range(i):
    # for j in range(i_max):
        dv1 = abs(v_Be[i]-v_Be[j])
        dv2 = abs(v_sed2[i]-v_sed2[j])
        rel_dev = abs(dv2-dv1)/max(dv1,1.0E-20)
        if rel_dev > 0.005: print(i,j,rel_dev)
        # strng += f" {abs(dv2-dv1)/max(dv1,1.0E-20):.2e}"
    # print(strng)

#%% COMPARE THE EFFICIENCY OF THE LONG KERNEL WITH BOTT RESULTS FROM UNTERSTR.

filename =\
"Re__Box_model_simulations_of_Lagrangian_cloud_microphysics/Rad_Longkernel.out"
data = np.loadtxt(filename).flatten()

R_Bott = np.append(data, 6330.53121399468)

filename =\
"Re__Box_model_simulations_of_Lagrangian_cloud_microphysics/E_Longkernel.out"
data = np.loadtxt(filename).flatten()
data = np.append(data, 1.00)
# E_Bott

E_Bott = np.reshape(data,(400,400))

E_Bott_my = np.zeros((400,400))

for j,R_j in enumerate(R_Bott):
    for i in range(j+1):
        E_Bott_my[j,i] = compute_E_col_Long_Bott(R_Bott[i],R_j)
        E_Bott_my[i,j] = E_Bott_my[j,i]

for j,R_j in enumerate(R_Bott):
    for i,R_i in enumerate(R_Bott):
        if abs(E_Bott_my[i,j]-E_Bott[i,j])/E_Bott[i,j] >= 1.0E-13:
            print(i,j, abs(E_Bott_my[i,j]-E_Bott[i,j])/E_Bott[i,j])

i_max = 220        
        
fig,ax = plt.subplots(figsize=(8,8) )
for i in range(0,i_max,20):
    ax.plot(R_Bott[:i_max], E_Bott_my[i,:i_max], label=f"Ri={R_Bott[i]}")
ax.set_yscale("log")        
ax.legend()

#%% COMPARE THE LONG KERNEL VALUES WITH BOTT RESULTS FROM UNTERSTR.

filename =\
"Re__Box_model_simulations_of_Lagrangian_cloud_microphysics/Rad_Longkernel.out"
data = np.loadtxt(filename).flatten()

R_Bott = np.append(data, 6330.53121399468)
from microphysics import compute_mass_from_radius
masses = compute_mass_from_radius(R_Bott, c.mass_density_water_liquid_NTP)

filename =\
"Re__Box_model_simulations_of_Lagrangian_cloud_microphysics/K_Longkernel.out"
data = np.loadtxt(filename).flatten()
data = np.append(data, 0.00)
# E_Bott

K_Bott = np.reshape(data,(400,400))

K_Bott_my = np.zeros((400,400))
K_Bott_my2 = np.zeros((400,400))

for j,R_j in enumerate(R_Bott):
    for i in range(j+1):
        # K_Bott_my[j,i] = kernel_Long_Bott(masses[j],masses[i])
        K_Bott_my[j,i] = kernel_Long_Bott_R(R_Bott[j],R_Bott[i])
        K_Bott_my[i,j] = K_Bott_my[j,i]

K_Bott_my *= 1.0E6

for i,R_j in enumerate(R_Bott):
    for j,R_i in enumerate(R_Bott):
        jm=max(j-1,1)
        im=max(i-1,1)
        jp=min(j+1,399)
        ip=min(i+1,399)
        if i != j:
            K_Bott_my2[i,j] =\
                0.125*(K_Bott_my[i,jm]+K_Bott_my[im,j]+
                       K_Bott_my[ip,j]+K_Bott_my[i,jp])\
                + 0.5*K_Bott_my[i,j]

i_max = 400
for i in range(i_max):
    strng = " "
    for j in range(i_max):
    # for j in range(i_max):
        rel_dev = abs(K_Bott[i,j]-K_Bott_my[i,j])/max(K_Bott[i,j],1.0E-20)
        # dv2 = abs(v_sed2[i]-v_sed2[j])
        # rel_dev = abs(dv2-dv1)/max(dv1,1.0E-20)
        if rel_dev > 0.005: print(i,j,rel_dev)


# i_min = 60
# i_max = 120
# for i in range(i_min,i_max):
#     print( (K_Bott_my[i,i_min:i_max]-K_Bott[i,i_min:i_max])\
#            / np.fmax( K_Bott[i,i_min:i_max], 1.0E-18) )
    

# for j,R_j in enumerate(R_Bott):
#     for i,R_i in enumerate(R_Bott):
#         if abs(K_Bott_my2[i,j]-K_Bott[i,j])/max(K_Bott[i,j],1E-18) >= 0.1:
#             print(i,j, abs(K_Bott_my2[i,j]-K_Bott[i,j])/max(K_Bott[i,j],1E-18))

# for j,R_j in enumerate(R_Bott):
#     for i,R_i in enumerate(R_Bott):
#         if abs(K_Bott_my[i,j]-K_Bott[i,j])/K_Bott[i,j] >= 1.0E-12:
#             print(i,j, abs(K_Bott_my[i,j]-K_Bott[i,j])/K_Bott[i,j])

#%%

# i_max = 220        
        
# fig,ax = plt.subplots(figsize=(8,8) )
# for i in range(0,i_max,20):
#     ax.plot(R_Bott[:i_max], E_Bott_my[i,:i_max], label=f"Ri={R_Bott[i]}")
# ax.set_yscale("log")        
# ax.legend()

