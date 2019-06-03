#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 12:18:48 2019

@author: jdesk
"""

import numpy as np

# row =  collector drop radius:

R_coll_drop = np.array([300,200,150,100,70,60,50,40,30,20,10])
R_coll_drop = R_coll_drop[::-1]

raw_data = np.loadtxt("Hall_1980_Collision_eff.csv")

raw_data = raw_data[::-1]

E_even_R_coll_drop = []

R_coll_drop_interpol = []

for i in range(len(R_coll_drop)-1):
    R_col = R_coll_drop[i]
    E_even_R_coll_drop.append(raw_data[i])
    R_coll_drop_interpol.append(R_col)
    R_interpol = R_col + 10
    dR2 = R_coll_drop[i+1] - R_interpol
    if dR2 > 0: dR = R_coll_drop[i+1] - R_col
    while (dR2 > 0):
        dR1 = R_interpol - R_col
        E_even_R_coll_drop.append((raw_data[i+1]*dR1 + raw_data[i]*dR2)/dR)
        R_coll_drop_interpol.append(R_interpol)
        R_interpol += 10
        dR2 = R_coll_drop[i+1] - R_interpol
        
R_coll_drop_interpol.append(300)        
E_even_R_coll_drop.append(raw_data[-1])

R_coll_drop_interpol = np.array(R_coll_drop_interpol)
E_even_R_coll_drop = np.array(E_even_R_coll_drop)

filename = "Hall_collision_efficiency.npy"
np.save(filename, E_even_R_coll_drop)
filename = "Hall_collector_radius.npy"
np.save(filename, R_coll_drop_interpol)

    # else: E_even_R_coll_drop.append(raw_data[])
    
    
#%%
    
Hall_E_col = np.load("Hall_collision_efficiency.npy")
Hall_R_col = np.load("Hall_collector_radius.npy")