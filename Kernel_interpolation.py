#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 12:18:48 2019

@author: jdesk
"""

import numpy as np

# row =  collector drop radius:

#%% CREATE TABLE "Hall_collision_efficiency.npy":
# two dimensional array: rows = collector radius, cols = ratio R_small/R_col

#R_col_ratio = np.arange(0.0,1.05,0.05)
#R_coll_drop = np.array([300,200,150,100,70,60,50,40,30,20,10])
#R_coll_drop = R_coll_drop[::-1]
#
#raw_data = np.loadtxt("Hall_1980_Collision_eff.csv")
#
#raw_data = raw_data[::-1]
#
#E_even_R_coll_drop = []
#
#R_coll_drop_interpol = []
#
#for i in range(len(R_coll_drop)-1):
#    R_col = R_coll_drop[i]
#    E_even_R_coll_drop.append(raw_data[i])
#    R_coll_drop_interpol.append(R_col)
#    R_interpol = R_col + 10
#    dR2 = R_coll_drop[i+1] - R_interpol
#    if dR2 > 0: dR = R_coll_drop[i+1] - R_col
#    while (dR2 > 0):
#        dR1 = R_interpol - R_col
#        E_even_R_coll_drop.append((raw_data[i+1]*dR1 + raw_data[i]*dR2)/dR)
#        R_coll_drop_interpol.append(R_interpol)
#        R_interpol += 10
#        dR2 = R_coll_drop[i+1] - R_interpol
#        
#R_coll_drop_interpol.append(300)        
#E_even_R_coll_drop.append(raw_data[-1])
#
#R_coll_drop_interpol = np.array(R_coll_drop_interpol)
#E_even_R_coll_drop = np.array(E_even_R_coll_drop)
#
#E_even_R_coll_drop = np.hstack((np.zeros_like(R_coll_drop_interpol, dtype=np.float64)[:,None], E_even_R_coll_drop))
#
#E_even_R_coll_drop = np.concatenate((np.zeros_like(R_col_ratio, dtype=np.float64)[None,:], E_even_R_coll_drop))
#
#R_coll_drop_interpol = np.insert(R_coll_drop_interpol, 0, 0).astype(float)
#
##R_col_ratio = np.arange(0.05,1.05,0.05)
#
#filename = "Hall_collision_efficiency.npy"
#np.save(filename, E_even_R_coll_drop)
#filename = "Hall_collector_radius.npy"
#np.save(filename, R_coll_drop_interpol)
#filename = "Hall_radius_ratio.npy"
#np.save(filename, R_col_ratio)
    
    
#%%

#R_col_ratio = np.arange(0.05,1.05,0.05)
    




#%%
import math

Hall_E_col = np.load("Hall_collision_efficiency.npy")
Hall_R_col = np.load("Hall_collector_radius.npy")
Hall_R_col_ratio = np.load("Hall_radius_ratio.npy")

from grid import bilinear_weight
def linear_weight(i, weight, f):
    return f[i+1]*weight + f[i]*(1.0-weight)
def compute_E_col_Hall(R_i, R_j):
    if R_i <= 0.0 or R_j <= 0.0:
        return 0.0
    if R_i < R_j:
        R_col = R_j
        R_ratio = R_i/R_j
    else:
        R_col = R_i
        R_ratio = R_j/R_i
    if R_col > 300.0:
        return 1.0
    else:
        # NOTE that ind_col is for index of R_collection,
        # which indicates the row of Hall_E_col, and NOT the coloumn
        ind_col = int(R_col/10.0)
        ind_ratio = int(R_ratio/0.05)
        if ind_col == Hall_R_col.shape[0]-1:
            if ind_ratio == Hall_R_col_ratio.shape[0]-1:
                return 1.0
            else:
                weight = (R_ratio - ind_ratio * 0.05) / 0.05
                return linear_weight(ind_ratio, weight, Hall_E_col[ind_col])
        elif ind_ratio == Hall_R_col_ratio.shape[0]-1:
            weight = (R_col - ind_col * 10.0) / 10.0
            return linear_weight(ind_col, weight, Hall_E_col[:,ind_ratio])
        else:
            weight_1 = (R_col - ind_col * 10.0) / 10.0
            weight_2 = (R_ratio - ind_ratio * 0.05) / 0.05
#        print(R_col, R_ratio)
#        print(ind_col, ind_ratio, weight_1, weight_2)
            return bilinear_weight(ind_col, ind_ratio,
                                   weight_1, weight_2, Hall_E_col)
        # E_col = bilinear_weight(ind_1, ind_2, weight_1, weight_2, Hall_E_col)
        
#i1 = 29
#j1 = 20
#R1 = Hall_R_col[i1]
#R2 = Hall_R_col_ratio[j1] * R1
#print()
#print("R1,R2, Hall_R_col_ratio[j1]")
#print(R1,R2, Hall_R_col_ratio[j1])
#print("Hall_E_col[i1,j1]")
#print(Hall_E_col[i1,j1])
#print(compute_E_col_Hall(R1,R2))
    
for i1 in range(0,21):
    for j1 in range(21):
        R1 = Hall_R_col[i1]
        R2 = Hall_R_col_ratio[j1] * R1        
        print(i1,j1,Hall_E_col[i1,j1]-compute_E_col_Hall(R1,R2))

#        dE = Hall_E_col[i1,j1]-compute_E_col_Hall(R1,R2)
#        if dE > 0.0 or math.isnan(dE):
#            print(i1,j1, R1, R2, Hall_R_col_ratio[j1], dE)

#        print(i1,j1, R1, R2, Hall_R_col_ratio[j1], Hall_E_col[i1,j1]-compute_E_col_Hall(R1,R2))
    