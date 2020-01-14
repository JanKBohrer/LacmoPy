#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 15:26:51 2019

@author: jdesk
"""

import numpy as np

no_cells = [4,3]

L = []

cell_i = []
cell_j = []

no_sp_placed = np.zeros(no_cells)


cnt = 0
for j in range(3):
    L_ = []
    for i in range(4):
        L_.append( np.arange(j+1) )
#        cell_i.append(i)
#        cell_j.append(j)
        no_sp_placed[i,j] = len(L_[i])
    L.append(L_)
    
#%%    
no_spcm = np.array([0, 20])

idx_mode = np.nonzero(no_spcm)

print(idx_mode)
print(no_spcm[idx_mode])
print(no_spcm[idx_mode].shape)

b = 2.3
a = np.log(b)
