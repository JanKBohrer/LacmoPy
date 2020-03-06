#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 12:31:27 2019

@author: jdesk
"""

import numpy as np

A = np.arange(10)
B = np.arange(10)
C = np.arange(5)

L = [A,B,C]

M = np.array(L)

M2 = np.array([A,B])

R = np.arange(10)

cells = np.array([[0,0,0,1,1,1,2,2,2,2],
                  [0,1,2,0,1,2,0,1,2,2]])

cells_T = cells.T

# make propagation separately for all particles
# then, e.g. every 1.0 seconds make collisions in each cell

# for given cell: array of R-indices which belong to this cell
ind_R = (1,2,3)

cell = np.array((0,0))
#
ind1 = (cells_T == cell)

#%%

#
#ind2 = ind1.nonzero

i = 2

ind_i = (cells[0] == i)

j = 2

ind_j = (cells[1] == j)

#ind_ij = (ind_i and ind_j)
ind_ij = np.logical_and(ind_i , ind_j)

#print(R[ind_ij])

#ind_ij2 = np.logical_and( *(ind_i, ind_j) )
#ind_ij2 = np.logical_and( *ind1 )

#condi = np.where(cells == (0,0))   

no_cells = [3, 3]

for i in range(no_cells[0]):
    mask_i = cells[0] == i
    for j in range(no_cells[1]):
        mask_j = cells[1] == j
        mask_ij = np.logical_and(mask_i , mask_j)
        
        print(i,j)
        print(cells[:,mask_ij])
        print(R[mask_ij])
        print()

R_ = R[mask_ij]

print(R)
print(R_)

R_[0] = 77

print(R)
print(R_)

R2 = R[ np.array([0,1]) ]

print(R)
print(R2)

R2 [0] = 77

print(R)
print(R2)

#%%

print()
print(R)

R[5] = 75

R[ np.array([2,3]) ] = np.array([72,73])

print(R)

print()

mask = [0,0,0,0,0,0,1,1,0,0]
mask = np.array(mask, dtype = bool)

print(mask)
print(R)
print(R[mask])

R[mask] = np.array([76,77])
print()
print(R)
print(R[mask])

# make R_cell = R[mask] (this is a copy)
# send R_cell to collision step -> is modified
# transfer values back:  R[mask] = R_cell






      