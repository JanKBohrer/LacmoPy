#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 11:44:32 2019

@author: jdesk
"""
#%%
import index as i

print(i.T)

#%%
from numba import njit
import math
import numpy as np

@njit()
def update_vel(vel, xi, A1, A2):
    for ID, xi_ in enumerate(xi):
        if xi_ != 0:
            dv = A1[ID] - A2[ID]
            vel_dev = math.sqrt(dv*dv)
            print(vel_dev)
            vel[ID] = 0.5*vel_dev
            print(vel[ID])

vel = 1.0*np.arange(10)            
xi = 2*np.arange(10)            

A1 =  0.3*np.arange(10)               
A2 =  0.5*np.arange(10)               

print(xi)
print(A1)
print(A2)

print(vel)

update_vel(vel, xi[::-1], A1, A2)

print(vel)