#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 23:13:09 2019

@author: jdesk
"""

import numpy as np

from numba import njit

id_list = np.arange(5)

active_ids = np.array([1,0,0,1,1]).astype(bool)


for i in id_list[active_ids]:
    print(i)
#for i in range(5):
#    print(i)

np.save("active_ids_test.npy", active_ids)

act2 = np.load("active_ids_test.npy")

print(act2)

XX = np.array([0.0])

@njit()
def raise_XX(XX,a):
    XX[0] += a

print()    
print(XX)

raise_XX(XX,1.1)

print(XX)