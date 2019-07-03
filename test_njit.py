#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 16:20:51 2019

@author: bohrer
"""

from numba import njit

import numpy as np

A = np.arange(1,13)
B = np.array([0,0,1,1,0,0,1,1,0])
print(A)
print(B)

@njit()
def func(A,B):
    ind = np.nonzero(B)[0]
    A = A[ind]
    return A
C = func(A,B)

print(C)

print(np.nonzero(B)[0])


a = 20

if a == 10 or a == 20:
    print("yes")
    
XX = np.ones((200,200))