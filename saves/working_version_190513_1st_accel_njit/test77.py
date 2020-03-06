#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 17:24:41 2019

@author: jdesk
"""
# import index
import numpy as np
#%%

A = np.arange(10)
B = np.arange(10,20)

C = np.array([A,B])

print("A")
print("C")
print(A)
print(C)

A[1] = 10

print(A)
print(C)

C[0,2] = 20

print(A)
print(C)

A2 = C[0]

print("A2")
print("C")
print(A2)
print(C)

A2[3] = 30

print(A2)
print(C)

C[0,4] = 40

print(A2)
print(C)
