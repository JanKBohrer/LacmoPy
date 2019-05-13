#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 17:24:41 2019

@author: jdesk
"""
# import index
import numpy as np
from numba import njit
#%%

A = np.arange(10)
B = np.arange(10,20)
C = np.arange(20,30)

G = np.array((A,B,C))
G[0] = A
G[1] = B
G[2] = C
print("A B C G")
print(A)
print(B)
print(C)
print(G)

G[0] = np.arange(50,60)
print("A B C G")
print(A)
print(B)
print(C)
print(G)
@njit
def change_element(G, E):
    # G[0] = np.arange(60,70)
    G[0] = E

# E = np.arange(70,80)
E = np.linspace(70.0,80.0,10)
change_element(G,E)
print("change_element(G,E)")
print("G")
print(G)
# G[0,0] = -1
# print("E")
# print("G")
# print(E)
# print(G)

#%%

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
