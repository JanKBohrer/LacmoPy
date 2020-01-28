#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
binomial:
    
P(k,n) = (n over k) * p**k *(1 - p)**(n-k)

P(k >= 1, n) = 1 - P(k=0, n)

= 1 - (1-p)**n

"""

import numpy as np
import math


def P_bin(n,p):
    return 1 - (1-p)**n

def P_bin_inv(P,p):
    return np.log(1-P) / np.log(1-p)

p = 0.02
n = np.array((0,1,2,3,4,5,10,20,50,100,200))

PP = P_bin(n,p)

print(n)
print(PP)

for i,pp in enumerate(PP):
    print(n[i],pp)

Q = np.array((1,2,5,10,20,50,68,90,95,98,99,99.5,99.9))

Qp = Q*0.01

NN = P_bin_inv(Qp,p)

print(Q)
print(NN)

for i,q in enumerate(Q):
    print(q,NN[i])
