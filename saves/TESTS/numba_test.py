#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 13:50:00 2019

@author: bohrer
"""

#import numba
from numba import njit
import numpy as np

#@njit()

def fib(n):
    a = np.int64(0)
    b = np.int64(1)
    for i in range(n):
        next_ = a + b
        a = b
        b = next_
    return a

fib_jit = njit("int64(int64)")(fib)
#fib_jit = njit("float64(float64)")(fib)

#print(fib(20))
#print(fib_jit(20))

#N = 100

#print(9,223,372,036,854,775,807)
print(f"{9223372036854775807:.3e}")

N1 = 79
N2 = 82

for N in range(N1, N2):
    res1 = fib(N)
    res2 = fib_jit(N)

#    print(N, f"{res1:.3e}")
#    print(N, f"{res2:.3e}")
    
    print(N,res1,f"{res1:.3e}")
    print(N,res2,f"{res2:.3e}")
