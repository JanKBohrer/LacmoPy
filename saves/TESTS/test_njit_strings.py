#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:57:54 2020

@author: bohrer
"""

from numba import njit
import numpy as np

@njit()
def compute_rho1(A,b):
    return A*b**2
@njit()
def compute_rho2(A,b):
    return A*b**3

@njit()
def decide(A, solute_type):
    if solute_type == "AS":
        return compute_rho1(A,2.0)
    elif solute_type == "NaCl":
        return compute_rho2(A,2.0)
    
A = 10.
solute_type="AS"
print(A, decide(A, solute_type))

solute_type="NaCl"
print(A, decide(A, solute_type))