#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 12:46:11 2019

@author: jdesk
"""

import numpy as np

a = 1
b = [1,1]
c = [[1,1],[1,1]]
d = np.array(b)
print(np.shape(a))
print(np.shape(b))
print(np.shape(b)[0])
print(np.shape(c))
print(np.shape(d))
print(d.shape)

print( type(a) )
print( type(b) )
print( type(c) )
print( type(d) )
print( isinstance(b, int) )


A = np.reshape(np.arange(24),(3,4,2))

print(A)
cell = [1,1]
cell_t = tuple([1,1])
print(A[1,1])
print(A[cell])
print(A[cell_t])

A = np.array([0,1])
B = A[np.nonzero(A)]
print(np.nonzero(A))
print( A[np.nonzero(A)] )
print(type(B))
print(B.shape)
print(len(B))

from numba import vectorize, njit

@vectorize("float64(float64,float64)")
def func_vec(x,y):
    return x + y

print()
print(func_vec(2.0,3.0))

a = np.arange(5) * 1.0
b = np.arange(5,10) * 1.0

print()
print(a,b)
print(func_vec(a,b))

@njit("float64[:](float64[:],float64[:],float64[:])")
def func_njit(x,y,z):
    a = func_vec(x,y)
    return a + z

c = np.arange(5) * 1.0

print()
print(a,b,c)
print(func_njit(a,b,c) )

print()
# def func3(x,y):
#     return x + y

# @vectorize("int32(int32,int32)")
# def func1(x,y):
#     @vectorize
#     def func2(x):
#         return func3(x,y)
#     return func2(x)
# print()
# print(func1(2,3))

import constants as c

@vectorize("float64(float64,float64)")
def compute_radius_from_mass(mass_, density_):
    return   ( c.pi_times_4_over_3_inv * mass_ / density_ ) ** (c.one_third)

par_sol_dens = np.array([  7.619443952135e+02,   1.021264281453e+03,
                  1.828970151543e+00, 2.405352122804e+02,
                 -1.080547892416e+00,  -3.492805028749e-03 ])
w_s_rho_p = 0.001
@vectorize("float64(float64,float64)")
def compute_density_particle(mass_fraction_solute_, temperature_):
    return    par_sol_dens[0] \
            + par_sol_dens[1] * mass_fraction_solute_ \
            + par_sol_dens[2] * temperature_ \
            + par_sol_dens[3] * mass_fraction_solute_ * mass_fraction_solute_ \
            + par_sol_dens[4] * mass_fraction_solute_ * temperature_ \
            + par_sol_dens[5] * temperature_ * temperature_

@njit("UniTuple(float64[:], 3)(float64[:], float64[:], float64[:])")
def compute_R_p_w_s_rho_p(m_w, m_s, T_p):
    m_p = m_w + m_s
    w_s = m_s / m_p
    rho_p = compute_density_particle(w_s, T_p)
    return compute_radius_from_mass(m_p, rho_p), w_s, rho_p


