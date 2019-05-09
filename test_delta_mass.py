#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 16:11:42 2019

@author: jdesk
"""
import numpy as np

import constants as c
from microphysics import\
    compute_delta_water_liquid_and_mass_rate_implicit_Newton_full,\
    compute_mass_from_radius, compute_density_particle,\
    compute_radius_from_mass

from atmosphere import\
    compute_saturation_pressure_vapor_liquid,\
    compute_density_air_dry,\
    compute_pressure_vapor,\
    compute_heat_of_vaporization,\
    compute_thermal_conductivity_air,\
    compute_diffusion_constant,\
    compute_surface_tension_water
    
# import microphysics as mic
# import atmosphere as atm

#%%
import numpy as np

n_p = np.array([2,2])

idx = np.nonzero(n_p)
print(np.nonzero(n_p))
print(idx[0][0])

#%%
R_s = np.array( [0.01,0.02,0.03] )
m_s = compute_mass_from_radius(R_s, c.mass_density_NaCl_dry)
w_s = np.array( [0.1,0.05,0.02] )
m_p = m_s/w_s
m_w = m_p - m_s
T_amb = 293.0
p_amb = 101325.0
T_p = 293.0 * np.ones_like(m_p)
rho_p = compute_density_particle(w_s, T_p)
R_p = compute_radius_from_mass(m_p, rho_p)

e_s_amb = compute_saturation_pressure_vapor_liquid(T_amb)
r_v = 0.07
rho_dry = compute_density_air_dry(T_amb, p_amb)
S_amb = compute_pressure_vapor( rho_dry * r_v, T_amb ) \
        / e_s_amb
        
L_v = compute_heat_of_vaporization(T_amb)
K = compute_thermal_conductivity_air(T_amb)
D_v = compute_diffusion_constant(T_amb, p_amb)
sigma_w = compute_surface_tension_water(T_amb)

dt_sub = 0.1
no_iter = 3
dm, gamma = compute_delta_water_liquid_and_mass_rate_implicit_Newton_full(
        dt_sub, no_iter, m_w, m_s, w_s, R_p, T_p, rho_p,
        T_amb, p_amb, S_amb, e_s_amb, L_v, K, D_v, sigma_w)

print(dm)
print()
print(gamma)


R_s = 0.01
# R_s = np.array( [0.01,0.02,0.03] )
m_s = compute_mass_from_radius(R_s, c.mass_density_NaCl_dry)
w_s = 0.1
m_p = m_s/w_s
m_w = m_p - m_s
T_amb = 293.0
p_amb = 101325.0
T_p = 293.0
# T_p = 293.0 * np.ones_like(m_p)
rho_p = compute_density_particle(w_s, T_p)
R_p = compute_radius_from_mass(m_p, rho_p)

e_s_amb = compute_saturation_pressure_vapor_liquid(T_amb)
r_v = 0.07
rho_dry = compute_density_air_dry(T_amb, p_amb)
S_amb = compute_pressure_vapor( rho_dry * r_v, T_amb ) \
        / e_s_amb
        
L_v = compute_heat_of_vaporization(T_amb)
K = compute_thermal_conductivity_air(T_amb)
D_v = compute_diffusion_constant(T_amb, p_amb)
sigma_w = compute_surface_tension_water(T_amb)

dt_sub = 0.1
no_iter = 3
dm, gamma = compute_delta_water_liquid_and_mass_rate_implicit_Newton_full(
        dt_sub, no_iter, m_w, m_s, w_s, R_p, T_p, rho_p,
        T_amb, p_amb, S_amb, e_s_amb, L_v, K, D_v, sigma_w)

print(dm)
print()
print(gamma)

#%%
# from atmosphere import epsilon_gc

# print(2.0 * epsilon_gc)
import numpy as np
cell_list = np.array([ [0,1,2], [3,4,5] ])
print(cell_list[:,0] )
print(cell_list[:,1] )

cell = cell_list[:,0]

A = np.reshape(np.arange(24), (4,6) )

cell2 = np.array( [1,1] )

print(A)
print(cell)
print(A[0,3])
print(A[cell])
print(A[cell2])

#%%

from numba import njit, jit
import numpy as np
from timeit import timeit


def func1(x,targ):
    
    res = []
    
    for x_ in x:
        if x_ in targ:
            res.append(x_)
            # print(x_)
    return res

@njit
def func2(x):
    rng = [2,4]
    # rng = range(2,4)
    for x_ in x:
        if x_ in rng:
            print(x_)
    pass
            


func1_njit = njit()(func1)
func1_jit = jit()(func1)

x = np.array((1,2,3,4,5))        
targ = (2,3,4)

print()
print(func2(x))
print()

# print(func1(x,targ))
tt = %timeit -r 5 -n 1000 -o func1(x,targ)
tt2 = %timeit -r 5 -n 1000 -o func1_njit(x,targ)
tt3 = %timeit -r 5 -n 1000 -o func1_jit(x,targ)

print(tt.best*1E6)
print(tt2.best*1E6)
print(tt3.best*1E6)

#%%



def func1(x,y):
    arr = np.empty((2,len(x)), dtype=np.float64)
    for i,x_ in enumerate(x):
        arr[0,i] = x_
        arr[1,i] = y[i]
    # arr = np.array( (x,y) )
    return arr

func1_njit = njit()(func1)

x = np.arange(10)
y = np.arange(10)




#%%

# map arrays:
# T[i,j]  i = 0, .. Nx-1, j = 0, .. , Nz - 1-> T[ID] ID = 0, ... no_spt

T = np.reshape( np.linspace(1.0, 12.0, 12), (3,4) )

i = np.array([0,1,2])
j = np.array([0,1,2,3])

i,j = np.meshgrid(i,j, indexing = "xy")

i = i.flatten()
j = j.flatten()

cells = np.array((i,j))

# print(T[i,j])


#%%

# R_s = np.array( [0.01,0.02,0.03] )
# m_s = mic.compute_mass_from_radius(R_s, c.mass_density_NaCl_dry)
# w_s = np.array( [0.1,0.05,0.02] )
# m_p = m_s/w_s
# m_w = m_p - m_s
# T_amb = 293.0
# p_amb = 101325.0
# T_p = 293.0 * np.ones_like(m_p)
# rho_p = mic.compute_density_particle(w_s, T_p)
# R_p = mic.compute_radius_from_mass(m_p, rho_p)

# e_s_amb = atm.compute_saturation_pressure_vapor_liquid(T_amb)
# r_v = 0.07
# rho_dry = atm.compute_density_air_dry(T_amb, p_amb)
# S_amb = atm.compute_pressure_vapor( rho_dry * r_v, T_amb ) \
#         / e_s_amb
        
# L_v = atm.compute_heat_of_vaporization(T_amb)
# K = atm.compute_thermal_conductivity_air(T_amb)
# D_v = atm.compute_diffusion_constant(T_amb, p_amb)
# sigma_w = atm.compute_surface_tension_water(T_amb)

# dt_sub = 0.1
# no_iter = 3
# dm, gamma = compute_delta_water_liquid_and_mass_rate_implicit_Newton_full(
#         dt_sub, no_iter, m_w, m_s, w_s, R_p, T_p, rho_p,
#         T_amb, p_amb, S_amb, e_s_amb, L_v, K, D_v, sigma_w)

# print(dm)
# print()
# print(gamma)

