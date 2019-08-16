#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 13:39:18 2019

@author: jdesk
"""

import math
import numpy as np
from numba import njit, vectorize
import matplotlib.pyplot as plt

import microphysics as mp
import constants as c
import atmosphere as atm   

#%%

T = 298.    
T_amb = T
T_p = T

S_amb = 1.005
p_amb = 8E4
e_s_amb = atm.compute_saturation_pressure_vapor_liquid(T_amb)
L_v = atm.compute_heat_of_vaporization(T_amb)
K = atm.compute_thermal_conductivity_air(T_amb)
D_v = atm.compute_diffusion_constant(T_amb, p_amb)

D_s_list = np.arange(5,101,10) * 1E-3
R_s_list = D_s_list * 0.5

m_s_AS = mp.compute_mass_from_radius_jit(R_s_list, c.mass_density_AS_dry)
m_s_SC = mp.compute_mass_from_radius_jit(R_s_list, c.mass_density_NaCl_dry)

w_s_init_AS = \
mp.compute_initial_mass_fraction_solute_m_s_AS(m_s_AS, S_amb, T_amb)

w_s_init_SC = \
mp.compute_initial_mass_fraction_solute_m_s_NaCl(m_s_SC, S_amb, T_amb)

m_p_init_AS = m_s_AS / w_s_init_AS
m_p_init_SC = m_s_SC / w_s_init_SC

S_eq_AS_init = mp.compute_equilibrium_saturation_AS_mf(w_s_init_AS, T_p, m_s_AS)
S_eq_SC_init = mp.compute_equilibrium_saturation_NaCl_mf(w_s_init_SC, T_p, m_s_SC)

#%%
w_s = np.logspace(-3.5, np.log10(0.78), 1000)

no_rows = 2
fig, axes = plt.subplots(no_rows, figsize=(10,8*no_rows))

for n,m_s_AS0 in enumerate(m_s_AS):

#m_s_AS0 = m_s_AS[0]
    rho_AS = mp.compute_density_AS_solution(w_s, T_p) 
    m_p_AS = m_s_AS0/w_s
    m_w_AS = m_p_AS - m_s_AS0
    R_p_AS = mp.compute_radius_from_mass_jit(m_p_AS, rho_AS)
    
    sigma_AS = mp.compute_surface_tension_AS(w_s, T_p) 
    
    S_eq_AS = mp.compute_equilibrium_saturation_AS_mf(w_s, T_p, m_s_AS0)
    S_eq_AS2 = mp.compute_equilibrium_saturation_AS(
                   w_s, R_p_AS, T_p, rho_AS, sigma_AS)
    
    
    ax = axes[0]
    ax.plot(m_p_AS, S_eq_AS, label = f"{m_s_AS0:.1e}")
    #ax.plot(R_p_AS, S_eq_AS2, label = "AS2")
    #ax.plot(R_p_SC, S_eq_SC, label = "SC")
    ax.scatter(m_p_init_AS[n], S_eq_AS_init[n], marker="x", s=100)
    ax.axhline(S_amb)
    ax.set_xscale("log")
    ax.legend(loc = "right")
    
axes[0].grid()
axes[0].set_ylim([0.99,1.05])

ax = axes[1]
for n,m_s_SC0 in enumerate(m_s_SC):

#m_s_AS0 = m_s_AS[0]
    rho_SC = mp.compute_density_NaCl_solution(w_s, T_p) 
    m_p_SC = m_s_SC0/w_s
    m_w_SC = m_p_SC - m_s_SC0
    R_p_SC = mp.compute_radius_from_mass_jit(m_p_SC, rho_SC)
    
    sigma_SC = mp.compute_surface_tension_NaCl(w_s, T_p) 
    
    S_eq_SC = mp.compute_equilibrium_saturation_NaCl_mf(w_s, T_p, m_s_SC0)
    S_eq_SC2 = mp.compute_equilibrium_saturation_NaCl(
                   m_w_SC, m_s_SC0, w_s, R_p_SC, T_p, rho_SC, sigma_SC)
    
    
    ax.plot(m_p_SC, S_eq_SC, label = f"{m_s_SC0:.1e}")
    #ax.plot(R_p_SC, S_eq_SC2, label = "SC2")
    #ax.plot(R_p_SC, S_eq_SC, label = "SC")
    ax.scatter(m_p_init_SC[n], S_eq_SC_init[n], marker="x", s=100)
    ax.axhline(S_amb)
    ax.set_xscale("log")
    ax.legend(loc = "right")
    
axes[1].grid()
axes[1].set_ylim([0.99,1.05])
    


#ax = axes[1]
##    ax.plot(R_p_AS, (S_eq_AS-S_eq_AS2)/S_eq_AS, label = "AS rel dev")
##ax.plot(R_p_AS, S_eq_AS2, label = "AS2")
##ax.plot(R_p_SC, S_eq_SC, label = "SC")
##ax.axhline(S_amb)
#ax.plot(m_p_init_AS, S_eq_AS_init, "x", c = "k")
##ax.plot(m_s_AS, S_eq_AS_init, "x", c = "k")
#ax.set_xscale("log")
#ax.grid()
##ax.legend()


