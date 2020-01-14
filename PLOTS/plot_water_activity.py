#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 13:03:34 2019

@author: jdesk
"""

import numpy as np

import matplotlib.pyplot as plt

from microphysics import compute_water_activity_AS
from microphysics import compute_water_activity_NaCl
from microphysics import compute_mass_from_radius_jit
from microphysics import compute_radius_from_mass_jit
from microphysics import compute_molality_from_mass_fraction
from microphysics import compute_mass_fraction_from_molality
from microphysics import compute_density_AS_solution
from microphysics import compute_density_NaCl_solution
from microphysics import compute_equilibrium_saturation_AS_mf
from microphysics import compute_equilibrium_saturation_NaCl_mf
from microphysics import compute_R_p_w_s_rho_p_AS
from microphysics import compute_R_p_w_s_rho_p_NaCl

import constants as c

#compute_water_activity_AS(w_s)

#compute_water_activity_NaCl(m_w, m_s, w_s):

#compute_molality_from_mass_fraction(mass_fraction_, molecular_weight_)

#compute_equilibrium_saturation_AS(w_s, R_p, T_p, rho_p, sigma_w)
#compute_equilibrium_saturation_AS_mf(w_s, T_p, m_s)

rho_AS = c.mass_density_AS_dry
rho_SC = c.mass_density_NaCl_dry


T_p = 285
R_s = 10E-3

m_AS = compute_mass_from_radius_jit(R_s, rho_AS)
m_SC = compute_mass_from_radius_jit(R_s, rho_SC)

w_s_max_AS = 0.6

w_s = np.logspace(-5,np.log10(w_s_max_AS),100)

#m_w = np.logspace( np.log10(m_AS) )

m_w_AS = m_AS * (1/w_s - 1)
m_w_SC = m_SC * (1/w_s - 1)

aw_AS = compute_water_activity_AS(w_s)
aw_SC = compute_water_activity_NaCl(m_w_SC, m_SC, w_s)

fig, axes = plt.subplots(figsize=(6,6))

ax = axes
ax.plot(m_w_AS, aw_AS)
ax.plot(m_w_SC, aw_SC)
ax.set_xscale("log")
#ax.set_yscale("log")

###

S_AS = compute_equilibrium_saturation_AS_mf(w_s, T_p, m_AS)
S_SC = compute_equilibrium_saturation_NaCl_mf(w_s, T_p, m_SC)

R_AS,_,_ = compute_R_p_w_s_rho_p_AS(m_w_AS, m_AS, T_p)
R_SC,_,_ = compute_R_p_w_s_rho_p_NaCl(m_w_SC, m_SC, T_p)

fig2, axes = plt.subplots(figsize=(6,6))
ax = axes
ax.plot(R_AS, S_AS)
ax.plot(R_SC, S_SC)
ax.set_xscale("log")

#fig2, axes = plt.subplots(figsize=(6,12))
#
#mol_SC = compute_molality_from_mass_fraction(w_s, c.molar_mass_NaCl)
#
#ax = axes
##ax.plot(m_w_AS, aw_AS)
#ax.plot(mol_SC, aw_SC)
#ax.set_xscale("log")
#ax.set_yticks(np.arange(0.18,1.02,0.02))
#ax.grid(which="both")
##ax.set_yscale("log")
