#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 14:23:37 2020

@author: bohrer
"""

import numpy as np
import matplotlib.pyplot as plt

import materialproperties as mat
#from materialproperties import compute_diffusion_constant
#from materialproperties import compute_heat_of_vaporization
#from materialproperties import compute_thermal_conductivity_air
#from materialproperties import compute_surface_tension_solution
#from materialproperties import compute_saturation_pressure_vapor_liquid

import constants as c
import microphysics as mp

#from microphysics import compute_mass_rate_NaCl
#from microphysics import compute_mass_rate_NaCl_vH
#from microphysics import compute_mass_rate_AS
#from microphysics import compute_R_p_w_s_rho_p


#from microphysics import compute_mass_rate_NaCl

T_p = 290.
T_amb = 290.
p_amb = 101325
S_amb = 1.05

def compute_material_properties(T_amb, p_amb):
    e_s_amb = mat.compute_saturation_pressure_vapor_liquid(T_amb)
    L_v = mat.compute_heat_of_vaporization(T_amb)
    K = mat.compute_thermal_conductivity_air(T_amb)
    D_v = mat.compute_diffusion_constant(T_amb, p_amb)
    return e_s_amb, L_v, K, D_v

e_s_amb, L_v, K, D_v = compute_material_properties(T_p, p_amb)

### NaCl

solute_type = "NaCl"

R_s = 10
m_s = mp.compute_mass_from_radius(R_s, c.mass_density_NaCl_dry)
w_s = np.logspace(-5,np.log10(0.5),1000)

m_w = m_s * (1. / w_s - 1.)

sigma_p = mat.compute_surface_tension_solution(w_s, T_p, solute_type)



R_p, w_s, rho_p = mp.compute_R_p_w_s_rho_p(m_w, m_s, T_p, solute_type)

gamma_NaCl_poly = mp.compute_mass_rate_NaCl(w_s, R_p, T_p, rho_p,
                                       T_amb, p_amb, S_amb, e_s_amb,
                                       L_v, K, D_v, sigma_p)
gamma_NaCl_vH = mp.compute_mass_rate_NaCl_vH(m_w, m_s, w_s, R_p, T_p, rho_p,
                                       T_amb, p_amb, S_amb, e_s_amb,
                                       L_v, K, D_v, sigma_p)

#%%
#fig, ax = plt.subplots(1, 1, figsize = (8,8))
##ax.plot(m_w, gamma_NaCl_vH, label="NaCl vH")
##ax.plot(m_w, gamma_NaCl_poly, label="NaCl poly")
#ax.loglog(m_w,
#        gamma_NaCl_vH,
#        label="vH")
#ax.loglog(m_w,
#        gamma_NaCl_poly,
#        label="poly")
#
#ax.set_xlabel("mass m_w")
#ax.set_ylabel("mass rate")
#ax.grid()
#ax.legend()
#fig.savefig("PLOTS/compare_mass_rate_vH_poly_Nacl.png")
#
##%%
#fig, ax = plt.subplots(1, 1, figsize = (8,8))
##ax.plot(m_w, gamma_NaCl_vH, label="NaCl vH")
##ax.plot(m_w, gamma_NaCl_poly, label="NaCl poly")
#ax.loglog(m_w,
#        np.abs(gamma_NaCl_vH - gamma_NaCl_poly),
#        label="abs diff")
##ax.plot(m_w, gamma_NaCl_poly, label="NaCl poly")
#ax.set_xlabel("mass m_w")
#ax.set_ylabel("mass rate")
#ax.grid()
#ax.legend()
#fig.savefig("PLOTS/compare_mass_rate_vH_poly_Nacl_abs_diff.png")
#
##%%
#fig, ax = plt.subplots(1, 1, figsize = (8,8))
#
##ax.plot(m_w, gamma_NaCl_vH, label="NaCl vH")
##ax.plot(m_w, gamma_NaCl_poly, label="NaCl poly")
#ax.loglog(m_w,
#        np.abs(gamma_NaCl_vH - gamma_NaCl_poly) / np.abs(gamma_NaCl_vH),
#        label="rel diff")
#
##ax.plot(m_w, gamma_NaCl_poly, label="NaCl poly")
#
#ax.set_xlabel("mass m_w")
#ax.set_ylabel("mass rate")
#ax.grid()
#ax.legend()
#fig.savefig("PLOTS/compare_mass_rate_vH_poly_Nacl_rel_diff.png")

#%%

#gamma_NaCl_poly2, dgamma_dm =\
#    mp.compute_mass_rate_and_derivative_NaCl(m_w, m_s, w_s, R_p, T_p, rho_p,
#                                                 T_amb, p_amb, S_amb, e_s_amb,
#                                                 L_v, K, D_v, sigma_p)

#fig, ax = plt.subplots(1, 1, figsize = (8,8))
#
##ax.plot(m_w, gamma_NaCl_vH, label="NaCl vH")
##ax.plot(m_w, gamma_NaCl_poly, label="NaCl poly")
#ax.loglog(m_w,
#        np.abs(gamma_NaCl_poly2 - gamma_NaCl_poly),
#        label="abs diff")
#
##ax.plot(m_w, gamma_NaCl_poly, label="NaCl poly")
#
#ax.set_xlabel("mass m_w")
#ax.set_ylabel("mass rate")
#ax.grid()
#ax.legend()
#fig.savefig("PLOTS/compare_mass_rate_poly1_poly2_w_deri.png")

#%%
# finer resolved m_w

solute_type = "NaCl"

R_s = 10
m_s = mp.compute_mass_from_radius(R_s, c.mass_density_NaCl_dry)
#w_s = np.logspace(-5,np.log10(0.5),1000)

w_s_max = 0.5
w_s_min = 0.1

m_w_min = m_s * (1. / w_s_max - 1.)
m_w_max = m_s * (1. / w_s_min - 1.)

# adjust datapoints here for higher precision of the num. derivative
m_w = np.linspace(m_w_min, m_w_max, 1000000)

w_s = m_s / (m_s + m_w)

R_p, w_s, rho_p = mp.compute_R_p_w_s_rho_p(m_w, m_s, T_p, solute_type)

sigma_p = mat.compute_surface_tension_solution(w_s, T_p, solute_type)


gamma_NaCl_poly2, dgamma_dm =\
    mp.compute_mass_rate_and_derivative_NaCl(m_w, m_s, w_s, R_p, T_p, rho_p,
                                                 T_amb, p_amb, S_amb, e_s_amb,
                                                 L_v, K, D_v, sigma_p)

dgamma_dm_num = (gamma_NaCl_poly2[2:] - gamma_NaCl_poly2[:-2]) \
                / (m_w[2:] - m_w[:-2])
                
fig, ax = plt.subplots(1, 1, figsize = (8,8))

#ax.plot(m_w, gamma_NaCl_vH, label="NaCl vH")
#ax.plot(m_w, gamma_NaCl_poly, label="NaCl poly")

ax.semilogx(m_w[1:-1],
          dgamma_dm[1:-1],
          label="ana")

ax.semilogx(m_w[1:-1],
          dgamma_dm_num,
          label="num")

ax.set_xlabel("mass m_w")
ax.set_ylabel("mass rate deriv.")
ax.grid()
ax.legend()

fig, ax = plt.subplots(1, 1, figsize = (8,8))

ax.loglog(m_w[1:-1],
        np.abs(dgamma_dm[1:-1] - dgamma_dm_num),
        label="abs diff")

ax.set_xlabel("mass m_w")
ax.set_ylabel("mass rate deriv.")
ax.grid()
ax.legend()
fig.savefig("PLOTS/compare_mass_rate_deriv_poly_ana_num_abs_dev.png")

fig, ax = plt.subplots(1, 1, figsize = (8,8))

ax.loglog(m_w[1:-1],
        np.abs(dgamma_dm[1:-1] - dgamma_dm_num)/np.abs(dgamma_dm[1:-1]),
        label="rel diff")

ax.set_xlabel("mass m_w")
ax.set_ylabel("mass rate deriv.")
ax.grid()
ax.legend()
fig.savefig("PLOTS/compare_mass_rate_deriv_poly_ana_num_rel_dev.png")
                
