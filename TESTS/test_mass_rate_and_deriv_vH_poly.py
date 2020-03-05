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
gamma_AS_poly = mp.compute_mass_rate_AS(w_s, R_p, T_p, rho_p,
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
m_w = np.linspace(m_w_min, m_w_max, 1000)

w_s = m_s / (m_s + m_w)

R_p, w_s, rho_p = mp.compute_R_p_w_s_rho_p(m_w, m_s, T_p, solute_type)

sigma_p = mat.compute_surface_tension_solution(w_s, T_p, solute_type)

gamma_NaCl_poly2, dgamma_dm_NaCl_poly2 =\
    mp.compute_mass_rate_and_derivative_NaCl(m_w, m_s, w_s, R_p, T_p, rho_p,
                                                 T_amb, p_amb, S_amb, e_s_amb,
                                                 L_v, K, D_v, sigma_p)
    

#%%
    
from numba import njit
from algebra import compute_polynom
   

 
par_rho_AS = mat.par_rho_AS
par_wat_act_AS = mat.par_wat_act_AS
par_sigma_AS = mat.par_sigma_AS

par_rho_deriv_AS = np.copy(par_rho_AS[:-1]) \
                       * np.arange(1,len(par_rho_AS))[::-1]

par_wat_act_deriv_AS = np.copy(par_wat_act_AS[:-1]) \
                       * np.arange(1,len(par_wat_act_AS))[::-1]
                       
                       
# convert now sigma_w to sigma_p (surface tension)
# return mass rate in fg/s and mass rate deriv in SI: 1/s
def compute_mass_rate_and_derivative_AS_np(m_w, m_s, w_s, R_p, T_p, rho_p,
                                           T_amb, p_amb, S_amb, e_s_amb,
                                           L_v, K, D_v, sigma_p):
#    R_p_SI = 1.0E-6 * R_p # in SI: meter   
    
    # thermal size correction in SI
    l_alpha_plus_R_p = 1.0E-6 * (R_p + mp.compute_l_alpha_lin(T_amb, p_amb, K))
    # diffusive size correction in SI
    l_beta_plus_R_p = 1.0E-6 * (R_p + mp.compute_l_beta_lin(T_amb, D_v) )
       
    m_p_inv_SI = 1.0E18 / (m_w + m_s) # in 1/kg
    #### diff1
    drho_dm_over_rho = -mat.compute_density_water(T_p) * m_p_inv_SI / rho_p\
                       * w_s * compute_polynom(par_rho_deriv_AS, w_s)
    ####                       

    dR_p_dm_over_R_p = c.one_third * ( m_p_inv_SI - drho_dm_over_rho)
    dR_p_dm = 1.0E-6 * dR_p_dm_over_R_p * R_p
    
    eps_k = mp.compute_kelvin_argument(R_p, T_p, rho_p, sigma_p) # in SI - no unit
    kelvin_term = np.exp(eps_k)

#    vH = compute_vant_Hoff_factor_NaCl(w_s)
#    dvH_dws = compute_dvH_dws(w_s)
#    dvH_dws = np.where(w_s < mf_cross_NaCl, np.zeros_like(w_s),
#                       np.ones_like(w_s) * par_vH_NaCl[1])
    # dont convert masses here
#    h1_inv = 1.0 / (m_w + m_s * molar_mass_ratio_w_NaCl * vH) 
    
    # no unit
    a_w = mat.compute_water_activity_AS(w_s)
    
    # in 1/kg
    da_w_dm = -m_p_inv_SI * w_s * compute_polynom(par_wat_act_deriv_AS, w_s)
    
    # IN WORK: UNITS?
    dsigma_dm = -par_sigma_AS * mat.compute_surface_tension_water(T_p) \
                * m_p_inv_SI * ( w_s / ( (1.-w_s)*(1.-w_s) ) )
            
#    S_eq = m_w * h1_inv * np.exp(eps_k)
#    S_eq = a_w * np.exp(eps_k)
    S_eq = a_w * kelvin_term
    
    dSeq_dm = da_w_dm * kelvin_term \
              + S_eq * eps_k * ( dsigma_dm / sigma_p
                                 - drho_dm_over_rho - dR_p_dm_over_R_p )
#        S_eq * (1.0E18 / m_w - eps_k * ( dR_p_dm_over_R_p + drho_dm_over_rho )\
#                - (1 - molar_mass_ratio_w_NaCl * dvH_dws * w_s * w_s)\
#                  * h1_inv * 1.0E18)
    
    c1 = L_v * L_v / (c.specific_gas_constant_water_vapor * K * T_amb * T_amb )
    c2 = c.specific_gas_constant_water_vapor * T_amb / (D_v * e_s_amb)
    # in SI : m^2 s / kg
    f3 = 1.0 / ( (l_alpha_plus_R_p) * S_eq * c1 + (l_beta_plus_R_p) * c2 ) 
    
    f1f3 = 4.0 * np.pi * R_p * R_p * f3 # in 1E-12
    # set l_alpha l_beta constant, i.e. neglect their change with m_p here
    dg1_dm = (dSeq_dm * (l_alpha_plus_R_p) + S_eq * dR_p_dm ) * c1 + dR_p_dm*c2
    # use name S_eq = f2
    S_eq = S_amb - S_eq
#    f2 = S_amb - S_eq
    # NOTE: here S_eq = f2 = S_amb - S_eq
#    return 1.0E-12 * f1f3\
#           * ( S_eq * ( 2.0 * dR_p_dm_over_R_p - f3 * dg1_dm ) - dSeq_dm )
    return 1.0E6 * f1f3 * S_eq,\
           1.0E-12 * f1f3\
           * ( S_eq * ( 2.0 * dR_p_dm_over_R_p - f3 * dg1_dm ) - dSeq_dm )
#           , \
#           dR_p_dm_over_R_p, dg1_dm, dSeq_dm
#    return 1.0E6 * f1f3 * f2,\
#           1.0E-12 * f1f3\
#           * ( f2 * ( 2.0 * dR_p_dm_over_R_p - f3 * dg1_dm ) - dSeq_dm )

compute_mass_rate_and_derivative_AS =\
    njit()(compute_mass_rate_and_derivative_AS_np)
    

gamma_AS_poly3 = mp.compute_mass_rate_AS(w_s, R_p, T_p, rho_p,
                                       T_amb, p_amb, S_amb, e_s_amb,
                                       L_v, K, D_v, sigma_p)
    
gamma_AS_poly2, dgamma_dm_AS =\
     mp.compute_mass_rate_and_derivative_AS(m_w, m_s, w_s, R_p, T_p, rho_p,
                                                 T_amb, p_amb, S_amb, e_s_amb,
                                                 L_v, K, D_v, sigma_p)
#    compute_mass_rate_and_derivative_AS_np(m_w, m_s, w_s, R_p, T_p, rho_p,
#                                                 T_amb, p_amb, S_amb, e_s_amb,
#                                                 L_v, K, D_v, sigma_p)


#
#dgamma_dm_num = (gamma_NaCl_poly2[2:] - gamma_NaCl_poly2[:-2]) \
#                / (m_w[2:] - m_w[:-2])
#                
#fig, ax = plt.subplots(1, 1, figsize = (8,8))
#
##ax.plot(m_w, gamma_NaCl_vH, label="NaCl vH")
##ax.plot(m_w, gamma_NaCl_poly, label="NaCl poly")
#
#ax.semilogx(m_w[1:-1],
#          dgamma_dm[1:-1],
#          label="ana")
#
#ax.semilogx(m_w[1:-1],
#          dgamma_dm_num,
#          label="num")
#
#ax.set_xlabel("mass m_w")
#ax.set_ylabel("mass rate deriv.")
#ax.grid()
#ax.legend()
#
#fig, ax = plt.subplots(1, 1, figsize = (8,8))
#
#ax.loglog(m_w[1:-1],
#        np.abs(dgamma_dm[1:-1] - dgamma_dm_num),
#        label="abs diff")
#
#ax.set_xlabel("mass m_w")
#ax.set_ylabel("mass rate deriv.")
#ax.grid()
#ax.legend()
#fig.savefig("PLOTS/compare_mass_rate_deriv_poly_ana_num_abs_dev.png")
#
#fig, ax = plt.subplots(1, 1, figsize = (8,8))
#
#ax.loglog(m_w[1:-1],
#        np.abs(dgamma_dm[1:-1] - dgamma_dm_num)/np.abs(dgamma_dm[1:-1]),
#        label="rel diff")
#
#ax.set_xlabel("mass m_w")
#ax.set_ylabel("mass rate deriv.")
#ax.grid()
#ax.legend()
#fig.savefig("PLOTS/compare_mass_rate_deriv_poly_ana_num_rel_dev.png")
                
