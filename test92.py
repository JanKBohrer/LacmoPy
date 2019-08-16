#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 10:04:00 2019

@author: jdesk
"""
import math
import numpy as np
from numba import njit, vectorize
import matplotlib.pyplot as plt


import microphysics as mp
import constants as c
from atmosphere import compute_surface_tension_water,\
                       compute_specific_heat_capacity_air_moist,\
                       compute_diffusion_constant,\
                       compute_thermal_conductivity_air
                
                

#%%
# water density in kg/m^3
# quad. fit to data from CRC 2005
# relative error is below 0.05 % in the range 0 .. 60°C
par_water_dens = np.array([  1.00013502e+03, -4.68112708e-03, 2.72389977e+02])
@vectorize("float64(float64)")  
def compute_density_water(temperature_):
    return par_water_dens[0]\
         + par_water_dens[1] * (temperature_ - par_water_dens[2])**2

@njit()
def compute_kelvin_term_mf(mass_fraction_solute_,
                        temperature_,
                        mass_solute_,
                        mass_density_particle_,
                        surface_tension_ ):
    return np.exp( 2.0 * surface_tension_ * 1.0E6\
                   * (mass_fraction_solute_ / mass_solute_)**(c.one_third)
                   / ( c.specific_gas_constant_water_vapor
                       * temperature_
                       * mass_density_particle_**(0.66666667)
                       * c.const_volume_to_radius)
                 )

# R_p in mu
# result without unit -> conversion from mu 
@vectorize("float64(float64,float64,float64,float64)")
def compute_kelvin_argument(R_p, T_p, rho_p, sigma_w):
    return 2.0E6 * sigma_w\
                   / ( c.specific_gas_constant_water_vapor * T_p * rho_p * R_p )

@vectorize( "float64(float64, float64, float64, float64)")
def compute_kelvin_term(R_p, T_p, rho_p, sigma_w):
    return np.exp(compute_kelvin_argument(R_p, T_p, rho_p, sigma_w))
                       
### mass rate
    
# Size corrections Fukuta (both in mu!)
# size corrections in droplet growth equation of Fukuta 1970
# used in Szumowski 1998 
# we use adiabatic index = 1.4 = 7/5 -> 1/1.4 = 5/7 = 0.7142857142857143
# also accommodation coeff = 1.0
# and c_v_air = c_v_air_dry_NTP
accommodation_coeff = 1.0
adiabatic_index_inv = 0.7142857142857143

T_alpha_0 = 289 # K
c_alpha_1 = 1.0E6 * math.sqrt(2.0 * np.pi * c.specific_gas_constant_air_dry 
                              * T_alpha_0 )\
                    / ( accommodation_coeff
                        * ( c.specific_heat_capacity_air_dry_NTP
                            * adiabatic_index_inv\
                            + 0.5 * c.specific_gas_constant_air_dry ) )
c_alpha_2 = 0.5E6\
            * math.sqrt(2.0 * np.pi * c.specific_gas_constant_air_dry
                        / T_alpha_0 )\
              / ( accommodation_coeff
                  * (c.specific_heat_capacity_air_dry_NTP * adiabatic_index_inv\
                      + 0.5 * c.specific_gas_constant_air_dry) )
# in mu
@vectorize("float64(float64,float64,float64)")
def compute_l_alpha_lin(T_amb, p_amb, K):
    return ( c_alpha_1 + c_alpha_2 * (T_amb - T_alpha_0) ) * K / p_amb

condensation_coeff = 0.0415
c_beta_1 = 1.0E6 * math.sqrt( 2.0 * np.pi * c.molar_mass_water\
                              / ( c.universal_gas_constant * T_alpha_0 ) )\
                   / condensation_coeff
c_beta_2 = -0.5 / T_alpha_0
# in mu
@vectorize("float64(float64,float64)")
def compute_l_beta_lin(T_amb, D_v):
    return c_beta_1 * ( 1.0 + c_beta_2 * (T_amb - T_alpha_0) ) * D_v

                
#%% AMMONIUM SULFATE
# for ammonium sulfate: fix a maximum border for w_s: w_s_max = 0.78
# w_s can not get larger than that.
# the border is chosen, because the approximation of sigma_AS(w_s)
# is only given for 0 < w_s < 0.78
w_s_max_AS = 0.78

# par[0] belongs to the largest exponential x^(n-1) for par[i], i = 0, .., n 
@njit()
def compute_polynom(par,x):
    res = par[0] * x + par[1]
    for a in par[2:]:
        res = res * x + a
    return res

# solubility of ammonium sulfate in water as mass fraction w_s_sol
# saturation mass fraction (kg_solute/kg_solution)    
# fit to data from CRC 2005 page 8-115
par_solub_AS = np.array([0.15767235, 0.00092684])
@vectorize("float64(float64)") 
def compute_solubility_AS(temperature_):
    return par_solub_AS[0] + par_solub_AS[1] * temperature_

# formula from Biskos 2006, he took it from Tang, Munkelwitz 1994 NOTE
# that the citation is wrong in his paper. it is NOT Tang 1997 but (I guess)
# Tang 1994:
# Water activities, densities, and refractive indices of aqueous sulfates
# and sodium nitrate droplets of atmospheric importance     
# I do not have access however...
# data from Kim 1994 agree well
par_wat_act_AS = np.array([1.0, -2.715E-1, 3.113E-1, -2.336, 1.412 ])[::-1]
#par_wat_act_AS = par_wat_act_AS[::-1]

@njit()  
def compute_water_activity_AS(w_s):
    return compute_polynom(par_wat_act_AS, w_s)

# NaCl solution density in kg/m^3
# data rho(w_s) from Tang 1994 (in Biskos 2006)
# this is for room temperature (298 K)
# then temperature effect of water by multiplication    
par_rho_AS = np.array([ 997.1, 592., -5.036E1, 1.024E1 ] )[::-1] / 997.1
@vectorize("float64(float64,float64)")  
def compute_density_AS_solution(mass_fraction_solute_, temperature_):
    return compute_density_water(temperature_) \
           * compute_polynom(par_rho_AS, mass_fraction_solute_)
#  / 997.1 is included in the parameters now
#    return compute_density_water(temperature_) / 997.1 \
#           * compute_polynom(par_rho_AS, mass_fraction_solute_)

# formula by Pruppacher 1997, only valid for 0 < w_s < 0.78
# the first term is surface tension of water for T = 298. K
# compute_surface_tension_AS(0.78, 298.) = 0.154954
par_sigma_AS = 0.325
@vectorize("float64(float64,float64)")
def compute_surface_tension_AS(w_s, T):
    return compute_surface_tension_water(T) \
               * (1.0 + par_sigma_AS * w_s / (1. - w_s))
#    return compute_surface_tension_water(T) / 0.072 \
#               * (0.072 + 0.0234 * w_s / (1. - w_s))
#    if w_s > 0.78:
#        return compute_surface_tension_water(T) * 0.154954
#    else:
#        return compute_surface_tension_water(T) / 0.072 \
#               * (0.072 + 0.0234 * w_s / (1. - w_s))

# ->> Take again super sat factor for AS such that is fits for D_s = 10 nm 
# other data from Haemeri 2000 and Onasch 1999 show similar results
# Haemeri: also 8,10,20 nm, but the transition at effl point not detailed
# Onasch: temperature dependence: NOT relevant in our range!
# S_effl does not change significantly in range 273 - 298 Kelvin
# Increases for smaller temperatures, note however, that they give
# S_effl = 32% pm 3%, while Cziczo 1997 give 33 % pm 1 % at 298 K           
# with data from Biskos 2006 -> ASSUME AS LOWER BORDER
# i.e. it is right for small particles with small growth factor of 1.1
# at efflorescence
# larger D_s will have larger growth factors at efflorescence
# thus larger water mass compared to dry mass and thus SMALLER w_s_effl
# than W_s_effl of D_s = 10 nm
# at T = 298., D_s = 10 nm, we find solubility mass fraction of 0.43387067
# and with ERH approx 35 % a growth factor of approx 1.09
# corresponding to w_s = 0.91
# note that Onasch gives w_s_effl of 0.8 for D_s_dry approx 60 to 70
# i.e. a SMALLER w_s_effl (more water)
# the super sat factor is thus
# supersaturation_factor_AS = 0.91/0.43387067
supersaturation_factor_AS = 2.097
# this determines the lower border of water contained in the particle
# the w_s = m_s / (m_s + m_w) can not get larger than w_s_effl
# (combined with solubility, which is dependent on temperature)
@njit()
def compute_efflorescence_mass_fraction_AS(temperature_):
    return supersaturation_factor_AS * compute_solubility_AS(temperature_)

@vectorize(
    "float64(float64, float64, float64, float64, float64)")
def compute_equilibrium_saturation_AS(w_s, R_p, rho_p, T, sigma_w):
    return compute_water_activity_AS(w_s)\
           * compute_kelvin_term(R_p, rho_p, T, sigma_w)

@njit()
def compute_kelvin_raoult_term_mf_AS(mass_fraction_solute_,
                                     mass_solute_,
                                      temperature_,
                                      mass_density_particle_,
                                      surface_tension_):
    return compute_water_activity_AS(mass_fraction_solute_) \
           * compute_kelvin_term_mf(mass_fraction_solute_,
                                    temperature_,
                                    mass_solute_,
                                    mass_density_particle_,
                                    surface_tension_)

# %% NEW PARAMETR. OF SURFACE TENSION:
# compute_surface_tension_water(298) = 0.0719953
# molality in mol/kg_water           
par_AS_Sven = 2.362E-3           
def compute_sigma_AS_Sven(molality, temperature):
    return compute_surface_tension_water(temperature)/0.072 \
           * (0.072 + par_AS_Sven * molality )

# compute_surface_tension_water(298) = 0.0719953
# molality in mol/kg_water           
par_NaCl_Sven = 1.62E-3           
def compute_sigma_NaCl_Sven(molality, temperature):
    return compute_surface_tension_water(temperature)/0.072 \
           * (0.072 + par_NaCl_Sven * molality )
          

#%% MASS RATE DERIV AS
           
#par_rho_deriv_AS = np.copy(par_rho_AS)
#for n in range(len(par_rho_deriv_AS)):
#    par_rho_deriv_AS[n] *= (len(par_rho_deriv_AS)-1-n)
#
#par_wat_act_deriv_AS = np.copy(par_wat_act_AS)
#for n in range(len(par_wat_act_deriv_AS)):
#    par_wat_act_deriv_AS[n] *= (len(par_wat_act_deriv_AS)-1-n)
#    
## convert now sigma_w to sigma_p (surface tension)
## return mass rate in fg/s and mass rate deriv in SI: 1/s
#def compute_mass_rate_and_derivative_AS_np(m_w, m_s, w_s, R_p, T_p, rho_p,
#                                           T_amb, p_amb, S_amb, e_s_amb,
#                                           L_v, K, D_v, sigma_p):
##    R_p_SI = 1.0E-6 * R_p # in SI: meter   
#    
#    # thermal size correction in SI
#    l_alpha_plus_R_p = 1.0E-6 * (R_p + compute_l_alpha_lin(T_amb, p_amb, K))
#    # diffusive size correction in SI
#    l_beta_plus_R_p = 1.0E-6 * (R_p + compute_l_beta_lin(T_amb, D_v) )
#       
#    m_p_inv_SI = 1.0E18 / (m_w + m_s) # in 1/kg
#    # dont use piecewise for now to avoid discontinuity in density...
#    drho_dm_over_rho = -compute_density_water(T_p) * m_p_inv_SI / rho_p\
#                       * compute_polynom(par_rho_deriv_AS, w_s)
#                           
#    dR_p_dm_over_R_p = c.one_third * ( m_p_inv_SI - drho_dm_over_rho)
#    dR_p_dm = 1.0E-6 * dR_p_dm_over_R_p * R_p
#    
#    eps_k = compute_kelvin_argument(R_p, T_p, rho_p, sigma_p) # in SI - no unit
#    
##    vH = compute_vant_Hoff_factor_NaCl(w_s)
##    dvH_dws = compute_dvH_dws(w_s)
##    dvH_dws = np.where(w_s < mf_cross_NaCl, np.zeros_like(w_s),
##                       np.ones_like(w_s) * par_vH_NaCl[1])
#    # dont convert masses here
##    h1_inv = 1.0 / (m_w + m_s * molar_mass_ratio_w_NaCl * vH) 
#
#    a_w = compute_water_activity_AS(w_s)
#    
#    da_w_dm = - m_p_inv_SI * compute_polynom(par_wat_act_deriv_AS, w_s)
#    
#    dsigma_dm = -par_sigma_AS * compute_surface_tension_water(T_p) \
#                * m_p_inv_SI * ( w_s / ( (1.-w_s)*(1.-w_s) ) )
#            
##    S_eq = m_w * h1_inv * np.exp(eps_k)
#    
#    S_eq = a_w * np.exp(eps_k)
#    
#    dSeq_dm =\
#        S_eq * ( da_w_dm / a_w + dsigma_dm / sigma_p - drho_dm_over_rho - dR_p_dm_over_R_p )
##        S_eq * (1.0E18 / m_w - eps_k * ( dR_p_dm_over_R_p + drho_dm_over_rho )\
##                - (1 - molar_mass_ratio_w_NaCl * dvH_dws * w_s * w_s)\
##                  * h1_inv * 1.0E18)
#    
#    c1 = L_v * L_v / (c.specific_gas_constant_water_vapor * K * T_amb * T_amb )
#    c2 = c.specific_gas_constant_water_vapor * T_amb / (D_v * e_s_amb)
#    # in SI : m^2 s / kg
#    f3 = 1.0 / ( (l_alpha_plus_R_p) * S_eq * c1 + (l_beta_plus_R_p) * c2 ) 
#    
#    f1f3 = 4.0 * np.pi * R_p * R_p * f3 # in 1E-12
#    # set l_alpha l_beta constant, i.e. neglect their change with m_p here
#    dg1_dm = (dSeq_dm * (l_alpha_plus_R_p) + S_eq * dR_p_dm ) * c1 + dR_p_dm*c2
#    # use name S_eq = f2
#    S_eq = S_amb - S_eq
##    f2 = S_amb - S_eq
#    # NOTE: here S_eq = f2 = S_amb - S_eq
##    return 1.0E-12 * f1f3\
##           * ( S_eq * ( 2.0 * dR_p_dm_over_R_p - f3 * dg1_dm ) - dSeq_dm )
#    return 1.0E6 * f1f3 * S_eq,\
#           1.0E-12 * f1f3\
#           * ( S_eq * ( 2.0 * dR_p_dm_over_R_p - f3 * dg1_dm ) - dSeq_dm )
##    return 1.0E6 * f1f3 * f2,\
##           1.0E-12 * f1f3\
##           * ( f2 * ( 2.0 * dR_p_dm_over_R_p - f3 * dg1_dm ) - dSeq_dm )
#compute_mass_rate_and_derivative_AS =\
#njit()(compute_mass_rate_and_derivative_AS_np)
#compute_mass_rate_and_derivative_AS_par =\
#njit(parallel = True)(compute_mass_rate_and_derivative_AS_np)

#%%

@njit()
# jitted for single mass value when function is used in another jitted function
def compute_mass_from_radius_jit(radius_, density_):
    return c.pi_times_4_over_3 * density_ * radius_ * radius_ * radius_ 

@njit()
# jitted for single mass value when function is used in another jitted function
def compute_radius_from_mass_jit(mass_, density_):
    return   ( c.pi_times_4_over_3_inv * mass_ / density_ ) ** (c.one_third)

T = 298.
##D_s_list = np.array([6,8,10,20,40,60]) * 1E-3
##R_s_list = D_s_list * 0.5
#
D_s = 10E-3 # mu = 10 nm
R_s = 0.5 * D_s
m_s = compute_mass_from_radius_jit(R_s, c.mass_density_AS_dry)
#
w_s = np.logspace(-2., np.log10(0.78), 100)
rho_p = compute_density_AS_solution(w_s, T) 
m_p = m_s/w_s
m_w = m_p - m_s

R_p = compute_radius_from_mass_jit(m_p, rho_p)
#

sigma_w = mp.compute_surface_tension_water(T)
sigma_AS = compute_surface_tension_AS(w_s, T)

molal = mp.compute_molality_from_mass_fraction(w_s, c.molar_mass_AS*1E-3)

sigma_AS_Sven = compute_sigma_AS_Sven(molal, T)
sigma_NaCl_Sven = compute_sigma_NaCl_Sven(molal, T)

#

a_w = compute_water_activity_AS(w_s)

kelvin_term_AS = compute_kelvin_term_mf(w_s, T, m_s, rho_p, sigma_AS)


S_eq = \
    compute_kelvin_raoult_term_mf_AS(w_s,
                                     m_s,
                                     T,
                                     rho_p,
                                     sigma_AS)
S_eq_AS_sigma_w = \
    compute_kelvin_raoult_term_mf_AS(w_s,
                                     m_s,
                                     T,
                                     rho_p,
                                     sigma_w)

###

rho_NaCl = mp.compute_density_NaCl_solution(w_s, T)
#m_p_NaCl = m_s/w_s
R_p_NaCl = compute_radius_from_mass_jit(m_p, rho_NaCl)

a_w_NaCl = mp.compute_water_activity(m_w, m_s, w_s)
S_eq_NaCl = mp.compute_kelvin_raoult_term_NaCl_mf(w_s, T, m_s)

S_eq_NaCl_sven = a_w_NaCl * compute_kelvin_term(R_p_NaCl, T,
                                                rho_NaCl, sigma_NaCl_Sven)

#%%

# exp data
wt_percent_NaCl = np.load("wt_percent_NaCl.npy")
w_s_exp = wt_percent_NaCl * 0.01
a_w_exp = np.load("water_act_NaCl_low.npy")

m_p_exp = m_s/w_s_exp

rho_NaCl_exp = mp.compute_density_NaCl_solution(w_s_exp, T)
R_p_NaCl_exp = compute_radius_from_mass_jit(m_p_exp, rho_NaCl_exp)



#import matplotlib.pyplot as plt

data = [rho_p, sigma_AS, a_w, kelvin_term_AS, S_eq]
data_string = ["rho_p", "sigma", "a_w", "kelvin_term", "S_eq"]

no_rows = 5

fig, axes = plt.subplots(no_rows, figsize=(10,6*no_rows))

for n in range(no_rows):
    axes[n].plot(R_p, data[n])
    axes[n].set_ylabel(data_string[n])

axes[0].plot(R_p_NaCl, rho_NaCl)    
axes[1].plot(R_p, sigma_AS_Sven, label = "Sven AS")    
axes[1].plot(R_p, sigma_NaCl_Sven, label = "Sven NaCl")    
axes[1].plot(R_p, np.ones_like(R_p)*sigma_w, label = "water")  
axes[1].legend()  
axes[2].plot(R_p_NaCl, a_w_NaCl)    
axes[2].plot(R_p_NaCl_exp, a_w_exp)    
axes[4].plot(R_p_NaCl, S_eq_AS_sigma_w, label = "AS sigma_w")    
axes[4].plot(R_p_NaCl, S_eq_NaCl, label = "NaCl sigma_w")    
axes[4].plot(R_p_NaCl, S_eq_NaCl_sven, label = "NaCl Sven")    
axes[4].legend()
#axes[4].plot(w_s_exp, a_w_exp)

#ax.plot(R_p, rho_p)
#ax.plot(R_p, sigma_p)
#ax.plot(R_p, a_w)
#ax.plot(R_p, S_eq)

#def compute_kelvin_raoult_term_mf_AS(mass_fraction_solute_,
#                                     mass_solute_,
#                                      temperature_,
#                                      mass_density_particle_,
#                                      surface_tension_):




