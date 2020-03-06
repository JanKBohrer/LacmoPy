#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 13:39:40 2019

@author: jdesk
"""

# https://pubchem.ncbi.nlm.nih.gov/compound/Ammonium-sulfate
molar_mass_AS = 132.14

# at efflorescence point (Onasch 1999): n_w/n_s approx 1.8 and independent on T
# m_w/m_s = 1.8 * 18/132.14 = 0.245
# w_s = 1/ (1 + m_w/m_s)
w_s_effl_AS_0 = 1./ (1. + 0.245)
print(w_s_effl_AS_0)

#%%

# Lide 2005:
# Solubility of (NH4)_2 SO4
# REF: 6 on page 8-110:
# Söhnel, O., and Novotny, P.,
# Densities of Aqueous Solutions of Inorganic Substances,
# Elsevier, Amsterdam, 1985.
# T; w_s
# 0 10 20 25 30 40 50 60 70 80 90 100
# 41.3  42.1  42.9  43.3  43.8  44.7  45.6  46.6  47.5  48.5  49.5  50.5 


# Onasch 1999
# Ammonium sulfate ((NH4)_2 SO4)
#The median diameter of the distributed
#particle surface area
#derived from the lognormal Mie fits ranged from 0.65 to 0.75 gm
#with a geometric standard deviation of 1.8 to 1.6, respectively.
#Using these median diameters and density data from Tang and
#Munkelwitz [1991] for supersaturated solutions at 80 wt%
#ammonium sulfate by mass (concentration of solution at 37%
#RH), the size of solution droplets in our experiments near the
#efflorescence
#point at 298 K are calculated to range from 0.75 to
#0.87 gm in diameter
## Efflorescence Point:
# Molar ratios water:AS
# T; mol ratio; error
# 294.8 293.5 273.6 264.0 263.9 263.9 263.1 253.8 249.1 244.1 234.1
# 1.9 1.5 1.8 1.6 1.8 1.7 1.7 1.7 1.9 1.9 1.7
# 0.4 0.3 0.3 0.3 0.3 0.3 0.3 0.3 0.4 0.4 0.3

# Biskos 2006: (w_t = weight percent of solute)
# give formula for rho_s, rho_aq (w_t), a_w(w_t) and sigma_aq directly
# maybe, rho_aq can be weighted with some factor for temperature dependence

# Biskos: curves for diameter = 6,8,10,20,40,60 nanometer
# -> not much variation in the point of ERH
# combined with results from Onasch: same order of magnitude
# and we get ERH(T) not dependent on R!


import numpy as np
import matplotlib.pyplot as plt

import constants as c

import microphysics as mp

from scipy.optimize import curve_fit

from numba import vectorize, njit

# %% Solubility -> needed ??


# par[0] belongs to the largest exponential x^(n-1) for par[i], i = 0, .., n 
@njit()
def compute_polynom(par,x):
    res = par[0] * x + par[1]
    for a in par[2:]:
        res = res * x + a
    return res

def fit_lin(x, a, b):
    return a + b * x

#par_solub_AS = np.array([0.15767235, 0.00092684])


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
par_wat_act_AS = np.array([1.0, -2.715E-3, 3.113E-5, -2.336E-6, 1.412E-8 ])
for i in range(1,len(par_wat_act_AS)):
    par_wat_act_AS[i] *= 100.0**i
    print(par_wat_act_AS[i])
par_wat_act_AS = par_wat_act_AS[::-1]

@njit()  
def compute_water_activity_AS(w_s):
    return compute_polynom(par_wat_act_AS, w_s)

from microphysics import compute_density_water
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

#
from microphysics import compute_surface_tension_water           
# formula by Pruppacher 1997, only valid for 0 < w_s < 0.78
# the first term is surface tension of water for T = 298. K
# compute_surface_tension_AS(0.78, 298.) = 0.154954
@vectorize()
def compute_surface_tension_AS(w_s, T):
    return compute_surface_tension_water(T) \
               * (1.0 + 0.325 * w_s / (1. - w_s))
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

from microphysics import compute_kelvin_term

@vectorize(
    "float64(float64, float64, float64, float64, float64)")
def compute_equilibrium_saturation_AS(w_s, R_p, rho_p, T, sigma_w):
    return compute_water_activity_AS(w_s)\
           * compute_kelvin_term(R_p, rho_p, T, sigma_w)

from microphysics import compute_kelvin_term_mf

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

#%%
           
T = np.array((0., 10, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100))
T += 273.15
solub = np.array((41.3,  42.1,  42.9,  43.3,  43.8,  44.7,
                  45.6,  46.6,  47.5,  48.5,  49.5,  50.5 ))
solub *= 0.01

p, cov = curve_fit(fit_lin, T, solub)           

w_s = np.linspace(0.0,0.8)

no_rows = 2
fig, axes = plt.subplots(no_rows, figsize=(10,no_rows*5))
ax = axes[0]
ax.plot(T, solub, "x")
ax.plot(T, fit_lin(T, *p))
ax.plot(T, compute_solubility_AS(T))
ax.set_title(r"solubility vs temperature for $(NH_4)_2 \, SO_4$")
ax = axes[1]
ax.plot(w_s, compute_water_activity_AS(w_s), "x")
ax.set_title(r"water activity vs mass fraction for $(NH_4)_2 \, SO_4$")

#plt.suptitle()

no_rows = 2
fig, axes = plt.subplots(no_rows, figsize=(10,no_rows*5))
ax = axes[0]
ax.plot(T, mp.compute_solubility_NaCl(T), "x")
ax.set_title(r"solubility vs temperature for NaCl")
ax = axes[1]
ax.plot(w_s,
        mp.compute_water_activity_mf(
                w_s,
                mp.compute_vant_Hoff_factor_NaCl(w_s)),
                "x")
ax.set_title(r"water activity vs mass fraction for NaCl")

no_rows = 2
fig, axes = plt.subplots(no_rows, figsize=(10,no_rows*5))
ax = axes[0]
ax.plot(T, mp.compute_density_water(T))
ax.plot(T, compute_density_AS_solution(0.0,T))
ax.set_title("density of water vs temperature")
ax = axes[1]
for w_s_ in np.linspace(0.0,0.23,10):
    ax.plot(T, compute_density_AS_solution(w_s_,T),
            label = f"{w_s_:.2}")
ax.legend()    
ax.set_title("density of $(NH_4)_2 \, SO_4 - H_2O$ solution vs temperature, "
             + "legend = mass fraction solute")


#%%
T = 298.
#D_s_list = np.array([6,8,10,20,40,60]) * 1E-3
#R_s_list = D_s_list * 0.5

D_s = 10E-3 # mu = 10 nm
R_s = 0.5 * D_s
m_s = mp.compute_mass_from_radius_jit(R_s, c.mass_density_AS_dry)

w_s = np.logspace(-2., np.log10(0.78), 100)
rho_p = compute_density_AS_solution(w_s, T) 
m_p = m_s/w_s
R_p = mp.compute_radius_from_mass_jit(m_p, rho_p)

sigma_p = compute_surface_tension_AS(w_s, T)

S_eq = \
    compute_kelvin_raoult_term_mf_AS(w_s,
                                     m_s,
                                     T,
                                     rho_p,
                                     sigma_p)
S_eq = \
    compute_equilibrium_saturation_AS(w_s, R_p, rho_p, T, sigma_p)
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(R_p, S_eq)


#%% test S_eq for AS

#import constants as c

T = 298.
#D_s = 10E-3 # mu = 10 nm
D_s_list = np.array([6,8,10,20,40,60]) * 1E-3
#R_s = 0.5 * D_s
R_s_list = D_s_list * 0.5
#m_s = mp.compute_mass_from_radius_jit(R_s, c.mass_density_AS_dry)
#
#w_s = np.logspace(-1, 0, 100)
#rho_p = compute_density_AS_solution(w_s, T) 
#m_p = m_s/w_s
#R_p = mp.compute_radius_from_mass_jit(m_p, rho_p)
#
#sigma_w = mp.compute_surface_tension_water(T)
#
#S_eq = \
#    compute_kelvin_raoult_term_mf_AS(w_s,
#                                     m_s,
#                                     T,
#                                     rho_p,
#                                     sigma_w)

w_s = np.logspace(-0.5, np.log10(0.78), 100)

no_rows = 1
fig, axes = plt.subplots(no_rows, figsize=(10,no_rows*10))
ax = axes

for i,R_s in enumerate(R_s_list):
    

    m_s = mp.compute_mass_from_radius_jit(R_s, c.mass_density_AS_dry)
    rho_p = compute_density_AS_solution(w_s, T) 
    m_p = m_s/w_s
    R_p = mp.compute_radius_from_mass_jit(m_p, rho_p)
    
#    sigma_w = mp.compute_surface_tension_water(T)
    
    sigma_AS = compute_surface_tension_AS(w_s, T)
    
    S_eq = \
        compute_equilibrium_saturation_AS(w_s, R_p, rho_p, T, sigma_AS)
#    S_eq = \
#        compute_kelvin_raoult_term_mf_AS(w_s,
#                                         m_s,
#                                         T,
#                                         rho_p,
#                                         sigma_AS)

    if i == 0:
        ax.plot(R_p/R_s, w_s, label= r"$w_s$")
#ax.plot(S_eq, R_p/R_s, "x")
    ax.plot(R_p/R_s, S_eq, "x", label=f"{R_s*2E3}")
    
#ax.plot(1.09, compute_solubility_AS(T), "D")
ax.plot(R_p/R_s,
        np.ones_like(R_p) * compute_efflorescence_mass_fraction_AS(T))
ax.axvline(1.09)
ax.axhline(0.40)
ax.axhline(0.35)
ax.axhline(0.33)
ax.axhline(0.3)
ax.axhline(0.28)
ax.plot(R_p/R_s, np.ones_like(R_p) * compute_solubility_AS
        (T))
ax.legend()
ax.set_yticks(np.arange(0.1,1.1,0.1))
ax.grid()
ax.set_title("S_eq vs R_p for D_s = 10 nm for SA")


#%%

#for w_s_ in w_s:
#    print(compute_surface_tension_AS(w_s_,T))

#%%

#for T_ in np.arange(290,300,0.1):
#    print(T_, mp.compute_density_water(T_))
