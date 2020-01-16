#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"material properties" are empirical formula, depending on ambient conditions
general physical laws and derived quantities as well as conversions
might be found in other scripts, as the ideal gas law in "atmosphere.py"
"""

#import math
import numpy as np
from numba import njit, vectorize

import constants as c
from algebra import compute_polynom

### DENSITY OF SOLUTIONS

# water density in kg/m^3
# quad. fit to data from CRC 2005
# relative error is below 0.05 % in the range 0 .. 60°C
par_water_dens = np.array([  1.00013502e+03, -4.68112708e-03, 2.72389977e+02])
#@vectorize("float64(float64)") 
@njit() 
def compute_density_water(temperature_):
    return par_water_dens[0]\
         + par_water_dens[1] * (temperature_ - par_water_dens[2])**2

# NaCl solution density in kg/m^3
# quad. fit to data from CRC 2005
# for w_s < 0.226:
# relative error is below 0.1 % for range 0 .. 50 °C
#                   below 0.17 % in range 0 .. 60 °C
# (only high w_s lead to high error)
par_sol_dens_NaCl = np.array([7.619443952135e+02,   1.021264281453e+03,
                              1.828970151543e+00, 2.405352122804e+02,
                              -1.080547892416e+00,  -3.492805028749e-03 ] )
#@vectorize("float64(float64,float64)")  
@njit()
def compute_density_NaCl_solution(mass_fraction_solute_, temperature_):
    return par_sol_dens_NaCl[0] \
        + par_sol_dens_NaCl[1] * mass_fraction_solute_ \
        + par_sol_dens_NaCl[2] * temperature_ \
        + par_sol_dens_NaCl[3] * mass_fraction_solute_ * mass_fraction_solute_\
        + par_sol_dens_NaCl[4] * mass_fraction_solute_ * temperature_ \
        + par_sol_dens_NaCl[5] * temperature_ * temperature_

# NaCl solution density in kg/m^3
# fit rho(w_s) from Tang 1997 (in Biskos 2006) for room temperature (298 K)
# also used in Haemeri 2000    
# the temperature effect of water is added by multiplication    
par_rho_AS = np.array([ 997.1, 592., -5.036E1, 1.024E1 ] )[::-1] / 997.1
#@vectorize("float64(float64,float64)")  
@njit()
def compute_density_AS_solution(mass_fraction_solute_, temperature_):
    return compute_density_water(temperature_) \
           * compute_polynom(par_rho_AS, mass_fraction_solute_)
#  / 997.1 is included in the parameters now
#    return compute_density_water(temperature_) / 997.1 \
#           * compute_polynom(par_rho_AS, mass_fraction_solute_)

@njit()
def compute_density_solution(mass_fraction_solute_, temperature_, solute_type):
    if solute_type == "AS":
        return compute_density_AS_solution(mass_fraction_solute_,
                                             temperature_)
    elif solute_type == "NaCl":
        return compute_density_NaCl_solution(mass_fraction_solute_,
                                             temperature_)

### SOLUBILITY OF SOLUTIONS
        
# solubility of NaCl in water as mass fraction w_s_sol
# saturation mass fraction (kg_solute/kg_solution)    
# fit to data from CRC 2005 page 8-115
par_solub = np.array([  3.77253081e-01,  -8.68998172e-04,   1.64705858e-06])
@vectorize("float64(float64)") 
def compute_solubility_NaCl(temperature_):
    return par_solub[0] + par_solub[1] * temperature_\
         + par_solub[2] * temperature_ * temperature_

# solubility of ammonium sulfate in water as mass fraction w_s_sol
# saturation mass fraction (kg_solute/kg_solution)    
# fit to data from CRC 2005 page 8-115
par_solub_AS = np.array([0.15767235, 0.00092684])
@vectorize("float64(float64)") 
def compute_solubility_AS(temperature_):
    return par_solub_AS[0] + par_solub_AS[1] * temperature_

def compute_solubility(temperature_, solute_type):
    if solute_type == "AS":
        return compute_solubility_AS(temperature_)
    elif solute_type == "NaCl":
        return compute_solubility_NaCl(temperature_)


### SURFACE TENSION OF SOLUTIONS

#    surface tension in N/m = J/m^2
#    depends on T and not significantly on pressure (see Massoudi 1974)
#    formula from IAPWS 2014
#    note that the surface tension is in gen. dep. on 
#    the mass fraction of the solution (kg solute/ kg solution)
#    which is not considered!
@vectorize("float64(float64)") 
def compute_surface_tension_water(temperature_):
    tau = 1 - temperature_ / 647.096
    return 0.2358 * tau**(1.256) * (1 - 0.625 * tau)


# compute_surface_tension_water(298) = 0.0719953
# molality in mol/kg_water           
# fitted formula by Svenningsson 2006           
# NOTE that the effect of sigma in comparison to sigma_water
# on the kelvin term and the equilibrium saturation is very small
# for small R_s = 5 nm AND small R_p = 6 nm
# the deviation reaches 6 %
# but for larger R_s AND/OR larger R_p=10 nm, the deviation resides at < 1 %            
# this is why it is possible to use the surface tension of water
# for the calculations with NaCl
# note that the deviation is larger for ammonium sulfate           
par_sigma_NaCl_Sven_mol = 1.62E-3           
def compute_surface_tension_NaCl_mol_Sven(molality, temperature):
    return compute_surface_tension_water(temperature)/0.072 \
           * (0.072 + par_sigma_NaCl_Sven_mol * molality )

# formula by Svenningsson 2006, only valid for 0 < w_s < 0.78
# the first term is surface tension of water for T = 298. K
# NOTE that the effect of sigma in comparison to sigma_water
# on the kelvin term and the equilibrium saturation is very small
# for small R_s = 5 nm AND small R_p = 6 nm
# the deviation reaches 6 %
# but for larger R_s AND/OR larger R_p=10 nm, the deviation resides at < 1 %            
#def compute_surface_tension_NaCl(w_s, T_p):
#    return compute_surface_tension_water(T_p)
# compute_surface_tension_AS(0.78, 298.) = 0.154954
par_sigma_NaCl = par_sigma_NaCl_Sven_mol / c.molar_mass_NaCl / 0.072
#par_sigma_AS_Sven /= 0.072
# = 1E3 * par_sigma_AS_Sven_mol / c.molar_mass_AS = 0.01787
# note that Pruppacher gives 0.0234 instead of 0.01787
# use Svens', because it is more recent and measured with higher precision
# also the curve S_eq(R_p) is closer to values of Biskos 2006 curves
@vectorize("float64(float64,float64)")
def compute_surface_tension_NaCl(w_s, T_p):
    return compute_surface_tension_water(T_p) \
               * (1.0 + par_sigma_NaCl * w_s / (1. - w_s))

# formula by Pruppacher 1997, only valid for 0 < w_s < 0.78
# the first term is surface tension of water for T = 298. K
# compute_surface_tension_AS(0.78, 298.) = 0.154954
# 0.0234 / 0.072 = 0.325       
par_sigma_AS_Prup = 0.325
@vectorize("float64(float64,float64)")
def compute_surface_tension_AS_Prup(w_s, T_p):
    return compute_surface_tension_water(T_p) \
               * (1.0 + par_sigma_AS_Prup * w_s / (1. - w_s))
#    return compute_surface_tension_water(T) / 0.072 \
#               * (0.072 + 0.0234 * w_s / (1. - w_s))
#    if w_s > 0.78:
#        return compute_surface_tension_water(T) * 0.154954
#    else:
#        return compute_surface_tension_water(T) / 0.072 \
#               * (0.072 + 0.0234 * w_s / (1. - w_s))

# compute_surface_tension_water(298) = 0.0719953
# molality in mol/kg_water           
par_sigma_AS_Sven_mol = 2.362E-3           
def compute_surface_tension_AS_mol_Sven(molality, temperature):
    return compute_surface_tension_water(temperature)/0.072 \
           * (0.072 + par_sigma_AS_Sven_mol * molality )

# formula by Svenningsson 2006, only valid for 0 < w_s < 0.78
# the first term is surface tension of water for T = 298. K
# compute_surface_tension_AS(0.78, 298.) = 0.154954
par_sigma_AS = par_sigma_AS_Sven_mol / c.molar_mass_AS / 0.072
#par_sigma_AS_Sven /= 0.072
# = 1E3 * par_sigma_AS_Sven_mol / c.molar_mass_AS = 0.01787
# note that Pruppacher gives 0.0234 instead of 0.01787
# use Svens', because it is more recent and measured with higher precision
# also the curve S_eq(R_p) is closer to values of Biskos 2006 curves
@vectorize("float64(float64,float64)")
def compute_surface_tension_AS(w_s, T_p):
    return compute_surface_tension_water(T_p) \
               * (1.0 + par_sigma_AS * w_s / (1. - w_s))
    
    
def compute_surface_tension_solution(w_s, T_p, solute_type):
    if solute_type == "AS":
        return compute_surface_tension_AS(w_s, T_p)
    elif solute_type == "NaCl":
        return compute_surface_tension_NaCl(w_s, T_p)    
    
### WATER ACTIVITY OF SOLUTIONS
        
# VANT HOFF FACTOR -> NOTE that we actually use a polynomal fit a_w(w_s) for
# the water activity of NaCl, the vant Hoff factor is currently not applied
# fitted to match data from Archer 1992
# in formula from Clegg 1997 in table Hargreaves 2010
# rel dev (data from Archer 1972 in formula
# from Clegg 1997 in table Hargreaves 2010) is 
# < 0.12 % for w_s < 0.22
# < 1 % for w_s < 0.25
# approx +6 % for w_s = 0.37
# approx +15 % for w_s = 0.46
# we overestimate the water activity a_w by 6 % and increasing
# 0.308250118 = M_w/M_s
par_vH_NaCl = np.array([ 1.55199086,  4.95679863])
@vectorize("float64(float64)") 
def compute_vant_Hoff_factor_NaCl_fit(mass_fraction_solute_):
    return par_vH_NaCl[0] + par_vH_NaCl[1] * mass_fraction_solute_

vant_Hoff_factor_NaCl_const = 2.0

molar_mass_ratio_w_NaCl = c.molar_mass_water/c.molar_mass_NaCl

#mf_cross_NaCl = 0.090382759623349337
mf_cross_NaCl = 0.09038275962335

# vectorized version
@vectorize("float64(float64)") 
def compute_vant_Hoff_factor_NaCl(mass_fraction_solute_):
#    return compute_vant_Hoff_factor_NaCl_fit(mass_fraction_solute_)
#    return vant_Hoff_factor_NaCl_const
    if mass_fraction_solute_ < mf_cross_NaCl: return vant_Hoff_factor_NaCl_const
    else: return compute_vant_Hoff_factor_NaCl_fit(mass_fraction_solute_)

# numpy version
def compute_vant_Hoff_factor_NaCl_np(mass_fraction_solute_):
#    return compute_vant_Hoff_factor_NaCl_fit(mass_fraction_solute_)
#    return vant_Hoff_factor_NaCl_const
    return np.where(mass_fraction_solute_ < mf_cross_NaCl,
                    vant_Hoff_factor_NaCl_const\
                    * np.ones_like(mass_fraction_solute_),
                    compute_vant_Hoff_factor_NaCl_fit(mass_fraction_solute_))

@vectorize("float64(float64)")
def compute_dvH_dws_NaCl(w_s):
    if w_s < mf_cross_NaCl: return 0.0
    else: return par_vH_NaCl[1]    

# NOTE that we actually use a polynomal fit a_w(w_s) for
# the water activity of NaCl, the vant Hoff factor is currently not applied
@vectorize( "float64(float64, float64, float64)")
def compute_water_activity_NaCl_vH(m_w, m_s, w_s):
    return m_w / ( m_w + molar_mass_ratio_w_NaCl\
                   * compute_vant_Hoff_factor_NaCl(w_s) * m_s )

# WATER ACTIVITY NACL by polynomial (this one is applied)
# Formula from Tang 1997
# note that in Tang, w_t is given in percent
# for NaCl: fix a maximum border for w_s: w_s_max = 0.45
# w_s can not get larger than that.
# the border is chosen, because the approximation of sigma_AS(w_s)
# is only given for 0 < w_s < 0.45
w_s_max_NaCl = 0.45
par_wat_act_NaCl =\
    np.array([1.0, -6.366E-1, 8.642E-1, -1.158E1, 1.518E1])[::-1]
#par_wat_act_AS = par_wat_act_AS[::-1]
@njit()  
def compute_water_activity_NaCl(w_s):
    return compute_polynom(par_wat_act_NaCl, w_s)

# Formula from Biskos 2006, he took it from Tang and Munkelwitz 1994:
# "Water activities, densities, and refractive indices of aqueous sulfates ..."
# data from Kim 1994 agree well
# note that in Tang, w_t is given in percent
# for ammonium sulfate: fix a maximum border for w_s: w_s_max = 0.78
# w_s can not get larger than that.
# the border is chosen, because the approximation of sigma_AS(w_s)
# is only given for 0 < w_s < 0.78
w_s_max_AS = 0.78
par_wat_act_AS = np.array([1.0, -2.715E-1, 3.113E-1, -2.336, 1.412])[::-1]
#par_wat_act_AS = par_wat_act_AS[::-1]
@njit()  
def compute_water_activity_AS(w_s):
    return compute_polynom(par_wat_act_AS, w_s)

### EFFLORESCENCE MASS FRACTION OF SOLUTIONS

# the supersat factor is a crude approximation
# such that the EffRH curve fits data from Biskos better
# there was no math. fitting algorithm included.
# Just graphical shifting the curve... until value
# for D_dry = 10 nm fits with fitted curve of Biskos
# the use here is to get an estimate and LOWER BOUNDARY
# I.e. the water content will not drop below the border given by this value
# to remain on the efflorescence fork
# of course, the supersat factor should be dependent on the dry solute diameter
# from Biskos for ammonium sulfate:
# D_p / D_dry is larger for larger D_dry at the effl point
# (in agreement with Kelvin theory)         
# since S_eq(w_s, T, a_w, m_s, rho_p, sigma_p)         
# and rho_p(w_s, T)         
# sigma_p(w_s, T)          (or take sigma_w(T_p))
# with S_effl(T_p) given, it is hard to find the corresponding w_s         
# this is why the same supersat factor is taken for all dry diameters
# NOTE again that this is only a lower boundary for the water content
# the initialization and mass rate calculation is done with the full
# kelvin raoult term         
supersaturation_factor_NaCl = 1.92
# efflorescence mass fraction at a given temperature
# @njit("[float64[:](float64[:]),float64(float64)]")
@njit()
def compute_efflorescence_mass_fraction_NaCl(temperature_):
    return supersaturation_factor_NaCl * compute_solubility_NaCl(temperature_)

# Take super sat factor for AS such that is fits for D_s = 10 nm 
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
# NOTE that for AS, the w_s_max is fixed independent on temperature to 0.78
# because the fitted material functions only work for 0 < w_s < 0.78           
supersaturation_factor_AS = 2.097
# this determines the lower border of water contained in the particle
# the w_s = m_s / (m_s + m_w) can not get larger than w_s_effl
# (combined with solubility, which is dependent on temperature)

@njit()
def compute_efflorescence_mass_fraction_AS(temperature_):
    return supersaturation_factor_AS * compute_solubility_AS(temperature_)

#%% ATMOSPHERIC PROPERTIES
   
# thermal conductivity in air dependent on ambient temperature in Kelvin 
# empirical formula from Beard and Pruppacher 1971 (in Lohmann, p. 191)
# K_air in W/(m K)
@vectorize("float64(float64)")
def compute_thermal_conductivity_air(temperature_):
    return 4.1868E-3 * ( 5.69 + 0.017 * ( temperature_ - 273.15 ) )

# dynamic viscosity "\mu" in Pa * s
## Fit to Data From Kadoya 1985,
#use linear approx of data in range 250 .. 350 K
#def compute_viscosity_air_approx(T_):
#    return 1.0E-6 * (18.56 + 0.0484 * (T_ - 300))
# ISO ISA 1975: "formula based on kinetic theory..."
# "with Sutherland's empirical coeff.
@vectorize("float64(float64)")
def compute_viscosity_air(T_):
    return 1.458E-6 * T_**(1.5) / (T_ + 110.4)
#@vectorize("float64(float64)", target="parallel") 
#def compute_viscosity_air_par(T_):
#    return 1.458E-6 * T_**(1.5) / (T_ + 110.4)    

# Diffusion coefficient of water vapor in air
# Formula from Pruppacher 1997
# m^2 / s
@vectorize("float64(float64, float64)")
def compute_diffusion_constant(ambient_temperature_ = 293.15,
                               ambient_pressure_ = 101325 ):
    return 4.01218E-5 * ambient_temperature_**1.94 / ambient_pressure_ # m^2/s
#@vectorize("float64(float64, float64)", target="parallel") 
#def compute_diffusion_constant_par(ambient_temperature_,
#                               ambient_pressure_):
#    return 4.01218E-5 * ambient_temperature_**1.94 / ambient_pressure_ # m^2/s

# latent enthalpy of vaporazation of water in J/kg
# formula by Dake 1972
# formula valid for 0 °C to 35 °C
# At NTP: 2.452E6 # J/kg
@vectorize("float64(float64)")
def compute_heat_of_vaporization(temperature_):
    return 1.0E3 * ( 2500.82 - 2.358 * (temperature_ - 273.0) ) 

# saturation pressure for the vapor-liquid phase transition of water
# Approximation by Rogers and Yau 1989 (in Lohmann 2016,  p.50)
# returns pressure in Pa = N/m^2
@vectorize("float64(float64)")
def compute_saturation_pressure_vapor_liquid(temperature_):
    return 2.53E11 * np.exp( -5420.0 / temperature_ )
#@vectorize("float64(float64)", target="parallel") 
#def compute_saturation_pressure_vapor_liquid_par(temperature_):
#    return 2.53E11 * np.exp( -5420.0 / temperature_ )
    