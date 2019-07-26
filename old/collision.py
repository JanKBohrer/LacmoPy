#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 16:10:24 2019

@author: bohrer

this module contains functions which are needed to describe the collision
of warm cloud droplets
"""

import os
import numpy as np
from numba import njit
import math

import constants as c
from microphysics import compute_radius_from_mass_jit
from microphysics import compute_mass_from_radius
from grid import bilinear_weight

#%% PATH CREATION AND FILE HANDLING

def generate_folder_collision(myOS, dV, dt, algorithm, kernel,
                              init, init_pars, no_sims, gen):
    if myOS == "Linux":
        simdata_path = "/home/jdesk/OneDrive/python/sim_data/"
    elif myOS == "MacOS":
        simdata_path = "/Users/bohrer/OneDrive - bwedu/python/sim_data/"
    if init == "SingleSIP":
        kappa = init_pars[0]
        init_pars_string = f"kappa_{kappa}"
        # folder =\
    # f"collision_box_model/kernels/{kernel}/init/{init}/\
# dV_{dV:.2}_dt_{dt:.2}_kappa_{kappa}_no_sims_{no_sims}/"
    elif init == "my_xi_random":
        no_spc = init_pars[0]
        eps = init_pars[1]
        init_pars_string = f"no_spc_{no_spc}_eps_{eps}"
        # folder =\
#     f"collision_box_model/kernels/{kernel}/init/{init}/\
# dV_{dV:.2}_dt_{dt:.2}_no_spc_{no_spc}_eps_{eps}_no_sims_{no_sims}/"
    folder =\
f"collision_box_model_nonint/algo/{algorithm}/kernels/{kernel}/init/{init}/\
dV_{dV:.2}_dt_{dt:.2}_{init_pars_string}_no_sims_{no_sims}/"
    path = simdata_path + folder
    if gen:
        if not os.path.exists(path):
            os.makedirs(path)
    return simdata_path, path


#%% PERMUTATION
# returns a permutation of the list of integers [0,1,2,..,N-1] by Shima method
@njit()
def generate_permutation(N):
    permutation = np.zeros(N, dtype=np.int64)
    for n_next in range(1, N):
        q = np.random.randint(0,n_next+1)
        if q==n_next:
            permutation[n_next] = n_next
        else:
            permutation[n_next] = permutation[q]
            permutation[q] = n_next
    return permutation

#%% COLLECTION, COLLISION, COALESCENCES EFFICIENCIES
# E_collection = E_collision * E_coalescence

### Hall 1980: Collision Efficiency table -> assume E_coalescence = 1.0 f.a. R

# in Hall_E_col:
# row =  collector drop radius (radius of larger drop)
# column = ratio of R_small/R_large "collector ratio"  
Hall_E_col = np.load("Hall_collision_efficiency.npy")
Hall_R_col = np.load("Hall_collector_radius.npy")
Hall_R_col_ratio = np.load("Hall_radius_ratio.npy")
@njit()
def linear_weight(i, weight, f):
    return f[i+1]*weight + f[i]*(1.0-weight)
@njit()
def compute_E_col_Hall(R_i, R_j):
    if R_i <= 0.0 or R_j <= 0.0:
        return 0.0
    if R_i < R_j:
        R_col = R_j
        R_ratio = R_i/R_j
    else:
        R_col = R_i
        R_ratio = R_j/R_i
    if R_col > 300.0:
        return 1.0
    else:
        # NOTE that ind_R is for index of R_collection,
        # which indicates the row of Hall_E_col
        ind_R = int(R_col/10.0)
        ind_ratio = int(R_ratio/0.05)
        if ind_R == Hall_R_col.shape[0]-1:
            if ind_ratio == Hall_R_col_ratio.shape[0]-1:
                return 1.0
            else:
                weight = (R_ratio - ind_ratio * 0.05) / 0.05
                return linear_weight(ind_ratio, weight, Hall_E_col[ind_R])
        elif ind_ratio == Hall_R_col_ratio.shape[0]-1:
            weight = (R_col - ind_R * 10.0) / 10.0
            return linear_weight(ind_R, weight, Hall_E_col[:,ind_ratio])
        else:
            weight_1 = (R_col - ind_R * 10.0) / 10.0
            weight_2 = (R_ratio - ind_ratio * 0.05) / 0.05
#        print(R_col, R_ratio)
#        print(ind_R, ind_ratio, weight_1, weight_2)
            return bilinear_weight(ind_R, ind_ratio,
                                   weight_1, weight_2, Hall_E_col)
        # E_col = bilinear_weight(ind_1, ind_2, weight_1, weight_2, Hall_E_col)

### Kernel "Hall" used by Unterstrasser 2017, he got it from Bott bin-code
# NOTE that there are differences in the Effi table:        
# For R_collec <= 30 mu: major differences: Botts Effis are LARGER
# Bott gives values for 6,8,15,25 mu, which are not incl in Hall orig.
# Bott: if Effi  is Effi_crit > 1.0 for R = R_crit and some R_ratio, then all
# Effi for R > R_crit are Effi = Effi_crit
# for R_ratio >= 0.2, the values are quite close, even for small R, BUT
# at some points, there are deviations...
# in Hall_E_col:
# row =  collector drop radius (radius of larger drop)
# column = ratio of R_small/R_large "collector ratio"  
Hall_Bott_E_col = np.load("Hall_Bott_collision_efficiency.npy")
Hall_Bott_R_col = np.load("Hall_Bott_collector_radius.npy")
Hall_Bott_R_col_ratio = np.load("Hall_Bott_radius_ratio.npy")        
@njit()
def compute_E_col_Hall_Bott(R_i, R_j):
    dR_ipol = 2.0
    dratio = 0.05
    # if R_i <= 0.0 or R_j <= 0.0:
    #     return 0.0
    if R_i < R_j:
        R_col = R_j
        if R_col <= 0.0:
            return Hall_Bott_E_col[0,0]
        else:
            R_ratio = R_j/R_i
        # R_ratio = R_i/R_j
    else:
        R_col = R_i
        if R_col <= 0.0:
            return Hall_Bott_E_col[0,0]
        else:
            R_ratio = R_j/R_i
        # R_ratio = R_j/R_i
    ind_ratio = int(R_ratio/dratio)
    # ind_R is index of R_collection,
    # which indicates the row of Hall_Bott_E_col 
    ind_R = int(R_col/dR_ipol)
    if ind_R == Hall_Bott_R_col.shape[0]-1:
        if ind_ratio == Hall_Bott_R_col_ratio.shape[0]-1:
            return 4.0
        else:
            weight = (R_ratio - ind_ratio * dratio) / dratio
            return linear_weight(ind_ratio, weight, Hall_Bott_E_col[ind_R])
    elif ind_ratio == Hall_Bott_R_col_ratio.shape[0]-1:
        weight = (R_col - ind_R * dR_ipol) / dR_ipol
        return linear_weight(ind_R, weight, Hall_Bott_E_col[:,ind_ratio])
    else:
        weight_1 = (R_col - ind_R * dR_ipol) / dR_ipol
        weight_2 = (R_ratio - ind_ratio * dratio) / dratio
#        print(R_col, R_ratio)
#        print(ind_R, ind_ratio, weight_1, weight_2)
        return bilinear_weight(ind_R, ind_ratio,
                               weight_1, weight_2, Hall_Bott_E_col)
    # E_col = bilinear_weight(ind_1, ind_2, weight_1, weight_2, Hall_E_col)

### TEST: PRINTING FOR EXACT RADII FROM TABLE
# for i1 in range(1,151):
#     diff_Eff = []    
#     strng = ""
#     for j1 in range(21):
#         R1 = Hall_Bott_R_col[i1]
#         R2 = Hall_Bott_R_col_ratio[j1] * R1        
#         dE = Hall_Bott_E_col[i1,j1]-compute_E_col_Hall_Bott(R1,R2)
#         diff_Eff.append(dE)
#         strng += f"{dE:.2} "
#         if abs(dE) > 1.0E-14 or math.isnan(dE):
#             print(i1,j1, R1, R2, Hall_Bott_R_col_ratio[j1], dE)
#     # print(i1,strng)

#%% TERMINAL VELOCITY AS FUNCTION OF RADIUS

from atmosphere import compute_viscosity_air
from atmosphere import compute_surface_tension_water
viscosity_air_NTP = compute_viscosity_air(293.15)
sigma_w_NTP = compute_surface_tension_water(293.15)

### terminal velocity dependent on the droplet radius 
# (under some assumptions for the environment variables (T, p) and density of
# water for the droplets)
# radius R in mu       
#def compute_terminal_velocity(R):
#    k_0 = 4.5 * viscosity_air_NTP / (c.mass_density_water_liquid_NTP * R * R)
fac_Cd_Re_sq = 32.0\
           * (c.mass_density_water_liquid_NTP - c.mass_density_air_dry_NTP)\
           * c.mass_density_air_dry_NTP * c.earth_gravity\
           / (3.0 * viscosity_air_NTP * viscosity_air_NTP)
def compute_Cd_Re_sq(R):
    return fac_Cd_Re_sq * R * R * R
fac_v_from_Re = viscosity_air_NTP/(2.0*c.mass_density_water_liquid_NTP)
# for intermediate 35 mu <= R < 300 mu
def solve_v_iteratively_Long(R):
    if R <= 100:
        Re = 10
    else:
        Re = 100
    R *= 1.0E-6    
    Cd_Re_sq = compute_Cd_Re_sq(R)
    Cd_Re_sq_star = 100.0 * Cd_Re_sq
    
    while abs(Cd_Re_sq_star - Cd_Re_sq) / Cd_Re_sq > 1.0E-3:
        ln_Re = math.log(Re)
        y = -2.44461E-2 * ln_Re - 6.98404E-3 * ln_Re*ln_Re\
            + 8.88634E-4 * ln_Re**3
        bracket_term = 24.0 * (1.0 + 0.10229 * Re**(0.94015 + y))
        Cd_Re_sq_star = Re * bracket_term
        Re = Re - (Cd_Re_sq_star - Cd_Re_sq) / bracket_term
    print(Re)
    return 1.0E2 * fac_v_from_Re * Re / R 

# par[0] belongs to the largest exponential x^(n-1) for par[i], i = 0, .., n 
@njit()
def compute_polynom(par,x):
    res = par[0] * x + par[1]
    for a in par[2:]:
        res = res * x + a
    return res

# import matplotlib.pyplot as plt
# x = np.linspace(-10.0,10.0,100)
# par = (0.1,0.2,0.05,0.04)
# y1 = compute_polynom(par, x)
# y2 = np.polyval(par,x)
# plt.plot(x, (y1-y2)/y1*1E25)
# # plt.plot(x, y2)

# terminal velocity as fct of R by Long 1974b
# "five sections":
# junction points = 15, 35, 300, 800
# NOTE: "assumes p = 1013 mb, T = 20 C, RH = 100 %
v_35 = 13.721114748125
v_300 = 244.10941058000004
fac_vT_1 = 2.0 * c.earth_gravity\
            * (c.mass_density_water_liquid_NTP - c.mass_density_air_dry_NTP) \
            / (9.0 * viscosity_air_NTP)
@njit()
def compute_terminal_velocity_Long(R):
    if R < 15:
        R *= 1.0E-6
        v = fac_vT_1 * R * R * 1.0E2
    elif R < 35:
        R *= 1.0E-4
        v = compute_polynom((-1.917038E13,
                        2.432383E11,
                        -1.263519E9,
                        4.243126E6,
                        -3.417072E3,
                        1.443646), R)
#        v = np.polyval((-1.917038E13,
#                        2.432383E11,
#                        -1.263519E9,
#                        4.243126E6,
#                        -3.417072E3,
#                        1.443646), R)
    elif R < 300:
#        v = solve_v_iteratively_Long(R)
        v = (v_300 - v_35) / (300 - 35) * (R-35) + v_35
    elif R < 800:
        R *=1.0E-4
        v = compute_polynom( (-1.135044E8, 3.971905E7, -5.064069E6,
                         2.559450E5, 2.865716E3, 3.510302E1), R )
#        v = np.polyval( (-1.135044E8, 3.971905E7, -5.064069E6,
#                         2.559450E5, 2.865716E3, 3.510302E1), R )
    else:
        R *= 1.0E-4
        D = compute_polynom( (2.036791E5, -3.815343E4,4.516634E3,
                         -8.020389E1, 1.44274121E1,1.0), (R-0.045))
#        D = np.polyval( (2.036791E5, -3.815343E4,4.516634E3,
#                         -8.020389E1, 1.44274121E1,1.0), (R-0.045))
        v = -(554.5/D) + 921.5 - (4.0E-3 / (R - 0.021))
    return v * 0.01

### From Beard 1976, used also in Bott 1998 and Unterstrasser 2017
# For radii larger the cutoff R_max, we set R = R_max for w_sed
# i.e. the term. velocity is set const. at a max. value for R >= R_max
# these parameters for the polys are checked with Beard 1976
# note, that they dont have units, because Y = polynom(pi,X)
# and Y and X are dimensionless
# in Bott Code: sigma_w = 73.0, rho_w = 1.0, rho_a = 1.225E-3, g = 980.665
## the function was tested vs the tabulated values from Unterstrasser/Beard
## the rel devs. are marginal (my_v_sed > v_sed_Beard) and 
## (rel dev <0.6% for small R and <0.8% for large R)
## and might come from different material
## constants: sigma_w, g, rho_a, rho_w ...
p1_Beard = (-0.318657e1, 0.992696, -0.153193e-2,
          -0.987059e-3,-0.578878e-3,0.855176e-4,-0.327815e-5)[::-1]
p2_Beard = (-0.500015e1,0.523778e1,-0.204914e1,0.475294,-0.542819e-1,
           0.238449e-2)[::-1]
one_sixth = 1.0/6.0
v_max_Beard = 9.11498
@njit()
def compute_terminal_velocity_Beard(R):
    # R in mu = 1E-6 m
    R_0 = 9.5
    R_1 = 535.0
    R_max = 3500.0
    drho = c.mass_density_water_liquid_NTP - c.mass_density_air_dry_NTP
    if R < R_0:
        l0 = 6.62E-2 # mu
        # this is converted for radius instead of diameter
        # i.e. my_C1 = 4*C1_Beard
        C1 = drho*c.earth_gravity / (4.5*viscosity_air_NTP)
        C_sc = 1.0 + 1.255 * l0 / R
        v = C1 * C_sc * R * R * 1.0E-12
    elif R < R_1:
        N_Da = 32.0E-18 * R*R*R * c.mass_density_air_dry_NTP * drho \
               * c.earth_gravity / (3.0 * viscosity_air_NTP*viscosity_air_NTP)
        Y = np.log(N_Da)
        Y = compute_polynom(p1_Beard,Y)
        l0 = 6.62E-2 # mu
        C_sc = 1.0 + 1.255 * l0 / R
        v = viscosity_air_NTP * C_sc * np.exp(Y)\
            / (c.mass_density_air_dry_NTP * R * 2.0E-6)
    elif R < R_max:
        N_Bo = 16.0E-12 * R*R * drho * c.earth_gravity / (3.0 * sigma_w_NTP)
        N_P16 = (sigma_w_NTP * sigma_w_NTP * sigma_w_NTP 
                 * c.mass_density_air_dry_NTP * c.mass_density_air_dry_NTP 
                 / (viscosity_air_NTP**4 * drho * c.earth_gravity))**one_sixth
        Y = np.log(N_Bo * N_P16)
        Y = compute_polynom(p2_Beard,Y)
        v = viscosity_air_NTP * N_P16 * np.exp(Y)\
            / (c.mass_density_air_dry_NTP * R * 2.0E-6)
    else: v = v_max_Beard
    # else:
    #     # IN WORK: precalc v_max form R_max
    #     N_Bo = 16.0E-12 * drho * c.earth_gravity\
    #            / (3.0 * sigma_w_NTP) * R_max * R_max
    #     N_P16 = (sigma_w_NTP * sigma_w_NTP * sigma_w_NTP 
    #              * c.mass_density_air_dry_NTP * c.mass_density_air_dry_NTP 
    #              / (viscosity_air_NTP**4 * drho * c.earth_gravity))**one_sixth
    #     Y = np.log(N_Bo * N_P16)
    #     Y = compute_polynom(p2_Beard,Y)
    #     v = viscosity_air_NTP * N_P16 * np.exp(Y)\
    #         / (c.mass_density_air_dry_NTP * R_max * 2.0E-6)
    return v

# with material const from Bott...
v_max_Beard2 = 9.04929248
@njit()
def compute_terminal_velocity_Beard2(R):
    rho_w = 1.0E3
    rho_a = 1.225
    viscosity_air_NTP = 1.818E-5
    sigma_w_NTP = 73.0E-3
    # R in mu = 1E-6 m
    R_0 = 10.0
    R_1 = 535.0
    R_max = 3500.0
    drho = rho_w-rho_a
    # drho = c.mass_density_water_liquid_NTP - c.mass_density_air_dry_NTP
    if R < R_0:
        l0 = 6.62E-2 # mu
        # this is converted for radius instead of diameter
        # i.e. my_C1 = 4*C1_Beard
        C1 = drho*c.earth_gravity / (4.5*viscosity_air_NTP)
        # C_sc = 1.0 + 1.257 * l0 / R
        C_sc = 1.0 + 1.255 * l0 / R
        v = C1 * C_sc * R * R * 1.0E-12
    elif R < R_1:
        N_Da = 32.0E-18 * R*R*R * rho_a * drho \
               * c.earth_gravity / (3.0 * viscosity_air_NTP*viscosity_air_NTP)
        Y = np.log(N_Da)
        Y = compute_polynom(p1_Beard,Y)
        l0 = 6.62E-2 # mu
        # C_sc = 1.0 + 1.257 * l0 / R
        C_sc = 1.0 + 1.255 * l0 / R
        v = viscosity_air_NTP * C_sc * np.exp(Y)\
            / (rho_a * R * 2.0E-6)
    elif R < R_max:
        N_Bo = 16.0E-12 * R*R * drho * c.earth_gravity / (3.0 * sigma_w_NTP)
        N_P16 = (sigma_w_NTP * sigma_w_NTP * sigma_w_NTP 
                 * rho_a * rho_a 
                 / (viscosity_air_NTP**4 * drho * c.earth_gravity))**one_sixth
        Y = np.log(N_Bo * N_P16)
        Y = compute_polynom(p2_Beard,Y)
        v = viscosity_air_NTP * N_P16 * np.exp(Y)\
            / (rho_a * R * 2.0E-6)
    else: v = v_max_Beard2
    # else:
    #     # IN WORK: precalc v_max form R_max
    #     N_Bo = 16.0E-12 * drho * c.earth_gravity\
    #             / (3.0 * sigma_w_NTP) * R_max * R_max
    #     N_P16 = (sigma_w_NTP * sigma_w_NTP * sigma_w_NTP 
    #               * rho_a * rho_a 
    #               / (viscosity_air_NTP**4 * drho * c.earth_gravity))**one_sixth
    #     Y = np.log(N_Bo * N_P16)
    #     Y = compute_polynom(p2_Beard,Y)
    #     v = viscosity_air_NTP * N_P16 * np.exp(Y)\
    #         / (rho_a * R_max * 2.0E-6)
    return v
# print(compute_terminal_velocity_Beard2(3500.0))
    
###
#v_35 = compute_terminal_velocity_Long(35)
#v_300 = compute_terminal_velocity_Long(300)

# drho = c.mass_density_water_liquid_NTP - c.mass_density_air_dry_NTP
# def compute_N_Da(R):
#     return 32.0E-18 * R*R*R * c.mass_density_air_dry_NTP * drho \
#            * c.earth_gravity / (3.0 * viscosity_air_NTP*viscosity_air_NTP)

# v = []
##R = np.arange(1,35,1)
##R = np.arange(1,1200,1)
# R = np.arange(10,30,1)
# R = np.arange(1,4000,1)
# R = np.arange(0.1,30.0,0.1)
# R = np.arange(534.9,535.1,0.001)
# for R_ in R:
#     # v.append(compute_terminal_velocity_Beard(R_)*100)
#     v.append(compute_terminal_velocity_Beard(R_))

# import matplotlib.pyplot as plt
# plt.plot(R,v)
# plt.plot(2.0*R*1.0E-3,v)
# plt.grid()
# plt.yticks(np.arange(0,10,1))
# plt.plot(R,compute_N_Da(R))


#%% COLLECTION KERNELS
    
# P_ij = dt/dV K_ij
# K_ij = b * (m_i + m_j)
# b = 1.5 m^3/(kg s) (Unterstrasser 2017, p. 1535)
@njit()
def kernel_Golovin(m_i, m_j):
    return 1.5E-18 * (m_i + m_j)

# R_i, R_j in mu
# 1E-12 to get m^3/s for the kernel
@njit()
def kernel_Hydro(R_i, R_j, E_col, v_dif):
    return math.pi * (R_i + R_j)*(R_i + R_j) * E_col * v_dif * 1.0E-12

# Hall 1980
# use hydrodynamic kernel with terminal velocity dependent on R_i
# and tabulated collection efficiencies from the paper
@njit()
def kernel_Hall(m_i, m_j):
    R_i = compute_radius_from_mass_jit(m_i, c.mass_density_water_liquid_NTP)
    R_j = compute_radius_from_mass_jit(m_j, c.mass_density_water_liquid_NTP)
    E_col = compute_E_col_Hall(R_i, R_j)
    dv = compute_terminal_velocity_Beard(R_i)\
         - compute_terminal_velocity_Beard(R_j)
    return kernel_Hydro(R_i, R_j, E_col, abs(dv))

# Bott 1998:
# "For the collection kernel in (26) the same collision efficienciesare 
# are used  as  described  in  Seeßelberg  et  al.  (1996)
# For small droplets  these data are taken from 
# Davis (1972)and Jonas (1972),
# whereas for larger drops the dataset of  Hall  (1980)  has  been  utilized. 
@njit()
def kernel_Bott(m_i, m_j):
    R_i = compute_radius_from_mass_jit(m_i, c.mass_density_water_liquid_NTP)
    R_j = compute_radius_from_mass_jit(m_j, c.mass_density_water_liquid_NTP)
    E_col = compute_E_col_Hall_Bott(R_i, R_j)
    dv = compute_terminal_velocity_Beard(R_i)\
         - compute_terminal_velocity_Beard(R_j)
    return kernel_Hydro(R_i, R_j, E_col, abs(dv))

# Long 1974 
# Kernel (two versions (?) he says: might be approx by ... OR ... (?))
# First:
# R < 50 mu: 9.44E9 * (Vi^2 + Vj^2)
# R > 50 mu: 5.78E3 * (Vi + Vj)
# Second:
# 1.1E10 * Vi^2 (no Vj)
# 6.33E3 Vi     (no Vj)
# here R = radius of LARGER drop, Vi = volume of larger drop in cm^3
# implement first version:
# R_i in mu
# m_i in fg
four_pi_over_three = 4.0/3.0*math.pi
@njit()
def kernel_Long_A(m_i, m_j):
    R_i = compute_radius_from_mass_jit(m_i, c.mass_density_water_liquid_NTP)
    R_j = compute_radius_from_mass_jit(m_j, c.mass_density_water_liquid_NTP)
    # R_max = max(R_i, R_j):
    if R_i <= 50 and R_j <= 50:
        # return 9.44E-15 * (m_i*m_i + m_j*m_j)\
        #        / (c.mass_density_water_liquid_NTP
        #           * c.mass_density_water_liquid_NTP)
        # return 9.44E-19 * (m_i*m_i + m_j*m_j)\
        #        / (c.mass_density_water_liquid_NTP
        #           * c.mass_density_water_liquid_NTP)
        # return 9.44E9*1.0E-24*(m_i*m_i + m_j*m_j)\
        #        / (c.mass_density_water_liquid_NTP
        #           * c.mass_density_water_liquid_NTP)
        return 9.44E9 * 1.0E-24 * 1.0E-6 * four_pi_over_three**2 * (R_i**6 + R_j**6)
    else:
        # return 5.78E-9 * (m_i + m_j) / c.mass_density_water_liquid_NTP
        # return 5.78E-12 * (m_i + m_j) / c.mass_density_water_liquid_NTP
        # return 5.78E3 * 1.0E-12 (m_i + m_j) / c.mass_density_water_liquid_NTP
        return 5.78E3 * 1.0E-12 * 1.0E-6 * four_pi_over_three * (R_i**3 + R_j**3)


# Kernel "Long" as used in Unterstrasser 2017, which he got from Bott
## the Effi function was tested vs tabulated values from Unterstrasser
## the raltive deviations (E1 - E2)/E1 are < 1.0E-13 f.a. radii
@njit()
def compute_E_col_Long_Bott(R_i, R_j):
    R_max = max(R_i, R_j)
    R_min = min(R_i, R_j)
    if R_max <= 50:
        E_col = 4.5E-4 * R_max*R_max \
                * ( 1.0 - 3.0 / ( max(3.0, R_min) + 1.0E-2) )
    else: E_col = 1.0
    return E_col
    
# Kernel "Long" as used in Unterstrasser 2017, which he got from Bott
four_pi_over_three = 4.0/3.0*math.pi
@njit()
def kernel_Long_Bott(m_i, m_j):
    R_i = compute_radius_from_mass_jit(m_i, c.mass_density_water_liquid_NTP)
    R_j = compute_radius_from_mass_jit(m_j, c.mass_density_water_liquid_NTP)
    R_max = max(R_i, R_j)
    R_min = min(R_i, R_j)
    if R_max <= 50:
        E_col = 4.5E-4 * R_max * R_max \
                * ( 1.0 - 3.0 / ( max(3.0, R_min) + 1.0E-2) )
    else: E_col = 1.0
    dv = compute_terminal_velocity_Beard2(R_i)\
         - compute_terminal_velocity_Beard2(R_j)
    # dv = compute_terminal_velocity_Beard(R_i)\
    #      - compute_terminal_velocity_Beard(R_j)
    return math.pi * (R_i + R_j) * (R_i + R_j) * E_col * abs(dv) * 1.0E-12

# Kernel "Long" as used in Unterstrasser 2017, which he got from Bott
four_pi_over_three = 4.0/3.0*math.pi
@njit()
def kernel_Long_Bott_R(R_i, R_j):
    # R_i = compute_radius_from_mass_jit(m_i, c.mass_density_water_liquid_NTP)
    # R_j = compute_radius_from_mass_jit(m_j, c.mass_density_water_liquid_NTP)
    R_max = max(R_i, R_j)
    R_min = min(R_i, R_j)
    if R_max <= 50:
        E_col = 4.5E-4 * R_max * R_max \
                * ( 1.0 - 3.0 / ( max(3.0, R_min) + 1.0E-2) )
    else: E_col = 1.0
    dv = compute_terminal_velocity_Beard2(R_i)\
         - compute_terminal_velocity_Beard2(R_j)
    return math.pi * (R_i + R_j) * (R_i + R_j) * E_col * abs(dv) * 1.0E-12

#m_i = 1E6
# # m_j = 1E7
#R_i = compute_radius_from_mass_jit(m_i, c.mass_density_water_liquid_NTP)
# # R_j = compute_radius_from_mass_jit(m_j, c.mass_density_water_liquid_NTP)
#print(R_i)
# # print(R_j)
#
#kernel_lon = []
#kernel_hal = []
#kernel_gol = []
#
# # print(kernel_Long(m_i,m_j))
#R_range = np.arange(10,1000,1)
#m_range = compute_mass_from_radius(R_range, c.mass_density_water_liquid_NTP)
#for m_j in m_range:
#    kernel_gol.append(kernel_Golovin(m_i, m_j))
#    kernel_lon.append(kernel_Long(m_i, m_j))
#    kernel_hal.append(kernel_Hall(m_i, m_j))
#
#kernel_gol = np.array(kernel_gol)
#kernel_lon = np.array(kernel_lon)
#kernel_hal = np.array(kernel_hal)
#
#import matplotlib.pyplot as plt
#
## R_j = np.arange(10,100,10)
## R_range
#plt.plot(m_range, kernel_gol,"o")
#plt.plot(m_range, kernel_hal,"x")
#plt.plot(m_range, kernel_lon,"d")
#plt.xscale("log")
#plt.yscale("log")
#m_50 = compute_mass_from_radius(50, c.mass_density_water_liquid_NTP)
##plt.vlines(m_50,
##           0.0, kernel_Long(m_i,m_50))

#%% SHIMA algorithm 01 ALPHA
    
# NOTE that if two droplets with the same xi collide,
# two droplets are created with [xi/2] and xi - [xi/2]
# and the same masses

@njit()
def collision_step_Golovin(xis, masses, dV, dt):
    # xis is a 1D array of multiplicities
    # elements of xis can be 0, indicating vanished droplets
    # look for the indices, which are not zero
    indices = np.nonzero(xis)[0]
    # N = no_spct ("number of super particles per cell total")
    # no_spct = len(indices)
    no_spct = indices.shape[0]
    # no_spct = 5940
    # Let I = [0,1,2,3,...,N-1] be the list of all EXISTING particle indices
    
    ### 1. make a random permutation of I by
    permutation = generate_permutation(no_spct)
    # print("permutation len")
    # print(len(permutation))
    
    ### 2. make [no_spct/2] candidate pairs [(i1,j1),(i2,j2),(),.]
    no_pairs = no_spct//2
    cand_i = permutation[0:no_pairs]
    cand_j = permutation[no_pairs:2*no_pairs]
    # map onto the original xis-list, to get the indices for that list
    
    cand_i = indices[cand_i]
    cand_j = indices[cand_j]
    
    # print("cand_i len")
    # print(len(cand_i))
    # print("cand_j len")
    # print(len(cand_j))
    
    ### 3. for each pair (i,j) generate uniform float random number from [0,1)
    rnd01 = np.random.rand(no_pairs)
    # print("len(rnd01)")
    # print(len(rnd01))
    for pair_n in range(no_pairs):
        i = cand_i[pair_n]
        j = cand_j[pair_n]
        # i_done.append(i)
        # j_done.append(j)
        pair_kernel =\
            kernel_Golovin(masses[i], masses[j])
        if xis[i] >= xis[j]:
            xi_max = xis[i]
            ind_max = i
            xi_min = xis[j]
            ind_min = j
        else:
            xi_max = xis[j]
            ind_max = j
            xi_min = xis[i]
            ind_min = i
            
        # xi_max = max(xis[i], xis[j])
        # now we need p = P_ij,s N_s*(N_s-1)/2 / (no_pairs),
        # where N_s = no_spct and
        # P_ij,s = max(xis_i, xis_j) P_ij
        # P_ij = K_ij dt/dV
        pair_prob = pair_kernel * dt/dV * xi_max\
                    * 0.5 * no_spct * (no_spct - 1) / no_pairs
        # then evaluate
        # beta
        # = [p] + 1; if rnd01 < p - [p]
        # = [p]    ; if rnd01 >= p - [p]
        pair_prob_int = int(pair_prob)
        if rnd01[pair_n] < pair_prob - pair_prob_int:
            g = pair_prob_int + 1
        else: g = pair_prob_int
        # print(pair_n, pair_prob, pair_prob_int, rnd01[pair_n], g)
        if g > 0:
            # print("collision for pair", i, j)
            # xi_min = min(xis[i], xis[j])
            g = min(g,int(xi_max/xi_min))
            if xi_max - g*xi_min > 0:
                # print("case 1")
                xis[ind_max] -= g * xi_min
                masses[ind_min] += g * masses[ind_max]
            else:
                # print("case 2")
                xis[ind_max] = int(0.5 * xi_min)
                xis[ind_min] -= int(0.5 * xi_min)
                masses[ind_min] += g * masses[ind_max]
                masses[ind_max] = masses[ind_min]

### Coll step for Hall Kernel
@njit()
def collision_step_Hall(xis, masses, dV, dt):
    # xis is a 1D array of multiplicities
    # elements of xis can be 0, indicating vanished droplets
    # look for the indices, which are not zero
    indices = np.nonzero(xis)[0]
    # N = no_spct ("number of super particles per cell total")
    # no_spct = len(indices)
    no_spct = indices.shape[0]
    # no_spct = 5940
    # Let I = [0,1,2,3,...,N-1] be the list of all EXISTING particle indices
    
    ### 1. make a random permutation of I by
    permutation = generate_permutation(no_spct)
    # print("permutation len")
    # print(len(permutation))
    
    ### 2. make [no_spct/2] candidate pairs [(i1,j1),(i2,j2),(),.]
    no_pairs = no_spct//2
    cand_i = permutation[0:no_pairs]
    cand_j = permutation[no_pairs:2*no_pairs]
    # map onto the original xis-list, to get the indices for that list
    
    cand_i = indices[cand_i]
    cand_j = indices[cand_j]
    
    # print("cand_i len")
    # print(len(cand_i))
    # print("cand_j len")
    # print(len(cand_j))
    
    ### 3. for each pair (i,j) generate uniform float random number from [0,1)
    rnd01 = np.random.rand(no_pairs)
    # print("len(rnd01)")
    # print(len(rnd01))
    for pair_n in range(no_pairs):
        i = cand_i[pair_n]
        j = cand_j[pair_n]
        # i_done.append(i)
        # j_done.append(j)
        pair_kernel =\
            kernel_Hall(masses[i], masses[j])
        if xis[i] >= xis[j]:
            xi_max = xis[i]
            ind_max = i
            xi_min = xis[j]
            ind_min = j
        else:
            xi_max = xis[j]
            ind_max = j
            xi_min = xis[i]
            ind_min = i
            
        # xi_max = max(xis[i], xis[j])
        # now we need p = P_ij,s N_s*(N_s-1)/2 / (no_pairs),
        # where N_s = no_spct and
        # P_ij,s = max(xis_i, xis_j) P_ij
        # P_ij = K_ij dt/dV
        pair_prob = pair_kernel * dt/dV * xi_max\
                    * 0.5 * no_spct * (no_spct - 1) / no_pairs
        # then evaluate
        # beta
        # = [p] + 1; if rnd01 < p - [p]
        # = [p]    ; if rnd01 >= p - [p]
        pair_prob_int = int(pair_prob)
        if rnd01[pair_n] < pair_prob - pair_prob_int:
            g = pair_prob_int + 1
        else: g = pair_prob_int
        # print(pair_n, pair_prob, pair_prob_int, rnd01[pair_n], g)
        if g > 0:
            # print("collision for pair", i, j)
            # xi_min = min(xis[i], xis[j])
            g = min(g,int(xi_max/xi_min))
            if xi_max - g*xi_min > 0:
                # print("case 1")
                xis[ind_max] -= g * xi_min
                masses[ind_min] += g * masses[ind_max]
            else:
                # print("case 2")
                xis[ind_max] = int(0.5 * xi_min)
                xis[ind_min] -= int(0.5 * xi_min)
                masses[ind_min] += g * masses[ind_max]
                masses[ind_max] = masses[ind_min]

### Coll step for Bott Kernel
@njit()
def collision_step_Bott(xis, masses, dV, dt):
    # xis is a 1D array of multiplicities
    # elements of xis can be 0, indicating vanished droplets
    # look for the indices, which are not zero
    indices = np.nonzero(xis)[0]
    # N = no_spct ("number of super particles per cell total")
    # no_spct = len(indices)
    no_spct = indices.shape[0]
    # no_spct = 5940
    # Let I = [0,1,2,3,...,N-1] be the list of all EXISTING particle indices
    
    ### 1. make a random permutation of I by
    permutation = generate_permutation(no_spct)
    # print("permutation len")
    # print(len(permutation))
    
    ### 2. make [no_spct/2] candidate pairs [(i1,j1),(i2,j2),(),.]
    no_pairs = no_spct//2
    cand_i = permutation[0:no_pairs]
    cand_j = permutation[no_pairs:2*no_pairs]
    # map onto the original xis-list, to get the indices for that list
    
    cand_i = indices[cand_i]
    cand_j = indices[cand_j]
    
    # print("cand_i len")
    # print(len(cand_i))
    # print("cand_j len")
    # print(len(cand_j))
    
    ### 3. for each pair (i,j) generate uniform float random number from [0,1)
    rnd01 = np.random.rand(no_pairs)
    # print("len(rnd01)")
    # print(len(rnd01))
    for pair_n in range(no_pairs):
        i = cand_i[pair_n]
        j = cand_j[pair_n]
        # i_done.append(i)
        # j_done.append(j)
        pair_kernel =\
            kernel_Bott(masses[i], masses[j])
        if xis[i] >= xis[j]:
            xi_max = xis[i]
            ind_max = i
            xi_min = xis[j]
            ind_min = j
        else:
            xi_max = xis[j]
            ind_max = j
            xi_min = xis[i]
            ind_min = i
            
        # xi_max = max(xis[i], xis[j])
        # now we need p = P_ij,s N_s*(N_s-1)/2 / (no_pairs),
        # where N_s = no_spct and
        # P_ij,s = max(xis_i, xis_j) P_ij
        # P_ij = K_ij dt/dV
        pair_prob = pair_kernel * dt/dV * xi_max\
                    * 0.5 * no_spct * (no_spct - 1) / no_pairs
        # then evaluate
        # beta
        # = [p] + 1; if rnd01 < p - [p]
        # = [p]    ; if rnd01 >= p - [p]
        pair_prob_int = int(pair_prob)
        if rnd01[pair_n] < pair_prob - pair_prob_int:
            g = pair_prob_int + 1
        else: g = pair_prob_int
        # print(pair_n, pair_prob, pair_prob_int, rnd01[pair_n], g)
        if g > 0:
            # print("collision for pair", i, j)
            # xi_min = min(xis[i], xis[j])
            g = min(g,int(xi_max/xi_min))
            if xi_max - g*xi_min > 0:
                # print("case 1")
                xis[ind_max] -= g * xi_min
                masses[ind_min] += g * masses[ind_max]
            else:
                # print("case 2")
                xis[ind_max] = int(0.5 * xi_min)
                xis[ind_min] -= int(0.5 * xi_min)
                masses[ind_min] += g * masses[ind_max]
                masses[ind_max] = masses[ind_min]

@njit()
def collision_step_Long(xis, masses, dV, dt):
    # xis is a 1D array of multiplicities
    # elements of xis can be 0, indicating vanished droplets
    # look for the indices, which are not zero
    indices = np.nonzero(xis)[0]
    # N = no_spct ("number of super particles per cell total")
    # no_spct = len(indices)
    no_spct = indices.shape[0]
    # no_spct = 5940
    # Let I = [0,1,2,3,...,N-1] be the list of all EXISTING particle indices
    
    ### 1. make a random permutation of I by
    permutation = generate_permutation(no_spct)
    # print("permutation len")
    # print(len(permutation))
    
    ### 2. make [no_spct/2] candidate pairs [(i1,j1),(i2,j2),(),.]
    no_pairs = no_spct//2
    cand_i = permutation[0:no_pairs]
    cand_j = permutation[no_pairs:2*no_pairs]
    # map onto the original xis-list, to get the indices for that list
    
    cand_i = indices[cand_i]
    cand_j = indices[cand_j]
    
    # print("cand_i len")
    # print(len(cand_i))
    # print("cand_j len")
    # print(len(cand_j))
    
    ### 3. for each pair (i,j) generate uniform float random number from [0,1)
    rnd01 = np.random.rand(no_pairs)
    # print("len(rnd01)")
    # print(len(rnd01))
    for pair_n in range(no_pairs):
        i = cand_i[pair_n]
        j = cand_j[pair_n]
        # i_done.append(i)
        # j_done.append(j)
        pair_kernel =\
            kernel_Long_A(masses[i], masses[j])
        if xis[i] >= xis[j]:
            xi_max = xis[i]
            ind_max = i
            xi_min = xis[j]
            ind_min = j
        else:
            xi_max = xis[j]
            ind_max = j
            xi_min = xis[i]
            ind_min = i
            
        # xi_max = max(xis[i], xis[j])
        # now we need p = P_ij,s N_s*(N_s-1)/2 / (no_pairs),
        # where N_s = no_spct and
        # P_ij,s = max(xis_i, xis_j) P_ij
        # P_ij = K_ij dt/dV
        pair_prob = pair_kernel * dt/dV * xi_max\
                    * 0.5 * no_spct * (no_spct - 1) / no_pairs
        # then evaluate
        # beta
        # = [p] + 1; if rnd01 < p - [p]
        # = [p]    ; if rnd01 >= p - [p]
        pair_prob_int = int(pair_prob)
        if rnd01[pair_n] < pair_prob - pair_prob_int:
            g = pair_prob_int + 1
        else: g = pair_prob_int
        # print(pair_n, pair_prob, pair_prob_int, rnd01[pair_n], g)
        if g > 0:
            # print("collision for pair", i, j)
            # xi_min = min(xis[i], xis[j])
            g = min(g,int(xi_max/xi_min))
            if xi_max - g*xi_min > 0:
                # print("case 1")
                xis[ind_max] -= g * xi_min
                masses[ind_min] += g * masses[ind_max]
            else:
                # print("case 2")
                xis[ind_max] = int(0.5 * xi_min)
                xis[ind_min] -= int(0.5 * xi_min)
                masses[ind_min] += g * masses[ind_max]
                masses[ind_max] = masses[ind_min]

@njit()
def collision_step_Long_Bott(xis, masses, dV, dt):
    # xis is a 1D array of multiplicities
    # elements of xis can be 0, indicating vanished droplets
    # look for the indices, which are not zero
    indices = np.nonzero(xis)[0]
    # N = no_spct ("number of super particles per cell total")
    # no_spct = len(indices)
    no_spct = indices.shape[0]
    # no_spct = 5940
    # Let I = [0,1,2,3,...,N-1] be the list of all EXISTING particle indices
    
    ### 1. make a random permutation of I by
    permutation = generate_permutation(no_spct)
    # print("permutation len")
    # print(len(permutation))
    
    ### 2. make [no_spct/2] candidate pairs [(i1,j1),(i2,j2),(),.]
    no_pairs = no_spct//2
    cand_i = permutation[0:no_pairs]
    cand_j = permutation[no_pairs:2*no_pairs]
    # map onto the original xis-list, to get the indices for that list
    
    cand_i = indices[cand_i]
    cand_j = indices[cand_j]
    
    # print("cand_i len")
    # print(len(cand_i))
    # print("cand_j len")
    # print(len(cand_j))
    
    ### 3. for each pair (i,j) generate uniform float random number from [0,1)
    rnd01 = np.random.rand(no_pairs)
    # print("len(rnd01)")
    # print(len(rnd01))
    for pair_n in range(no_pairs):
        i = cand_i[pair_n]
        j = cand_j[pair_n]
        # i_done.append(i)
        # j_done.append(j)
        pair_kernel =\
            kernel_Long_Bott(masses[i], masses[j])
        if xis[i] >= xis[j]:
            xi_max = xis[i]
            ind_max = i
            xi_min = xis[j]
            ind_min = j
        else:
            xi_max = xis[j]
            ind_max = j
            xi_min = xis[i]
            ind_min = i
        # xi_max = max(xis[i], xis[j])
        
        # now we need p = P_ij,s N_s*(N_s-1)/2 / (no_pairs),
        # where N_s = no_spct and
        # P_ij,s = max(xis_i, xis_j) P_ij
        # P_ij = K_ij dt/dV
        pair_prob = pair_kernel * dt/dV * xi_max\
                    * 0.5 * no_spct * (no_spct - 1) / no_pairs
        # then evaluate
        # beta
        # = [p] + 1; if rnd01 < p - [p]
        # = [p]    ; if rnd01 >= p - [p]
        pair_prob_int = int(pair_prob)
        if rnd01[pair_n] < pair_prob - pair_prob_int:
            g = pair_prob_int + 1
        else: g = pair_prob_int
        # print(pair_n, pair_prob, pair_prob_int, rnd01[pair_n], g)
        if g > 0:
            # print("collision for pair", i, j)
            # xi_min = min(xis[i], xis[j])
            g = min(g,int(xi_max/xi_min))
            if xi_max - g*xi_min > 0:
                # print("case 1")
                xis[ind_max] -= g * xi_min
                masses[ind_min] += g * masses[ind_max]
            else:
                # print("case 2")
                xis[ind_max] = int(0.5 * xi_min)
                xis[ind_min] -= int(0.5 * xi_min)
                masses[ind_min] += g * masses[ind_max]
                masses[ind_max] = masses[ind_min]

#%% AON UNTERSTRASSER
# AON collision model from Unterstrasser 2017, p. 1534
# in each step, check each (i-j) combination for a possible coalescence event
# NOTE that I use xi instead of nu (in Unterstrasser), i.e. nu_k = xi_k etc.
@njit()
def collision_step_Long_Bott_Unt(xis, masses, dV, dt):
    # xis = array of multiplicities
    ind = np.nonzero(xis)[0]
    # N = no_spct ("number of super particles per cell total")
    # no_spct = len(indices)
    no_spct = ind.shape[0]
    # rnd = np.random.rand(no_spct*(no_spct-1)/2)
    for i_SIP in range(1, no_spct):
        i = ind[i_SIP]
        # for j_SIP in range(i_SIP):
        j_SIP = 0
        while xis[i] > 0 and j_SIP < i_SIP:
            j = ind[j_SIP]
            if xis[j] > 0:
                if xis[i] >= xis[j]:
                    xi_max = xis[i]
                    ind_max = i
                    xi_min = xis[j]
                    ind_min = j
                else:
                    xi_max = xis[j]
                    ind_max = j
                    xi_min = xis[i]
                    ind_min = i            
                # compute xi_k according to xi_k = dt/dV * xi_i * xi_j * K_ij
                p_crit = dt/dV * xi_max\
                         * kernel_Long_Bott(masses[i], masses[j])
                xi_k = p_crit * xi_min
                # xi_k = dt/dV * xi_min * xi_max\
                #        * kernel_Long_Bott(masses[i], masses[j])
                # p_crit = xi_k / xi_min
                if p_crit > 1.0:
                    # multiple collisions: one droplet of SIP i collects
                    # more than one droplet of SIP j:
                    # SIP (i = ind_min) collects
                    # xi_k droplets of SIP (j = ind_max)
                    # and distributes them on xi_i droplets:
                    # now, we need to decide, what xi_k should to be:
                    # integer and xi_k <= xi_max
                    # p_crit = p_crit - int(p_crit)
                    
                    # NON INT!!!
                    # if np.random.rand() < p_crit:
                    #     xi_k = int(xi_k) + xi_min
                    # else: xi_k = int(xi_k)
                    
                    xi_k = int(xi_k)
                    # if xi_k >= xi_max:
                    #     xi_k = xi_max
                    # xi_k = min(xi_k, xi_max)
                    masses[ind_min] =\
                        (xi_min*masses[ind_min] + xi_k*masses[ind_max])\
                        / xi_min
                    xis[ind_max] -= xi_k
                else:
                    if np.random.rand() < p_crit:
                        if xi_max - xi_min < 0.1:
                            # NON INT!!
                            # xis[ind_max] = int(xi_min/2)
                            # xis[ind_min] = xi_min - int(xi_min/2)
                            xis[ind_min] = 0.5 * xi_max
                            xis[ind_max] = xis[ind_min]
                            masses[ind_min] =\
                                (xi_min*masses[ind_min]
                                 + xi_max*masses[ind_max]) / xi_max
                            masses[ind_max] = masses[ind_min]
                        else:
                            masses[ind_min] += masses[ind_max]
                            xis[ind_max] -= xi_min
                # if xis[ind_max] == 0 and xi_min > 1:
                #     xis[ind_max] = int(xi_min/2)
                #     xis[ind_min] = xi_min - int(xi_min/2)
                #     masses[ind_max] = masses[ind_min]
            j_SIP += 1

@njit()
def collision_step_Golovin_Unt(xis, masses, dV, dt):
    # xis = array of multiplicities
    ind = np.nonzero(xis)[0][::-1]
    # N = no_spct ("number of super particles per cell total")
    # no_spct = len(indices)
    no_spct = ind.shape[0]
    # rnd = np.random.rand(no_spct*(no_spct-1)/2)
    for i_SIP in range(1, no_spct):
        i = ind[i_SIP]
        # for j_SIP in range(i_SIP):
        j_SIP = 0
        while xis[i] > 0 and j_SIP < i_SIP:
            j = ind[j_SIP]
            if xis[j] > 0:
                if xis[i] >= xis[j]:
                    xi_max = xis[i]
                    ind_max = i
                    xi_min = xis[j]
                    ind_min = j
                else:
                    xi_max = xis[j]
                    ind_max = j
                    xi_min = xis[i]
                    ind_min = i            
                # compute xi_k according to xi_k = dt/dV * xi_i * xi_j * K_ij
                p_crit = dt/dV * xi_max\
                         * kernel_Golovin(masses[i], masses[j])
                xi_k = p_crit * xi_min
                # xi_k = dt/dV * xi_min * xi_max\
                #        * kernel_Long_Bott(masses[i], masses[j])
                # p_crit = xi_k / xi_min
                if p_crit > 1.0:
                    # multiple collisions: one droplet of SIP i collects
                    # more than one droplet of SIP j:
                    # SIP (i = ind_min) collects
                    # xi_k droplets of SIP (j = ind_max)
                    # and distributes them on xi_i droplets:
                    # now, we need to decide, what xi_k should to be:
                    # integer and xi_k <= xi_max
                    p_crit = p_crit - int(p_crit)
                    
                    if np.random.rand() < p_crit:
                        xi_k = int(xi_k) + xi_min
                    else: xi_k = int(xi_k)
                    # xi_k = int(xi_k)
                    
                    xi_k = min(xi_k, xi_max)
                    masses[ind_min] =\
                        (xi_min*masses[ind_min] + xi_k*masses[ind_max])\
                        / xi_min
                    xis[ind_max] -= xi_k
                else:
                    if np.random.rand() < p_crit:
                        masses[ind_min] += masses[ind_max]
                        xis[ind_max] -= xi_min
                if xis[ind_max] == 0 and xi_min > 1:
                    xis[ind_max] = int(xi_min/2)
                    xis[ind_min] = xi_min - int(xi_min/2)
                    masses[ind_max] = masses[ind_min]
            j_SIP += 1





@njit()
def collision_step_Golovin_Unt2(xis, masses, dV, dt):
    ind = np.nonzero(xis)[0]
    
    no_spc = ind.shape[0]
    
    for i_ in range(1,no_spc):
        i = ind[i_]
        for j_ in range (i_):
            j = ind[j_]
            if xis[i] < xis[j]:
                xi_min = xis[i]
                xi_max = xis[j]
                i_min = i
                i_max = j
            else:
                xi_min = xis[j]
                xi_max = xis[i]
                i_min = j
                i_max = i
            xi_k = xi_max * xi_min * kernel_Golovin(masses[i], masses[j])\
                   * dt / dV
            p_crit = xi_k / xi_min
            if p_crit > 1:
                masses[i_min] = (xi_min*masses[i_min] + xi_k*masses[i_max])\
                                / xi_min
                xis[i_max] -= xi_k
            
            else:
                rnd = np.random.rand()
                if p_crit > rnd:
                    if (xi_max-xi_min)/xi_max < 1.0E-5:
                        b = (0.25 + 0.5 * rnd)
                        xi_avg = 0.5 * (xi_min + xi_max)
                        xis[i] = b * xi_avg
                        xis[j] = (xi_min*masses[i_min] + xi_max*masses[i_max])\
                                 / (masses[i_min]+masses[i_max])\
                                 - b*xi_avg
                        masses[i] = masses[i] + masses[j]
                        masses[j] = masses[i]
                    else:
                        masses[i_min] += masses[i_max]
                        xis[i_max] -= xi_min

@njit()
def collision_step_Long_Bott_Unt2(xis, masses, dV, dt):
    ind = np.nonzero(xis)[0]
    
    no_spc = ind.shape[0]
    
    for i_ in range(1,no_spc):
        i = ind[i_]
        for j_ in range (i_):
            j = ind[j_]
            if xis[i] < xis[j]:
                xi_min = xis[i]
                xi_max = xis[j]
                i_min = i
                i_max = j
            else:
                xi_min = xis[j]
                xi_max = xis[i]
                i_min = j
                i_max = i
            xi_k = xi_max * xi_min * kernel_Long_Bott(masses[i], masses[j])\
                   * dt / dV
            p_crit = xi_k / xi_min
            if p_crit > 1:
                masses[i_min] = (xi_min*masses[i_min] + xi_k*masses[i_max])\
                                / xi_min
                xis[i_max] -= xi_k
            
            else:
                rnd = np.random.rand()
                if p_crit > rnd:
                    if (xi_max-xi_min)/xi_max < 1.0E-5:
                        b = (0.25 + 0.5 * rnd)
                        xi_avg = 0.5 * (xi_min + xi_max)
                        xis[i] = b * xi_avg
                        xis[j] = (xi_min*masses[i_min] + xi_max*masses[i_max])\
                                 / (masses[i_min]+masses[i_max])\
                                 - b*xi_avg
                        masses[i] = masses[i] + masses[j]
                        masses[j] = masses[i]
                    else:
                        masses[i_min] += masses[i_max]
                        xis[i_max] -= xi_min
            



#%% WAITING TIME ALGORITHM

# coalescence events with wating time distribution:
# orig. by Gillespie 1972, 1975 for water droplets (direct = all weights = 1)
# "Efficient algorithm" by Goodson and Kraft 2002 (also direct, no multiplic.)
# with "stochastic weights": Patterson (WIAS) 2011, 2012
# IDEA: (Gillespie 1975 J. atm. sci.)
# joint PDF: P(tau,i,j) = prob, at time t, that the next coalescence will
# occur in the time interval (t+tau, t+tau+dtau)
# AND will be between particles "i" and "j":
# P(\tau,i,j) = C_ij exp [ -\sum_{k=1}^{N-1} \sum_{l=k+1}^N C_kl \tau ]
# We have int_0^inf dtau \sum_{i=1}^{N-1} \sum_{j=i+1}^N P(tau,i,j) = 1
# i.e. a discrete prob. in (i,j) and a PDF in tau
# he defines: C_kl dtau = prob, that particles k and l will coalesce in
# the next infinit. time interval dtau
# this is equivalent to
# the kernel K_kl divided by the cell volume dV C_kl = K_kl / dV
# using the "Full-conditional method" from Gillespie 1975, p. 1981, sec 4a.
# (no majorant or other techniques):
# P(tau,i,j) = P1(tau) * P2(i|tau) * P3(j|tau,i)
# C_i = sum_{j=i+1}^N C_ij (i = 1,...,N-1)
# C_0 = sum_{i=1}^{N-1} C_i
# 1. choose waiting time tau from PDF
# P1(tau) = C_0 exp(-C_0 tau)
# 2. generate "i" from discrete PDF P2(i):
# P2(i) = C_i/C_0
# 3. generate "j" from discrete PDF P3(j|tau,i) = P3(j|i):
# P3(j|i) = C_ij/C_i
@njit()
def collision_step_tau_Long_Bott(xis, masses, dV):
    indices = np.nonzero(xis)[0]
    # N = no_spct ("number of super particles per cell total")
    # no_spct = len(indices)
    no_spct = indices.shape[0]
    
    dV_inv = 1.0/dV
    
    # 1. choose waiting time tau from PDF P1(tau) = C_0 exp(-C_0 tau)
    # on the way: store all C_i = sum_{j=i+1}^N C_ij (i = 1,...,N-1)
    # the number of elements in this list is N-1, if N = no_spct
#    C_is = np.zeros(no_spct-1, dtype=np.float64)
    C = np.zeros((no_spct-1,no_spct-1), dtype=np.float64)
    
    rnd = np.random.rand(3)
    
    # this does not work, we need C_0 first
    # C_ijs = np.zeros(no_spct, dtype=np.float64) # the 0th index is not used
    for i_ in range(no_spct-1):
        i = indices[i_]
        # C_i = 0.0
        for j_ in range(i+1, no_spct):
            j = indices[j_]
            C[i_,j_] = max(xis[i], xis[j]) \
                       * kernel_Long_Bott(masses[i], masses[j])
#            C_is[i_] += max(xis[i], xis[j]) \
#                       * kernel_Long_Bott(masses[i], masses[j]) / dV
#    C_0 = C_is.sum()
    C *= dV_inv
    C_0 = C.sum()
#    rnd01 = np.random.rand()
    tau = -1.0/C_0 * np.log(1 - rnd[0])
    
    # 2. get the first index of the tuple (i,j)
#    rnd02 = np.random.rand()
    rnd[1] *= C_0
#    C_sum = C_is[0]
    C_sum = C[0,:].sum()
    i_ = 0
    while C_sum < rnd[1]:
        i_ += 1
#        C_sum += C_is[i_]
        C_sum += C[i_,:].sum()
    i = indices[i_]
    
    # 3. get the second index of the tuple (i,j)
#    C_i = C_is[i_]
    C_i = C[i_,:].sum()
#    rnd03 = np.random.rand()
    rnd[2] *= C_i
    j_ = i_+1
#    if i != 0: j = 0
#    else: j = 1
    j = indices[j_]
    
#    C_sum = max(xis[i], xis[j]) \
#                       * kernel_Long_Bott(masses[i], masses[j]) / dV
    C_sum = C[i_,j_]
    
    while C_sum < rnd[2]:
        j_ += 1
        j = indices[j_]
#        if j != i: 
#        C_sum += max(xis[i], xis[j]) \
#                     * kernel_Long_Bott(masses[i], masses[j]) / dV
        C_sum += C[i_,j_]
        
    # now, we have found tau, i, and j
    # collision takes place after the AON algorithm
    if xis[i] < xis[j]:
        xi_min = xis[i]
#        xi_max = xis[j]
        i_min = i
        i_max = j
    else:
        xi_min = xis[j]
#        xi_max = xis[i]
        i_min = j
        i_max = i
    masses[i_min] += masses[i_max]
    xis[i_max] -= xi_min
    
    return tau

@njit()
def collision_step_tau_Golovin(xis, masses, dV):
    indices = np.nonzero(xis)[0]
    # N = no_spct ("number of super particles per cell total")
    # no_spct = len(indices)
    no_spct = indices.shape[0]
    
    # 1. choose waiting time tau from PDF P1(tau) = C_0 exp(-C_0 tau)
    # on the way: store all C_i = sum_{j=i+1}^N C_ij (i = 1,...,N-1)
    # the number of elements in this list is N-1, if N = no_spct
    C_is = np.zeros(no_spct-1, dtype=np.float64)
    
    rnd = np.random.rand(3)
    
    # this does not work, we need C_0 first
    # C_ijs = np.zeros(no_spct, dtype=np.float64) # the 0th index is not used
    for i_ in range(no_spct-1):
        i = indices[i_]
        # C_i = 0.0
        for j_ in range(i+1, no_spct):
            j = indices[j_]
            C_is[i_] += max(xis[i], xis[j]) \
                       * kernel_Golovin(masses[i], masses[j]) / dV
    C_0 = C_is.sum()
#    rnd01 = np.random.rand()
    tau = -1.0/C_0 * np.log(1 - rnd[0])
    
    # 2. get the first index of the tuple (i,j)
#    rnd02 = np.random.rand()
    rnd[1] *= C_0
    C_sum = C_is[0]
    i_ = 0
    while C_sum < rnd[1]:
        i_ += 1
        C_sum += C_is[i_]
    i = indices[i_]
    
    # 3. get the second index of the tuple (i,j)
    C_i = C_is[i_]
#    rnd03 = np.random.rand()
    rnd[2] *= C_i
    j_ = i_+1
#    if i != 0: j = 0
#    else: j = 1
    j = indices[j_]
    
    C_sum = max(xis[i], xis[j]) \
                       * kernel_Golovin(masses[i], masses[j]) / dV
    
    while C_sum < rnd[2]:
        j_ += 1
        j = indices[j_]
#        if j != i: 
        C_sum += max(xis[i], xis[j]) \
                     * kernel_Golovin(masses[i], masses[j]) / dV
    
    # now, we have found tau, i, and j
    # collision takes place after the AON algorithm
    if xis[i] < xis[j]:
        xi_min = xis[i]
#        xi_max = xis[j]
        i_min = i
        i_max = j
    else:
        xi_min = xis[j]
#        xi_max = xis[i]
        i_min = j
        i_max = i
    masses[i_min] += masses[i_max]
    xis[i_max] -= xi_min
    
    return tau




#%% SIMULATE COLLISIONS

def simulate_collisions_np(xis, masses, dV, dt, t_end, dt_store,
                           algorithm="Shima", kernel="Golovin"):
        
    store_every = int(math.ceil(dt_store/dt))                                      
    times = np.zeros(1+int(t_end/(store_every * dt)), dtype = np.float64)
    concentrations = np.zeros(1+int( math.ceil(t_end / (store_every * dt)) ),
                              dtype = np.float64)
    masses_vs_time = np.zeros((1+int(t_end/(store_every * dt)), len(masses)),
                              dtype = np.float64)
    
    collision_step = collision_step_Long_Bott
    
    if algorithm == "AON_Unt" or algorithm == "Shima":        
        xis_vs_time = np.zeros((1+int(t_end/(store_every * dt)), len(masses)),
                                  dtype = np.float64)
        if algorithm == "Shima":
            xis_vs_time = np.zeros((1+int(t_end/(store_every * dt)), len(masses)),
                                          dtype = np.int64)
#        elif algorithm == "AON_Unt":
        if algorithm == "Shima":
            if kernel == "Long": collision_step = collision_step_Long
            elif kernel == "Long_Bott": collision_step = collision_step_Long_Bott
            elif kernel == "Golovin": collision_step = collision_step_Golovin
            elif kernel == "Hall": collision_step = collision_step_Hall
            elif kernel == "Bott": collision_step = collision_step_Bott
        elif algorithm == "AON_Unt":
            if kernel == "Long_Bott":
                collision_step = collision_step_Long_Bott_Unt2
            elif kernel == "Golovin": collision_step =\
                                          collision_step_Golovin_Unt2
        cnt_c = 0 
    
        for cnt, t in enumerate(np.arange(0.0,t_end,dt)):
            # print("sim at", t)
            if cnt%store_every == 0:
                times[cnt_c] = t
                concentrations[cnt_c] = np.sum(xis)/dV
                masses_vs_time[cnt_c] = masses
                xis_vs_time[cnt_c] = xis
                cnt_c += 1
            collision_step(xis, masses, dV, dt)  
            
        times[cnt_c] = t_end
        concentrations[cnt_c] = np.sum(xis)/dV
        masses_vs_time[cnt_c] = masses
        xis_vs_time[cnt_c] = xis
        taus = np.zeros(1)
        print()
        print("ALGO = SHIMA OR AON_UNT")
#        print("ALGO=", algorithm)
        print()
        return times, concentrations, masses_vs_time, xis_vs_time, taus
    
    elif algorithm == "waiting_time":
        print()
        print("ALGO = WAITING TIME")
        print()
        xis_vs_time = np.zeros((1+int(t_end/(store_every * dt)), len(masses)),
                                  dtype = np.float64)
        if kernel == "Golovin": collision_step =\
                                    collision_step_tau_Golovin
        elif kernel == "Long_Bott": collision_step =\
                                        collision_step_tau_Long_Bott
        taus = []
        t = 0.0
        cnt_c = 0
        times[cnt_c] = t
        concentrations[cnt_c] = np.sum(xis)/dV
        masses_vs_time[cnt_c] = masses
        xis_vs_time[cnt_c] = xis
        cnt_c = 1
        while t < t_end:
            if t - cnt_c * dt_store > 0.0:
                times[cnt_c] = t
                concentrations[cnt_c] = np.sum(xis)/dV
                masses_vs_time[cnt_c] = masses
                xis_vs_time[cnt_c] = xis
                cnt_c += 1
            tau = collision_step(xis, masses, dV) 
            t += tau
            taus.append(tau)
            print(t)
#            print(taus)
        times[cnt_c] = t
        concentrations[cnt_c] = np.sum(xis)/dV
        masses_vs_time[cnt_c] = masses
        xis_vs_time[cnt_c] = xis
        return times, concentrations, masses_vs_time, xis_vs_time, taus
# simulate_collisions = njit()(simulate_collisions_np)



#%% SHIMA ALGORITHM OLD
# @njit()
# def collision_step(xi, masses, dV, dt):
#     # xi is a 1D array of multiplicities
#     # elements of xi can be 0, indicating vanished droplets
#     # look for the indices, which are not zero
#     indices = np.nonzero(xi)[0]
#     # N = no_spct ("number of super particles per cell total")
#     # no_spct = len(indices)
#     # no_spct = indices.shape[0]
#     no_spct = 5940
#     # Let I = [0,1,2,3,...,N-1] be the list of all EXISTING particle indices
    
#     ### 1. make a random permutation of I by
#     permutation = generate_permutation(no_spct)
#     # print("permutation len")
#     # print(len(permutation))
    
#     ### 2. make [no_spct/2] candidate pairs [(i1,j1),(i2,j2),(),.]
#     no_pairs = no_spct//2
#     cand_i = permutation[0:no_pairs]
#     cand_j = permutation[no_pairs:2*no_pairs]
#     # map onto the original xi-list, to get the indices for that list
    
#     cand_i = indices[cand_i]
#     cand_j = indices[cand_j]
    
#     # print("cand_i len")
#     # print(len(cand_i))
#     # print("cand_j len")
#     # print(len(cand_j))
    
#     ### 3. for each pair (i,j) generate uniform float random number from [0,1)
#     rnd01 = np.random.rand(no_pairs)
#     # print("len(rnd01)")
#     # print(len(rnd01))
#     for pair_n in range(no_pairs):
#         i = cand_i[pair_n]
#         j = cand_j[pair_n]
#         # i_done.append(i)
#         # j_done.append(j)
#         pair_kernel =\
#             kernel_Golovin(masses[i], masses[j])
#         if xi[i] >= xi[j]:
#             xi_max = xi[i]
#             ind_max = i
#             xi_min = xi[j]
#             ind_min = j
#         else:
#             xi_max = xi[j]
#             ind_max = j
#             xi_min = xi[i]
#             ind_min = i
            
#         # xi_max = max(xi[i], xi[j])
#         # now we need p = P_ij,s N_s*(N_s-1)/2 / (no_pairs),
#         # where N_s = no_spct and
#         # P_ij,s = max(xi_i, xi_j) P_ij
#         # P_ij = K_ij dt/dV
#         pair_prob = pair_kernel * dt/dV * xi_max\
#                     * 0.5 * no_spct * (no_spct - 1) / no_pairs
#         # then evaluate
#         # beta
#         # = [p] + 1; if rnd01 < p - [p]
#         # = [p]    ; if rnd01 >= p - [p]
#         pair_prob_int = int(pair_prob)
#         if rnd01[pair_n] < pair_prob - pair_prob_int:
#             g = pair_prob_int + 1
#         else: g = pair_prob_int
#         # print(pair_n, pair_prob, pair_prob_int, rnd01[pair_n], g)
#         if g > 0:
#             # print("collision for pair", i, j)
#             # xi_min = min(xi[i], xi[j])
#             g = min(g,int(xi_max/xi_min))
#             if xi_max - g*xi_min > 0:
#                 # print("case 1")
#                 xi[ind_max] -= g * xi_min
#                 masses[ind_min] += g * masses[ind_max]
#             else:
#                 # print("case 2")
#                 xi[ind_max] = int(0.5 * xi_min)
#                 xi[ind_min] -= int(0.5 * xi_min)
#                 masses[ind_min] += g * masses[ind_max]
#                 masses[ind_max] = masses[ind_min]

# # runtime tests njit/no jit:
# # -> without jit is slightly faster in this case ;)
# # no_spct= 297000
# # mu_m= 3367003.367003367
# # mu_R= 9.303487789072296
# # masses.shape= (297000,)
# # m_max= 48429996.65590665
# # simulate_collisions_np: repeats = 5 no reps =  10
# # simulate_collisions: repeats = 5 no reps =  10
# # best =  6.037e+05 us; worst =  1.163e+06 us; mean = 7.166e+05 +- 2.5e+05 us
# # best =  6.13e+05 us; worst =  1.236e+06 us; mean = 7.385e+05 +- 2.78e+05 us
# def simulate_collisions_np(xi, masses, dV, dt, t_end, store_every):
#     times = np.zeros(1+int(t_end/(store_every * dt)), dtype = np.float64)
#     concentrations = np.zeros(1+int( math.ceil(t_end / (store_every * dt)) ),
#                               dtype = np.float64)
#     masses_vs_time = np.zeros((1+int(t_end/(store_every * dt)), len(masses)),
#                               dtype = np.float64)
#     xis_vs_time = np.zeros((1+int(t_end/(store_every * dt)), len(masses)),
#                               dtype = np.float64)
#     cnt_c = 0 

#     for cnt, t in enumerate(np.arange(0.0,t_end,dt)):
#         if cnt%store_every == 0:
#             times[cnt_c] = t
#             concentrations[cnt_c] = np.sum(xi)/dV
#             masses_vs_time[cnt_c] = masses
#             xis_vs_time[cnt_c] = xi
#             cnt_c += 1
#         collision_step(xi, masses, dV, dt)  
        
#     times[cnt_c] = t_end
#     concentrations[cnt_c] = np.sum(xi)/dV
#     masses_vs_time[cnt_c] = masses
#     xis_vs_time[cnt_c] = xi
    
#     return times, concentrations, masses_vs_time, xis_vs_time
# simulate_collisions = njit()(simulate_collisions_np)
