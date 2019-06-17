#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 16:19:17 2019

@author: bohrer

Simulate the collision of a given population of warm cloud droplets in a given
grid box of size dV = dx * dy * dz
The ensemble of droplets is considered to be well mixed inside the box volume,
such that the probability of each droplet to be at a certain position is
equal for all possible positions in the box

Possible Kernels to test:
    
Golovin kernel:
    
K_ij = b * (m_i + m_j)

-> analytic solution for specific initial distributions
    
Hydrodynamic Collection kernel:
    
K_ij = E(i,j) * pi * (R_i + R_j)**2 * |v_i - v_j|

for the box model the sedimentation velocity is a direct function
of the droplet radii
we need some description or tabulated values for the collection
efficiency E(i,j), e.g. (Long, 1974; Hall, 1980;
Wang et al., 2006) (in Unterstrasser)

(we can later add a kernel for small particles "aerosols", which has
another form (Brownian motion))

Note that the kernel is defined here by

P_ij = K_ij * dt/dV = Probability, that droplets i and j in dV will coalesce
in a time interval (t, t + dt)

Dimension check:
K = m^2 * m / s = m^3/s
P = m^3/s * s/m^3 = 1
"""

#%% MODULE IMPORTS

import math
import numpy as np
import matplotlib.pyplot as plt
# from numba import njit
import os

import constants as c
# from microphysics import compute_mass_from_radius
from microphysics import compute_radius_from_mass
from collision import simulate_collisions_np
from collision import compute_E_col_Hall
from collision import generate_folder_collision
# from collision import simulate_collisions
from init import dst_expo
from init import generate_SIP_ensemble_expo_my_xi_rnd
from init import generate_SIP_ensemble_expo_SingleSIP_weak_threshold
from init import generate_SIP_ensemble_expo_SingleSIP_weak_threshold_nonint
# from collision import generate_permutation
# from collision import simulate_collisions

#Hall_E_col = np.load("Hall_collision_efficiency.npy")
#Hall_R_col = np.load("Hall_collector_radius.npy")
#Hall_R_col_ratio = np.load("Hall_radius_ratio.npy")
#
#for i1 in range(0,31):
#    for j1 in range(21):
#        R1 = Hall_R_col[i1]
#        R2 = Hall_R_col_ratio[j1] * R1        
#        print(i1,j1,Hall_E_col[i1,j1]-compute_E_col_Hall(R1,R2))


#%% INIT

### SET PARAMETERS
myOS = "Linux"
# myOS = "MacOS"

dV = 1.0
# dt = 1.0
# dt = 10.0
dt = 20.0
t_end = 3600.0
dt_store = 600.0

# algorithm = "Shima"
algorithm = "AON_Unt"

# kernel = "Bott"
# kernel = "Hall"
# kernel = "Long"
kernel = "Long_Bott"
# kernel = "Golovin"

init = "SingleSIP"
# init = "my_xi_random"

no_sims = 500
start_seed = 4711

## for SingleSIP random:
# bin exponential scaling factor
kappa = 100

## for my xi random initialization:
# INTENDED number of SIP:
no_spc = 160
# bin linear spreading parameter
eps = 200
# area of cumulative PDF that is covered, also determines the bin width
p_min = 0
p_max = 1.0 - 1.0E-6

# droplet concentration
#n = 100 # cm^(-3)
n0 = 297.0 # cm^(-3)
# liquid water content (per volume)
LWC0 = 1.0E-6 # g/cm^3
# total number of droplets
no_rpc = int(n0 * dV * 1.0E6)
print("no_rpc=", no_rpc)

### DERIVED
# Unterstrasser 2017 uses monomodal exponential distribution:
# f = 1/mu exp(m/mu)
# mean droplet mass
mu = 1.0E15*LWC0 / n0
print("mu_m=", mu)
# mean radius
# mu_R = 9.3 # mu
mu_R = compute_radius_from_mass(mu, c.mass_density_water_liquid_NTP)
print("mu_R=", mu_R)
total_mass_in_cell = dV*LWC0*1.0E6*1.0E15 # in fg = 1.0E-18 kg

# numerical integration parameters for my xi random init
dm = mu*1.0E-5
m0 = 0.0
m1 = 100*mu

if init == "SingleSIP":
    init_pars = [kappa]
elif init == "my_xi_random":
    init_pars = [no_spc, eps]
simdata_path, path =\
    generate_folder_collision(myOS, dV, dt, algorithm, kernel,
                              init, init_pars, no_sims, gen = True)
store_every = int(math.ceil(dt_store/dt))
seed_list = np.arange(start_seed, start_seed+no_sims*2, 2)

### SIMULATION LOOP FOR no_sims SIMULATIONS
par = 1.0/mu
dst = dst_expo
for sim_n in range(no_sims):
    seed = seed_list[sim_n]
    np.random.seed(seed)
    if init == "SingleSIP":
        if algorithm == "AON_Unt":
            masses, xis, m_low, m_high, bins =\
                generate_SIP_ensemble_expo_SingleSIP_weak_threshold_nonint(
                                      1.0/mu, no_rpc, m_high_by_m_low=1.0E6,
                                      kappa=kappa, seed = seed)    
        elif algorithm == "Shima":
            masses, xis, m_low, m_high, bins =\
                generate_SIP_ensemble_expo_SingleSIP_weak_threshold(
                                      1.0/mu, no_rpc, m_high_by_m_low=1.0E6,
                                      kappa=kappa, seed = seed)    
    elif init == "my_xi_random": 
        masses, xis, m_low, m_high =\
            generate_SIP_ensemble_expo_my_xi_rnd(par, no_spc, no_rpc,
                                  total_mass_in_cell,
                                  p_min, p_max, eps,
                                  m0, m1, dm, seed, setseed = True)
# def generate_SIP_ensemble_expo_my_xi_rnd(par, no_spc, no_rpc,
#                                          total_mass_in_cell,
#                                          p_min, p_max, eps,
#                                          m0, m1, dm, seed, setseed = True):    
    # masses = np.random.exponential(mu, no_rpc) # in mu
    # xi = np.ones(no_rpc, dtype = np.int64)
    
    m_max = np.amax(masses)
    print("sim_n", sim_n, "; masses.shape=", masses.shape, "; m_max=", m_max,
          "m_true-m_sys (%)=",
          (total_mass_in_cell-np.sum(masses*xis))/total_mass_in_cell*100,
          "no_pt_true-no_sys (%)=", (no_rpc - np.sum(xis))/no_rpc*100 )
    print("m_high=", f"{m_high:.2e}", "m_low=", m_low)
    
    radii = compute_radius_from_mass(masses, c.mass_density_water_liquid_NTP)
    
    ###
#     fig_name = f"mass_distribution_init_{sim_n}.png"
#     fig, ax = plt.subplots(figsize=(8,8))
#     m_ = np.linspace(0.0,m_max, 10000)
#     no_bins = 50
#     # ax.hist(masses, density=True, bins=50)
#     bins = ax.hist(masses, bins=no_bins)[1]
#     ax.plot(m_, (bins[-1]-bins[0])/no_bins*masses.shape[0]*dst_expo(m_,1.0/mu))
#     # ax.plot(m_, dst_expo(m_, 1.0/mu))
#     ax.set_yscale("log")
#     ax.set_xlabel("particle mass (fg)")
#     ax.set_ylabel("counts")
#     # ax.set_ylabel("normalized counts")
#     ax.set_title(
# f"sim#={sim_n}, n0={n0:.3} cm^-3, LWC0={LWC0} g/cm^3, dV={dV} \
# #SIP={len(xi)}, seed={seed}",
#     pad=20)
#     ax.set_title(
# f"sim#={sim_n}, n0={n0:.3} cm^-3, LWC0={LWC0} g/cm^3, dV={dV} \
# #SIP={no_rpc:.2e}, seed={seed}",
#     pad=20)
    ###
    
    ###
    # fig_name = f"SIP_ensemble_dV_{dV}_no_spc_aim_{no_spc}_sim_no_{sim_n}.png"
    fig_name = f"SIP_ensemble_dV_{dV}_kappa_{kappa}_sim_no_{sim_n}.png"
    fig, ax = plt.subplots(figsize=(8,8))
    # ax.plot(masses, xi, "o")
    ax.plot(radii, xis, "x-")
    # m_max = np.amax(masses)
    # m_ = np.linspace(0.0,m_max, 10000)
    # m_max = masses[-2]
    # m_min = np.amin(masses)
    # no_spc = len(masses)
    # no_rpc = xi.sum()
    # m_ges = np.sum(masses*xis)
    # bin_size = m_max/no_spc
    # bin_size = m_max/(no_spc-1)
    
    # bin_size = (m_high - m_low)/(no_spc)
    
    # ax.plot(m_, no_rpc*np.exp(-m_/mu)\
    #             *(np.exp(0.5*bin_size/mu)-np.exp(-0.5*bin_size/mu)))
    
    # ax.plot(m_, no_rpc*bin_size*dst_expo(m_,1.0/mu))
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ###
    
    fig.tight_layout()
    fig.savefig(path + fig_name)
    plt.close()
    # print("start sim here")

    times, concentrations, masses_vs_time, xis_vs_time =\
        simulate_collisions_np(xis, masses, dV, dt, t_end, store_every,
                               algorithm=algorithm, kernel=kernel)
        # simulate_collisions(xi, masses, dV, dt, t_end, store_every)
    
    np.save(path + f"conc_{sim_n}.npy",concentrations)
    np.save(path + f"times_{sim_n}.npy",times)
    np.save(path + f"masses_vs_time_{sim_n}.npy",masses_vs_time)
    np.save(path + f"xis_vs_time_{sim_n}.npy",xis_vs_time)

# print(xi)
# print(masses)

#%% runtime tests

# from analysis import compare_functions_run_time
# funcs = ["simulate_collisions_np"]
# funcs = ["simulate_collisions"]
# pars = "xi, masses, dt, t_end, store_every"
# rs = [5,5]
# ns = [10,10]
# compare_functions_run_time(funcs, pars, rs, ns, globals_ = globals())

# ll = [0,1,2,34,837,23]

# ll2 = np.array(ll)

# print(ll2)
# print(ll2.dtype)
