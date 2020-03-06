#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 11:15:03 2019

@author: jdesk
"""
import os
import math
import numpy as np
from numba import njit

import matplotlib.pyplot as plt

import constants as c
# from microphysics import compute_mass_from_radius
from microphysics import compute_radius_from_mass
from microphysics import compute_radius_from_mass_jit

kappa = 40
eta = 1.0E-9

no_sims = 500
start_seed = 3711
# start_seed = 4107
# start_seed = 4385
# start_seed = 3811

gen_method = "SinSIP"
# kernel = "Golovin"
kernel = "Long_Bott"

# dt = 1.0
dt = 20.0
dV = 1.0
# dt_save = 40.0
dt_save = 600.0
# t_end = 200.0
t_end = 3600.0

seed = 3711
seed_list = np.arange(start_seed, start_seed+no_sims*2, 2)
ensemble_dir =\
    f"/mnt/D/sim_data/col_box_mod/ensembles/{gen_method}/eta_{eta:.0e}/kappa_{kappa}/"


for seed in seed_list:
    masses = np.load(ensemble_dir + f"masses_seed_{seed}.npy")
    xis = np.load(ensemble_dir + f"xis_seed_{seed}.npy")
    print(xis[-3]/xis[-1],xis[-2]/xis[-1])
# m_max = masses.max()

# print( compute_radius_from_mass(masses
#                                 *1.0e18, c.mass_density_water_liquid_NTP) )


