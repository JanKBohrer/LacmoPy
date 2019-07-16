#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 14:47:43 2019

@author: jdesk
"""
#%% IMPORTS AND DEFINITIONS
import os
import math
import numpy as np
from numba import njit

import matplotlib.pyplot as plt

import constants as c
# from microphysics import compute_mass_from_radius
from microphysics import compute_radius_from_mass
from microphysics import compute_radius_from_mass_jit
# from microphysics import compute_mass_from_radius

# from collision_Unt import analyze_ensemble_data

# from collision import kernel_Golovin

# plt.loglog(masses,xis)
# plt.loglog(radii,xis,"x")

# par[0] belongs to the largest exponential x^(n-1) for par[i], i = 0, .., n 
@njit()
def compute_polynom(par,x):
    res = par[0] * x + par[1]
    for a in par[2:]:
        res = res * x + a
    return res

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

# with material const from Bott...
p1_Beard = (-0.318657e1, 0.992696, -0.153193e-2,
          -0.987059e-3,-0.578878e-3,0.855176e-4,-0.327815e-5)[::-1]
p2_Beard = (-0.500015e1,0.523778e1,-0.204914e1,0.475294,-0.542819e-1,
           0.238449e-2)[::-1]
one_sixth = 1.0/6.0
v_max_Beard = 9.11498

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

# P_ij = dt/dV K_ij
# K_ij = b * (m_i + m_j)
# b = 1.5 m^3/(kg s) (Unterstrasser 2017, p. 1535)
@njit()
def kernel_Golovin(m_i, m_j):
    return 1.5 * (m_i + m_j)

# Kernel "Long" as used in Unterstrasser 2017, which he got from Bott
four_pi_over_three = 4.0/3.0*math.pi
@njit()
def kernel_Long_Bott(m_i, m_j):
    R_i = compute_radius_from_mass_jit(m_i*1.0E18,
                                       c.mass_density_water_liquid_NTP)
    R_j = compute_radius_from_mass_jit(m_j*1.0E18,
                                       c.mass_density_water_liquid_NTP)
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

# kernel = kernel_Golovin
def collision_step_Golovin_np(xis, masses, dt, dV):
    # check each i-j combination for a possible collection event
    no_SIPs = xis.shape[0]

    rnd = np.random.rand( (no_SIPs*(no_SIPs-1))//2 )
    # print(rnd[0])
    cnt = 0
    for j in range(1,no_SIPs):
        for i in range(j):
            # print(j,i)
            if xis[i] <= xis[j]:
                ind_min = i
                ind_max = j
            else:
                ind_min = j
                ind_max = i
            xi_min = xis[ind_min] # = nu_i in Unt
            xi_max = xis[ind_max] # = nu_j in Unt
            m_min = masses[ind_min] # = mu_i in Unt
            m_max = masses[ind_max] # = mu_j in Unt

            p_crit = xi_max * kernel_Golovin(m_min,m_max) * dt/dV

            if p_crit > 1.0:
                # multiple collection
                xi_col = p_crit * xi_min
                masses[ind_min] = (xi_min*m_min + xi_col*m_max) / xi_min
                xis[ind_max] -= xi_col
                # print(j,i,p_crit)
            elif p_crit > rnd[cnt]:
                masses[ind_min] += m_max
                xis[ind_max] -= xi_min
                # print(j,i,p_crit,rnd[cnt])
            cnt += 1
collision_step_Golovin = njit()(collision_step_Golovin_np)

def collision_step_Long_Bott_np(xis, masses, dt, dV):
    # check each i-j combination for a possible collection event
    no_SIPs = xis.shape[0]
    # ind = generate_permutation(no_SIPs)

    rnd = np.random.rand( (no_SIPs*(no_SIPs-1))//2 )
    # print(rnd[0])
    cnt = 0
    for j in range(1,no_SIPs):
        for i in range(j):
    # for j_ in range(1,no_SIPs):
    #     for i_ in range(j):
            # j = ind[j_]
            # i = ind[i_]

            # print(j,i)
            if xis[i] <= xis[j]:
                ind_min = i
                ind_max = j
            else:
                ind_min = j
                ind_max = i
            xi_min = xis[ind_min] # = nu_i in Unt
            xi_max = xis[ind_max] # = nu_j in Unt
            m_min = masses[ind_min] # = mu_i in Unt
            m_max = masses[ind_max] # = mu_j in Unt

            p_crit = xi_max * kernel_Long_Bott(m_min, m_max) * dt/dV

            if p_crit > 1.0:
                # multiple collection
                xi_col = p_crit * xi_min
                masses[ind_min] = (xi_min*m_min + xi_col*m_max) / xi_min
                xis[ind_max] -= xi_col
                # print(j,i,p_crit)
            elif p_crit > rnd[cnt]:
                xi_rel_dev = (xi_max-xi_min)/xi_max
                if xi_rel_dev < 1.0E-4:
                    print("xi_i approx xi_j, xi_rel_dev =",
                          xi_rel_dev,
                          " in collision")
                    xi_ges = xi_min + xi_max
                    masses[ind_min] = 2.0 * ( xi_min*m_min + xi_max*m_max ) \
                                      / xi_ges
                    masses[ind_max] = masses[ind_min]
                    xis[ind_max] = 0.5 * 0.7 * xi_ges
                    xis[ind_min] = 0.5 * xi_ges - xis[ind_max]
                else:
                    masses[ind_min] += m_max
                    xis[ind_max] -= xi_min
                # print(j,i,p_crit,rnd[cnt])
            cnt += 1
collision_step_Long_Bott = njit()(collision_step_Long_Bott_np)

# def save_data(masses, xis, t, path):
#     t = int(t)
#     np.save(path + f"xis_{t}", xis)
#     np.save(path + f"masses_{t}", masses)

def simulate_collisions(xis, masses, dt, dV, t_end, dt_save, seed, save_dir,
                        kernel):
    if kernel == "Golovin":
        collision_step = collision_step_Golovin
    elif kernel == "Long_Bott":
        collision_step = collision_step_Long_Bott
    np.random.seed(seed)
    # save_path = save_dir + f"seed_{seed}/"
    # t = 0.0
    no_SIPs = xis.shape[0]
    no_steps = int(math.ceil(t_end/dt))
    # save data at t=0, every dt_save and at the end
    no_saves = int(t_end/dt_save - 0.001) + 2
    dn_save = int(math.ceil(dt_save/dt))

    xis_vs_time = np.zeros((no_saves,no_SIPs), dtype=np.float64)
    masses_vs_time = np.zeros((no_saves,no_SIPs), dtype=np.float64)
    save_times = np.zeros(no_saves)
    # step_n = 0
    save_n = 0
    for step_n in range(no_steps):
        if step_n % dn_save == 0:
            t = step_n * dt
            xis_vs_time[save_n] = np.copy(xis)
            masses_vs_time[save_n] = np.copy(masses)
            save_times[save_n] = t
            save_n += 1

            # save_data()
        collision_step(xis, masses, dt, dV)
        # t += dt
    t = (step_n+1) * dt
    xis_vs_time[save_n] = np.copy(xis)
    masses_vs_time[save_n] = np.copy(masses)
    save_times[save_n] = t
    np.save(save_dir + f"xis_vs_time_{seed}", xis_vs_time)
    np.save(save_dir + f"masses_vs_time_{seed}", masses_vs_time)
    np.save(save_dir + f"save_times_{seed}", save_times)

    # return xis_vs_time, masses_vs_time, save_times

#%% AON Method from Unterstrasser

# simulate = True
simulate = False
analyze = True
# analyze = False
plotting = True
# plotting = False
plot_moments_kappa_var = True

# kappa = 40
# kappa = 800
eta = 1.0E-9

# kappa_list=[40]
# kappa_list=[5,10,20,40,60,100,200]
# kappa_list=[5,10,20,40,60,100,200,400,600]
kappa_list=[5,10,20,40,60,100,200,400,600,800]
# no_sims = 163
# no_sims = 450
no_sims = 59
start_seed = 3711
# start_seed = 4107
# start_seed = 4385
# start_seed = 3811

no_bins = 50

gen_method = "SinSIP"
# kernel = "Golovin"
kernel = "Long_Bott"

bin_method = "auto_bin"

# dt = 1.0
dt = 20.0
dV = 1.0
# dt_save = 40.0
dt_save = 600.0
# t_end = 200.0
t_end = 3600.0

# ensemble_dir =\
#     f"/mnt/D/sim_data/col_box_mod/ensembles/{gen_method}/kappa_{kappa}/"
    # f"/mnt/D/sim_data/col_box_mod/ensembles/{gen_method}/eta_{eta:.0e}/"

# seed = start_seed
# radii = compute_radius_from_mass(masses*1.0E18,
#                                  c.mass_density_water_liquid_NTP)

### SIMULATE COLLISIONS

if simulate:
    for kappa in kappa_list:
        # SIP ensembles are already stored in directory
        # LINUX desk
        ensemble_dir =\
        f"/mnt/D/sim_data/col_box_mod/ensembles/{gen_method}/eta_{eta:.0e}/kappa_{kappa}/"
        save_dir =\
        f"/mnt/D/sim_data/col_box_mod/results/{kernel}/{gen_method}/eta_{eta:.0e}/kappa_{kappa}/dt_{int(dt)}/"
        # f"/mnt/D/sim_data/col_box_mod/results/{kernel}/{gen_method}/kappa_{kappa}/dt_{int(dt)}/perm/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        sim_params = [dt, dV, no_sims, kappa, start_seed]
        
        seed_list = np.arange(start_seed, start_seed+no_sims*2, 2)
        for cnt,seed in enumerate(seed_list):
            # LOAD ENSEMBLE DATA
            masses = np.load(ensemble_dir + f"masses_seed_{seed}.npy")
            xis = np.load(ensemble_dir + f"xis_seed_{seed}.npy")
            print(f"sim {cnt}: seed {seed} simulation start")
            simulate_collisions(xis, masses, dt, dV, t_end, dt_save, seed, save_dir,
                                kernel)
            print(f"sim {cnt}: seed {seed} simulation finished")

#%% DATA ANALYSIS
### DATA ANALYSIS
# IN WORK: SAVE MORE DATA AFTER ANALYSIS FOR FASTER PLOTTING
# kappa_list=[5,10,20,40,60,100,200,400,600,800]
# kappa_list=[5,10,20,40,60,100,200]
# kappa_list=[5,10,20,40,60,100,200,400,600]

# kappa = 5
if analyze:
    for kappa in kappa_list:
        # no_sims = 500
        # no_bins = 10
        # no_bins = 25
        # no_bins = 30
        # gen_method = "SinSIP"
        # no_bins = 50
        # no_bins = 60
        # bin_mode = 1 for bins equal dist on log scale
        # bin_mode = 1
        
        # dt = 20.0
        # dV = 1.0
        
        # start_seed = 3711
        
        # kernel = "Golovin"
        # kernel = "Long_Bott"
    
        ### for myHisto generation (additional parameters)
        # spread_mode = 0 # spreading based on lin scale
        # # shift_factor = 1.0
        # shift_factor = 0.5
        # overflow_factor = 2.0
        # scale_factor = 1.0
        
        # load data vs time
        # seed = start_seed
        
        # load_dir = \
        # f"/mnt/D/sim_data/col_box_mod/results/{kernel}/{gen_method}/kappa_{kappa}/dt_{int(dt)}/"
        # f"/mnt/D/sim_data/col_box_mod/results/{gen_method}/kappa_{kappa}/dt_{int(dt)}/"
        
        load_dir =\
        f"/mnt/D/sim_data/col_box_mod/results/{kernel}/{gen_method}/eta_{eta:.0e}/kappa_{kappa}/dt_{int(dt)}/"
        # f"/mnt/D/sim_data/col_box_mod/results/{kernel}/{gen_method}/kappa_{kappa}/dt_{int(dt)}/"
        # f"/mnt/D/sim_data/col_box_mod/results/{kernel}/{gen_method}/kappa_{kappa}/dt_{int(dt)}/perm/"
        
        save_times = np.load(load_dir + f"save_times_{start_seed}.npy")
        
        seed_list = np.arange(start_seed, start_seed+no_sims*2, 2)
        
        masses_vs_time = []
        xis_vs_time = []
        for seed in seed_list:
            masses_vs_time.append(np.load(load_dir + f"masses_vs_time_{seed}.npy"))
            xis_vs_time.append(np.load(load_dir + f"xis_vs_time_{seed}.npy"))
        
        masses_vs_time_T = []
        xis_vs_time_T = []
        
        no_times = len(save_times)
        
        for time_n in range(no_times):
            masses_ = []
            xis_ = []
            for i,m in enumerate(masses_vs_time):
                masses_.append(m[time_n])
                xis_.append(xis_vs_time[i][time_n])
            masses_vs_time_T.append(masses_)
            xis_vs_time_T.append(xis_)
        
        f_m_num_avg_vs_time = np.zeros( (no_times, no_bins), dtype = np.float64 )
        f_m_num_std_vs_time = np.zeros( (no_times, no_bins), dtype = np.float64 )
        g_m_num_avg_vs_time = np.zeros( (no_times, no_bins), dtype = np.float64 )
        g_m_num_std_vs_time = np.zeros( (no_times, no_bins), dtype = np.float64 )
        g_ln_r_num_avg_vs_time = np.zeros( (no_times, no_bins), dtype = np.float64 )
        g_ln_r_num_std_vs_time = np.zeros( (no_times, no_bins), dtype = np.float64 )
        
        bins_mass_vs_time = np.zeros((no_times,no_bins+1),dtype=np.float64)
        bins_mass_width_vs_time = np.zeros((no_times,no_bins),dtype=np.float64)
        bins_rad_width_log_vs_time = np.zeros((no_times,no_bins),dtype=np.float64)
        bins_mass_centers = []
        bins_rad_centers = []
        
        m_max_vs_time = np.zeros(no_times,dtype=np.float64)
        m_min_vs_time = np.zeros(no_times,dtype=np.float64)
        bin_factors_vs_time = np.zeros(no_times,dtype=np.float64)
        
        moments_vs_time = np.zeros((no_times,4,no_sims),dtype=np.float64)
        
        last_bin_factor = 1.0
        # last_bin_factor = 1.5
        first_bin_factor = 1.0
        # first_bin_factor = 0.8
        for time_n, masses in enumerate(masses_vs_time_T):
            xis = xis_vs_time_T[time_n]
            masses_sampled = np.concatenate(masses)
            xis_sampled = np.concatenate(xis)
            # print(time_n, xis_sampled.min(), xis_sampled.max())
        
            m_min = masses_sampled.min()
            m_max = masses_sampled.max()
    
            R_min = compute_radius_from_mass(m_min*1.0E18,
                                                c.mass_density_water_liquid_NTP)
            R_max = compute_radius_from_mass(m_max*1.0E18,
                                                c.mass_density_water_liquid_NTP)
    
            xi_min = xis_sampled.min()
            xi_max = xis_sampled.max()
    
            print(kappa, time_n, f"{xi_max/xi_min:.3e}",
                  xis_sampled.shape[0]/no_sims, R_min, R_max)
        
            m_min_vs_time[time_n] = m_min
            m_max_vs_time[time_n] = m_max
        
            bin_factor = (m_max/m_min)**(1.0/no_bins)
            bin_factors_vs_time[time_n] = bin_factor
            # bin_log_dist = np.log(bin_factor)
            # bin_log_dist_half = 0.5 * bin_log_dist
            # add dummy bins for overflow
            # bins_mass = np.zeros(no_bins+3,dtype=np.float64)
            bins_mass = np.zeros(no_bins+1,dtype=np.float64)
            bins_mass[0] = m_min
            # bins_mass[0] = m_min / bin_factor
            for bin_n in range(1,no_bins+1):
                bins_mass[bin_n] = bins_mass[bin_n-1] * bin_factor
            # the factor 1.01 is for numerical stability: to be sure
            # that m_max does not contribute to a bin larger than the
            # last bin
            # bins_mass[-1] *= 1.0001
            bins_mass[-1] *= last_bin_factor
            # the factor 0.99 is for numerical stability: to be sure
            # that m_min does not contribute to a bin smaller than the
            # 0-th bin
            # bins_mass[0] *= 0.9999
            bins_mass[0] *= first_bin_factor
            # m_0 = m_min / np.sqrt(bin_factor)
            # bins_mass_log = np.log(bins_mass)
        
            bins_mass_vs_time[time_n] = bins_mass
            bins_rad = compute_radius_from_mass(bins_mass*1.0E18,
                                                c.mass_density_water_liquid_NTP)
            bins_mass_log = np.log(bins_mass)
            bins_rad_log = np.log(bins_rad)
        
            bins_mass_width = (bins_mass[1:]-bins_mass[:-1])
            bins_rad_width = (bins_rad[1:]-bins_rad[:-1])
            bins_rad_width_log = (bins_rad_log[1:]-bins_rad_log[:-1])
            bins_mass_width_vs_time[time_n] = bins_mass_width
            bins_rad_width_log_vs_time[time_n] = bins_rad_width_log
        
            f_m_counts = np.histogram(masses_sampled,bins_mass)[0]
        
            # define centers on lin scale
            bins_mass_center_lin = 0.5 * (bins_mass[:-1] + bins_mass[1:])
            bins_rad_center_lin = 0.5 * (bins_rad[:-1] + bins_rad[1:])
        
            # define centers on the logarithmic scale
            bins_mass_center_log = np.exp(0.5 * (bins_mass_log[:-1] + bins_mass_log[1:]))
            bins_rad_center_log = np.exp(0.5 * (bins_rad_log[:-1] + bins_rad_log[1:]))
        
            # bins_mass are not equally spaced on log scale because of scaling
            # of the first and last bin
            # bins_mass_center_log = bins_mass[:-1] * np.sqrt(bin_factor)
            # bins_rad_center_log = bins_rad[:-1] * np.sqrt(bin_factor)
        
            # bins_mass_center_log = bins_mass[:-1] * 10**(1.0/(2.0*kappa))
            # bins_rad_center_log = bins_rad[:-1] * 10**(1.0/(2.0*kappa))
        
            # define the center of mass for each bin and set it as the "bin center"
            # bins_mass_center_COM = g_m_num_sampled/f_m_num_sampled
            # bins_rad_center_COM =\
            #     compute_radius_from_mass(bins_mass_center_COM*1.0E18,
            #                              c.mass_density_water_liquid_NTP)
        
            # set the bin "mass centers" at the right spot such that
            # f_avg_i in bin in = f(mm_i), where mm_i is the "mass center"
            m_avg = masses_sampled.sum() / xis_sampled.sum()
            bins_mass_center_exact = bins_mass[:-1] \
                                     + m_avg * np.log(bins_mass_width\
                  / (m_avg * (1-np.exp(-bins_mass_width/m_avg))))
            bins_rad_center_exact =\
                compute_radius_from_mass(bins_mass_center_exact*1.0E18,
                                         c.mass_density_water_liquid_NTP)
            bins_mass_centers.append( np.array((bins_mass_center_lin,
                                      bins_mass_center_log,
                                      bins_mass_center_exact)) )
        
            bins_rad_centers.append( np.array((bins_rad_center_lin,
                                     bins_rad_center_log,
                                     bins_rad_center_exact)) )
        
            ### STATISTICAL ANALYSIS OVER no_sim runs
            # get f(m_i) curve for each "run" with same bins for all ensembles
            f_m_num = []
            g_m_num = []
            g_ln_r_num = []
        
        
            for sim_n,mass in enumerate(masses):
                rad = compute_radius_from_mass(mass*1.0E18,
                                               c.mass_density_water_liquid_NTP)
                f_m_num.append(np.histogram(mass, bins_mass, weights=xis[sim_n])[0] \
                               / (bins_mass_width * dV))
                g_m_num.append(np.histogram(mass, bins_mass, weights=xis[sim_n]*mass)[0] \
                               / (bins_mass_width * dV))
        
                # build g_ln_r = 3*m*g_m DIRECTLY from data
                g_ln_r_num.append( np.histogram(rad, bins_rad,
                                                weights=xis[sim_n]*mass)[0] \
                                   / (bins_rad_width_log * dV) )
        
                moments_vs_time[time_n,0,sim_n] = xis[sim_n].sum() / dV
                for n in range(1,4):
                    moments_vs_time[time_n,n,sim_n] = np.sum(xis[sim_n]*mass**n) / dV
        
            # f_m_num = np.array(f_m_num)
            # g_m_num = np.array(g_m_num)
            # g_ln_r_num = np.array(g_ln_r_num)
            
            f_m_num_avg_vs_time[time_n] = np.average(f_m_num, axis=0)
            f_m_num_std_vs_time[time_n] = \
                np.std(f_m_num, axis=0, ddof=1) / np.sqrt(no_sims)
            g_m_num_avg_vs_time[time_n] = np.average(g_m_num, axis=0)
            g_m_num_std_vs_time[time_n] = \
                np.std(g_m_num, axis=0, ddof=1) / np.sqrt(no_sims)
            g_ln_r_num_avg_vs_time[time_n] = np.average(g_ln_r_num, axis=0)
            g_ln_r_num_std_vs_time[time_n] = \
                np.std(g_ln_r_num, axis=0, ddof=1) / np.sqrt(no_sims)
        
        R_min_vs_time = compute_radius_from_mass(m_min_vs_time*1.0E18,
                                                 c.mass_density_water_liquid_NTP)
        R_max_vs_time = compute_radius_from_mass(m_max_vs_time*1.0E18,
                                                 c.mass_density_water_liquid_NTP)
        
        moments_vs_time_avg = np.average(moments_vs_time, axis=2)
        moments_vs_time_std = np.std(moments_vs_time, axis=2, ddof=1) \
                              / np.sqrt(no_sims)
        
        moments_vs_time_Unt = np.zeros_like(moments_vs_time_avg)
        
        # mom_fac = math.log(10)/(3*kappa)
        for time_n in range(no_times):
            for n in range(4):
                moments_vs_time_Unt[time_n,n] =\
                    math.log(bin_factors_vs_time[time_n]) / 3.0 \
                    * np.sum( g_ln_r_num_avg_vs_time[time_n]
                                        * (bins_mass_centers[time_n][1])**(n-1) )
                    # np.sum( g_ln_r_num_avg_vs_time[time_n]
                    #         * (bins_mass_centers[time_n][1])**(n-1)
                    #         * bins_rad_width_log_vs_time[time_n] )
                    # mom_fac * np.sum( g_m_num_avg_vs_time[time_n]
                    #                    * (bins_mass_centers[time_n][1])**(n-1) )

        np.save(load_dir + f"moments_vs_time_avg_no_sims_{no_sims}_no_bins_{no_bins}.npy", moments_vs_time_avg)
        np.save(load_dir + f"moments_vs_time_std_no_sims_{no_sims}_no_bins_{no_bins}.npy", moments_vs_time_std)
        np.save(load_dir + f"f_m_num_avg_vs_time_no_sims_{no_sims}_no_bins_{no_bins}.npy", f_m_num_avg_vs_time)
        np.save(load_dir + f"g_m_num_avg_vs_time_no_sims_{no_sims}_no_bins_{no_bins}.npy", g_m_num_avg_vs_time)
        np.save(load_dir + f"g_ln_r_num_avg_vs_time_no_sims_{no_sims}_no_bins_{no_bins}.npy", g_ln_r_num_avg_vs_time)
        np.save(load_dir + f"bins_mass_centers_{no_sims}_no_bins_{no_bins}.npy", bins_mass_centers)
        np.save(load_dir + f"bins_rad_centers_{no_sims}_no_bins_{no_bins}.npy", bins_rad_centers)


#%% PLOTTING
if plotting:
    for kappa in kappa_list:
        load_dir =\
        f"/mnt/D/sim_data/col_box_mod/results/{kernel}/{gen_method}/eta_{eta:.0e}/kappa_{kappa}/dt_{int(dt)}/"

        # bins_mass_centers =
        bins_mass_centers = np.load(load_dir + f"bins_mass_centers_{no_sims}_no_bins_{no_bins}.npy")
        bins_rad_centers = np.load(load_dir + f"bins_rad_centers_{no_sims}_no_bins_{no_bins}.npy")

        f_m_num_avg_vs_time = np.load(load_dir + f"f_m_num_avg_vs_time_no_sims_{no_sims}_no_bins_{no_bins}.npy")
        g_m_num_avg_vs_time = np.load(load_dir + f"g_m_num_avg_vs_time_no_sims_{no_sims}_no_bins_{no_bins}.npy")
        g_ln_r_num_avg_vs_time = np.load(load_dir + f"g_ln_r_num_avg_vs_time_no_sims_{no_sims}_no_bins_{no_bins}.npy")

        moments_vs_time_avg = np.load(load_dir + f"moments_vs_time_avg_no_sims_{no_sims}_no_bins_{no_bins}.npy")
        moments_vs_time_std = np.load(load_dir + f"moments_vs_time_std_no_sims_{no_sims}_no_bins_{no_bins}.npy")

        fig_name = "fm_gm_glnr_vs_t"
        fig_name += f"_kappa_{kappa}_dt_{int(dt)}_no_sims_{no_sims}_no_bins_{no_bins}.png"
        no_rows = 3
        fig, axes = plt.subplots(nrows=no_rows, figsize=(10,8*no_rows))
        # ax.loglog(radii, xis, "x")
        # ax.loglog(bins_mid[:51], H, "x-")
        # ax.vlines(bins_rad, xis.min(), xis.max(), linewidth=0.5, linestyle="dashed")
        ax = axes[0]
        for time_n in range(no_times):
            ax.plot(bins_mass_centers[time_n][0], f_m_num_avg_vs_time[time_n])
        # ax.plot(bins_mass_centers[0][0], f_m_num_avg_vs_time[0], "x")
        # ax.plot(m_, f_m_ana_)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("mass (kg)")
        ax.set_ylabel(r"$f_m$ $\mathrm{(kg^{-1} \, m^{-3})}$")
        if kernel == "Golovin":
            ax.set_xticks( np.logspace(-15,-5,11) )
            ax.set_yticks( np.logspace(5,21,17) )
        ax.grid()
        
        ax = axes[1]
        for time_n in range(no_times):
            ax.plot(bins_mass_centers[time_n][0], g_m_num_avg_vs_time[time_n])
        # ax.plot(bins_mass_centers[0][0], g_m_num_avg_vs_time[0], "x")
        # ax.plot(m_, g_m_ana_)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("mass (kg)")
        ax.set_ylabel(r"$g_m$ $\mathrm{(m^{-3})}$")
        if kernel == "Golovin":
            ax.set_xticks( np.logspace(-15,-5,11) )
            ax.set_yticks( np.logspace(-2,9,12) )
        ax.grid()
        
        ax = axes[2]
        for time_n in range(no_times):
            ax.plot(bins_rad_centers[time_n][0], g_ln_r_num_avg_vs_time[time_n]*1000.0)
        # ax.plot(bins_rad_centers[0][0], g_ln_r_num_avg_vs_time[0]*1000.0, "x")
        # ax.plot(R_, g_ln_r_ana_)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel("radius $\mathrm{(\mu m)}$")
        ax.set_ylabel(r"$g_{\ln(r)}$ $\mathrm{(g \; m^{-3})}$")
        # ax.xaxis.set_ticks(np.logspace(np.log10(0.6), np.log10(30),18))
        # ax.xaxis.set_ticks([0.6,1.0,2.0,5.0,10.0,20.0,30.0])
        # ax.set_xticks([0.6,1.0,2.0,5.0,10.0,20.0,30.0])
        # import matplotlib
        # ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        # ax.get_xaxis().get_major_formatter().labelOnlyBase = False
        # ax.yaxis.set_ticks(np.logspace(-11,0,12))
        if kernel == "Golovin":
            ax.set_xticks( np.logspace(0,3,4) )
            # ax.set_yticks( np.logspace(-4,1,6) )
            ax.set_xlim([1.0,2.0E3])
            ax.set_ylim([1.0E-4,10.0])
        elif kernel == "Long_Bott":
            ax.set_xlim([1.0,5.0E3])
            ax.set_ylim([1.0E-4,10.0])
        ax.grid(which="major")
        
        for ax in axes:
            ax.tick_params(which="both", bottom=True, top=True, left=True, right=True)
        
        # ax = axes[4]
        # for n in range(4):
        #     ax.plot(n*np.ones_like(moments_sampled[n]),
        #             moments_sampled[n]/moments_an[n], "o")
        # ax.errorbar(np.arange(4), moments_sampled_avg_norm, moments_sampled_std_norm,
        #             fmt = "x" , c = "k", markersize = 20.0, linewidth =5.0,
        #             capsize=10, elinewidth=5, markeredgewidth=2,
        #             zorder=99)
        # ax.plot(np.arange(4), np.ones_like(np.arange(4)))
        # ax.xaxis.set_ticks([0,1,2,3])
        # ax.set_xlabel("$k$")
        # # ax.set_ylabel(r"($k$-th moment of $f_m$)/(analytic value)")
        # ax.set_ylabel(r"$\lambda_k / \lambda_{k,analytic}$")

        fig.suptitle(
f"dt={dt}, kappa={kappa}, eta={eta:.0e}, no_sims={no_sims}, no_bins={no_bins}\n\
gen_method={gen_method}, kernel={kernel}, bin_method={bin_method}")
        fig.tight_layout()
        plt.subplots_adjust(top=0.95)
        fig.savefig(load_dir + fig_name)
        
        ### PLOT MOMENTS VS TIME
        t_Unt = [0,10,20,30,35,40,50,55,60]
        lam0_Unt = [2.97E8, 2.92E8, 2.82E8, 2.67E8, 2.1E8, 1.4E8,  1.4E7, 4.0E6, 1.2E6]
        t_Unt2 = [0,10,20,30,40,50,60]
        lam2_Unt = [8.0E-15, 9.0E-15, 9.5E-15, 6E-13, 2E-10, 7E-9, 2.5E-8]
        
        fig_name = "moments_vs_time"
        fig_name += f"_kappa_{kappa}_dt_{int(dt)}_no_sims_{no_sims}_no_bins_{no_bins}.png"
        no_rows = 4
        fig, axes = plt.subplots(nrows=no_rows, figsize=(10,5*no_rows))
        for i,ax in enumerate(axes):
            ax.plot(save_times/60, moments_vs_time_avg[:,i],"x-")
            if i != 1:
                ax.set_yscale("log")
            ax.grid()
            ax.set_xticks(save_times/60)
            ax.set_xlim([save_times[0]/60, save_times[-1]/60])
            # ax.tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=False,
            #                  bottom=True, top=False, left=False, right=False)
            ax.tick_params(which="both", bottom=True, top=True, left=True, right=True)
        if kernel == "Golovin":
            axes[0].set_yticks([1.0E6,1.0E7,1.0E8,1.0E9])
            axes[2].set_yticks( np.logspace(-15,-9,7) )
            axes[3].set_yticks( np.logspace(-26,-15,12) )
        
        axes[0].plot(t_Unt,lam0_Unt, "o")
        axes[2].plot(t_Unt2,lam2_Unt, "o")
        fig.suptitle(
f"dt={dt}, kappa={kappa}, eta={eta:.0e}, no_sims={no_sims}, no_bins={no_bins}\n\
gen_method={gen_method}, kernel={kernel}, bin_method={bin_method}")
        fig.tight_layout()
        plt.subplots_adjust(top=0.95)
        fig.savefig(load_dir + fig_name)
#%% PLOT MOMENTS VS TIME for several kappa
# plot_moments_kappa_var = True


if plot_moments_kappa_var:

    # dt = 20.0
    # kappa_list = np.array([5,10,20,40,60,100,200,400,600,800])
    # kappa_list = np.array([5,10,20,40,60,100,200])
    # kappa_list = np.array([5,10,20,40,60,100,200,400,600])
    
    # no_bins = 50
    # no_sims = 50
    # no_sims = 150

    # start_seed = 3711
    
    t_Unt = [0,10,20,30,35,40,50,55,60]
    lam0_Unt = [2.97E8, 2.92E8, 2.82E8, 2.67E8, 2.1E8, 1.4E8,  1.4E7, 4.0E6, 1.2E6]
    t_Unt2 = [0,10,20,30,40,50,60]
    lam2_Unt = [8.0E-15, 9.0E-15, 9.5E-15, 6E-13, 2E-10, 7E-9, 2.5E-8]
    
    fig_dir = \
    f"/mnt/D/sim_data/col_box_mod/results/{kernel}/{gen_method}/eta_{eta:.0e}/"
    # fig_dir = \
    # f"/mnt/D/sim_data/col_box_mod/results/{kernel}/{gen_method}/"
    
    # save_times = np.load(load_dir + f"save_times_{start_seed}.npy")
    
    fig_name = "moments_vs_time_kappa_var"
    fig_name += f"_dt_{int(dt)}_no_sims_{no_sims}_no_bins_{no_bins}.png"
    no_rows = 4
    
    fig, axes = plt.subplots(nrows=no_rows, figsize=(10,6*no_rows))
    
    for kappa in kappa_list:
        load_dir = \
        f"/mnt/D/sim_data/col_box_mod/results/{kernel}/{gen_method}/eta_{eta:.0e}/kappa_{kappa}/dt_{int(dt)}/"
        # load_dir = \
        # f"/mnt/D/sim_data/col_box_mod/results/{kernel}/{gen_method}/kappa_{kappa}/dt_{int(dt)}/"
        save_times = np.load(load_dir + f"save_times_{start_seed}.npy")
        moments_vs_time_avg = np.load(load_dir + f"moments_vs_time_avg_no_sims_{no_sims}_no_bins_{no_bins}.npy")
        for i,ax in enumerate(axes):
            ax.plot(save_times/60, moments_vs_time_avg[:,i],"x-",label=f"{kappa}")
            if i != 1:
                ax.set_yscale("log")
            ax.grid()
            # if i ==0: ax.legend()
            ax.legend()
            ax.set_xticks(save_times/60)
            ax.set_xlim([save_times[0]/60, save_times[-1]/60])
            # ax.tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=False,
            #                  bottom=True, top=False, left=False, right=False)
            ax.tick_params(which="both", bottom=True, top=True, left=True, right=True)
        if kernel == "Golovin":
            axes[0].set_yticks([1.0E6,1.0E7,1.0E8,1.0E9])
            axes[2].set_yticks( np.logspace(-15,-9,7) )
            axes[3].set_yticks( np.logspace(-26,-15,12) )
        
        axes[0].plot(t_Unt,lam0_Unt, "o", c = "k")
        axes[2].plot(t_Unt2,lam2_Unt, "o", c = "k")
    
    fig.tight_layout()
    fig.savefig(fig_dir + fig_name)

