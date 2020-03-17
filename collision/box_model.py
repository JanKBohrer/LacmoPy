#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TROPOS LAGRANGIAN CLOUD MODEL
Super-Droplet method in two-dimensional kinematic framework
(Test Case 1 ICMW 2012, Muhlbauer et al. 2013)
Author: Jan Bohrer (bohrer@tropos.de)
Further contact: Oswald Knoth (knoth@tropos.de)

COLLISION BOX MODEL METHODS

for initialization, the "SingleSIP" method is applied, as proposed by
Unterstrasser 2017, GMD 10: 1521–1548

the all-or-nothing collision algorithm is motivated by 
Shima et al. 2009, Q. J. R. Meteorol. Soc. 135: 1307–1320 and adapted from
Unterstrasser 2017, GMD 10: 1521–1548

basic units:
particle mass, water mass, solute mass in femto gram = 10^-18 kg
particle radius in micro meter ("mu")
all other quantities in SI units
"""

#%% IMPORTS
import math
import numpy as np
import matplotlib.pyplot as plt

import microphysics as mp
import collision.kernel as ker
import collision.all_or_nothing as aon

import distributions as dst

from evaluation import generate_myHisto_SIP_ensemble_np

from plotting import plot_ensemble_data

from distributions import conc_per_mass_expo_np, conc_per_mass_lognormal_np
from golovin import compute_moments_Golovin, dist_vs_time_golo_exp

#%% SIMULATION

def simulate_collisions(SIP_quantities,
                        kernel_quantities, kernel_name, kernel_method,
                        dV, dt, t_end, dt_save, no_cols, seed, save_dir):
    if kernel_name == "Golovin":
        collision_step = aon.collision_step_Golovin
        (xis, masses) = SIP_quantities

    elif kernel_method == "Ecol_grid_R":
        collision_step = aon.collision_step_Ecol_grid_R
        (xis, masses, radii, vel, mass_densities) = SIP_quantities
        (E_col_grid, no_kernel_bins, R_kernel_low_log, bin_factor_R_log) =\
            kernel_quantities
    elif kernel_method == "kernel_grid_m":
        collision_step = aon.collision_step_kernel_grid_m
        (xis, masses) = SIP_quantities
        (kernel_grid, no_kernel_bins, m_kernel_low_log, bin_factor_m_log)=\
            kernel_quantities
        
    np.random.seed(seed)
    no_SIPs = xis.shape[0]
    no_steps = int(math.ceil(t_end/dt))
    # save data at t=0, every dt_save and at the end
    no_saves = int(t_end/dt_save - 0.001) + 2
    dn_save = int(math.ceil(dt_save/dt))
    
    dt_over_dV = dt/dV
    
    xis_vs_time = np.zeros((no_saves,no_SIPs), dtype=np.float64)
    masses_vs_time = np.zeros((no_saves,no_SIPs), dtype=np.float64)
    save_times = np.zeros(no_saves)
    save_n = 0
    if kernel_method == "Ecol_grid_R":
        for step_n in range(no_steps):
            if step_n % dn_save == 0:
                t = step_n * dt
                xis_vs_time[save_n] = np.copy(xis)
                masses_vs_time[save_n] = np.copy(masses)
                save_times[save_n] = t
                save_n += 1
            # for box model: calc. velocity from terminal vel.
            # in general: vel given from dynamic simulation
            collision_step(xis, masses, radii, vel, mass_densities,
                           dt_over_dV, E_col_grid, no_kernel_bins,
                           R_kernel_low_log, bin_factor_R_log, no_cols)
            ker.update_velocity_Beard(vel,radii)
    elif kernel_method == "kernel_grid_m":
        for step_n in range(no_steps):
            if step_n % dn_save == 0:
                t = step_n * dt
                xis_vs_time[save_n] = np.copy(xis)
                masses_vs_time[save_n] = np.copy(masses)
                save_times[save_n] = t
                save_n += 1
            collision_step(xis, masses, dt_over_dV,
                           kernel_grid, no_kernel_bins,
                           m_kernel_low_log, bin_factor_m_log, no_cols)
    elif kernel_method == "analytic":           
        if kernel_name == "Golovin":
            for step_n in range(no_steps):
                if step_n % dn_save == 0:
                    t = step_n * dt
                    xis_vs_time[save_n] = np.copy(xis)
                    masses_vs_time[save_n] = np.copy(masses)
                    save_times[save_n] = t
                    save_n += 1
                collision_step(xis, masses, dt_over_dV, no_cols)
    t = no_steps * dt
    xis_vs_time[save_n] = np.copy(xis)
    masses_vs_time[save_n] = np.copy(masses)
    save_times[save_n] = t
    np.save(save_dir + f"xis_vs_time_{seed}", xis_vs_time)
    np.save(save_dir + f"masses_vs_time_{seed}", masses_vs_time)
    np.save(save_dir + f"save_times_{seed}", save_times)
    
#%% ANALYSIS OF ENSEMBLE DATA

# load data from .npy files, analyze, plot and save plot files
def analyze_ensemble_data(dist, mass_density, kappa, no_sims, ensemble_dir,
                          no_bins, bin_mode,
                          spread_mode, shift_factor, overflow_factor,
                          scale_factor, act_plot_ensembles):
    if dist == "expo":
        conc_per_mass_np = conc_per_mass_expo_np
        dV, DNC0, DNC0_over_LWC0, r_critmin, kappa, eta,no_sims00,start_seed =\
            tuple(np.load(ensemble_dir + "ensemble_parameters.npy"))
        LWC0_over_DNC0 = 1.0 / DNC0_over_LWC0
        dist_par = (DNC0, DNC0_over_LWC0)
        moments_analytical = dst.moments_analytical_expo
    elif dist =="lognormal":
        conc_per_mass_np = conc_per_mass_lognormal_np
        dV, DNC0, mu_m_log, sigma_m_log, mass_density, r_critmin, \
        kappa, eta, no_sims00, start_seed = \
            tuple(np.load(ensemble_dir + "ensemble_parameters.npy"))
        dist_par = (DNC0, mu_m_log, sigma_m_log)
        moments_analytical = dst.moments_analytical_lognormal_m

    start_seed = int(start_seed)
    no_sims00 = int(no_sims00)
    seed_list = np.arange(start_seed, start_seed+no_sims*2, 2)
    
    ### ANALYSIS START
    masses = []
    xis = []
    radii = []
    
    moments_sampled = []
    for i,seed in enumerate(seed_list):
        masses.append(np.load(ensemble_dir + f"masses_seed_{seed}.npy"))
        xis.append(np.load(ensemble_dir + f"xis_seed_{seed}.npy"))
        radii.append(np.load(ensemble_dir + f"radii_seed_{seed}.npy"))
    
        moments = np.zeros(4,dtype=np.float64)
        moments[0] = xis[i].sum() / dV
        for n in range(1,4):
            moments[n] = np.sum(xis[i]*masses[i]**n) / dV
        moments_sampled.append(moments)
    
    masses_sampled = np.concatenate(masses)
    radii_sampled = np.concatenate(radii)
    xis_sampled = np.concatenate(xis)
    
    # moments analysis
    moments_sampled = np.transpose(moments_sampled)
    moments_an = np.zeros(4,dtype=np.float64)
    for n in range(4):
        moments_an[n] = moments_analytical(n, *dist_par)
        
    print(f"### kappa {kappa} ###")    
    print("moments analytic:", moments_an)    
    print("rel. deviation")    
    print("moment-order    (average-analytic)/analytic:")
    for n in range(4):
        print(n, (np.average(moments_sampled[n])-moments_an[n])/moments_an[n] )
    
    moments_sampled_avg_norm = np.average(moments_sampled, axis=1) / moments_an
    moments_sampled_std_norm = np.std(moments_sampled, axis=1) \
                               / np.sqrt(no_sims) / moments_an
    
    m_min = masses_sampled.min()
    m_max = masses_sampled.max()
    
    R_min = radii_sampled.min()
    R_max = radii_sampled.max()
    
    bins_mass = np.load(ensemble_dir + "bins_mass.npy")
    bins_rad = np.load(ensemble_dir + "bins_rad.npy")
    bin_factor = 10**(1.0/kappa)
    
    ### build log bins "intuitively" = "auto"
    if bin_mode == 1:
        bin_factor_auto = (m_max/m_min)**(1.0/no_bins)
        # bin_log_dist = np.log(bin_factor)
        # bin_log_dist_half = 0.5 * bin_log_dist
        # add dummy bins for overflow
        # bins_mass = np.zeros(no_bins+3,dtype=np.float64)
        bins_mass_auto = np.zeros(no_bins+1,dtype=np.float64)
        bins_mass_auto[0] = m_min
        # bins_mass[0] = m_min / bin_factor
        for bin_n in range(1,no_bins+1):
            bins_mass_auto[bin_n] = bins_mass_auto[bin_n-1] * bin_factor_auto
        # the factor 1.01 is for numerical stability: to be sure
        # that m_max does not contribute to a bin larger than the
        # last bin
        bins_mass_auto[-1] *= 1.0001
        # the factor 0.99 is for numerical stability: to be sure
        # that m_min does not contribute to a bin smaller than the
        # 0-th bin
        bins_mass_auto[0] *= 0.9999
        # m_0 = m_min / np.sqrt(bin_factor)
        # bins_mass_log = np.log(bins_mass)

        bins_rad_auto = mp.compute_radius_from_mass_vec(bins_mass_auto*1.0E18,
                                                        mass_density)

    ###################################################
    ### histogram generation for given bins
    f_m_counts = np.histogram(masses_sampled,bins_mass)[0]
    f_m_ind = np.nonzero(f_m_counts)[0]
    f_m_ind = np.arange(f_m_ind[0],f_m_ind[-1]+1)
    
    no_SIPs_avg = f_m_counts.sum()/no_sims

    bins_mass_ind = np.append(f_m_ind, f_m_ind[-1]+1)
    
    bins_mass = bins_mass[bins_mass_ind]
    
    bins_rad = bins_rad[bins_mass_ind]
    bins_rad_log = np.log(bins_rad)
    bins_mass_width = (bins_mass[1:]-bins_mass[:-1])
    bins_rad_width = (bins_rad[1:]-bins_rad[:-1])
    bins_rad_width_log = (bins_rad_log[1:]-bins_rad_log[:-1])
    
    ### approximate the functions f_m, f_lnR = 3*m*f_m, g_lnR=3*m^2*f_m
    # estimate f_m(m) by binning:
    # DNC_i = f_m(m_i) * dm_i = droplet number conc in bin i with size dm_i
    f_m_num_sampled = np.histogram(masses_sampled,bins_mass,
                                   weights=xis_sampled)[0]
    g_m_num_sampled = np.histogram(masses_sampled,bins_mass,
                                   weights=xis_sampled*masses_sampled)[0]
    
    f_m_num_sampled = f_m_num_sampled / (bins_mass_width * dV * no_sims)
    g_m_num_sampled = g_m_num_sampled / (bins_mass_width * dV * no_sims)
    
    # build g_ln_r = 3*m*g_m DIRECTLY from data
    g_ln_r_num_sampled = np.histogram(radii_sampled,
                                      bins_rad,
                                      weights=xis_sampled*masses_sampled)[0]
    g_ln_r_num_sampled = g_ln_r_num_sampled \
                         / (bins_rad_width_log * dV * no_sims)
    # g_ln_r_num_derived = 3 * bins_mass_center * g_m_num * 1000.0
    
    # define centers on lin scale
    bins_mass_center_lin = 0.5 * (bins_mass[:-1] + bins_mass[1:])
    bins_rad_center_lin = 0.5 * (bins_rad[:-1] + bins_rad[1:])
    
    # define centers on the logarithmic scale
    bins_mass_center_log = bins_mass[:-1] * np.sqrt(bin_factor)
    bins_rad_center_log = bins_rad[:-1] * np.sqrt(bin_factor)
    # bins_mass_center_log = bins_mass[:-1] * 10**(1.0/(2.0*kappa))
    # bins_rad_center_log = bins_rad[:-1] * 10**(1.0/(2.0*kappa))
    
    # define the center of mass for each bin and set it as the "bin center"
    bins_mass_center_COM = g_m_num_sampled/f_m_num_sampled
    bins_rad_center_COM =\
        mp.compute_radius_from_mass_vec(bins_mass_center_COM*1.0E18,
                                        mass_density)
    
    # set the bin "mass centers" at the right spot such that
    # f_avg_i in bin in = f(mm_i), where mm_i is the "mass center"
    if dist == "expo":
        m_avg = LWC0_over_DNC0
    elif dist == "lognormal":
        m_avg = moments_an[1] / dist_par[0]
        
    bins_mass_center_exact = bins_mass[:-1] \
                             + m_avg * np.log(bins_mass_width\
          / (m_avg * (1-np.exp(-bins_mass_width/m_avg))))
    bins_rad_center_exact =\
        mp.compute_radius_from_mass_vec(bins_mass_center_exact*1.0E18,
                                        mass_density)
    
    bins_mass_centers = np.array((bins_mass_center_lin,
                                  bins_mass_center_log,
                                  bins_mass_center_COM,
                                  bins_mass_center_exact))
    bins_rad_centers = np.array((bins_rad_center_lin,
                                  bins_rad_center_log,
                                  bins_rad_center_COM,
                                  bins_rad_center_exact))

    ###################################################
    ### histogram generation for auto bins
    f_m_counts_auto = np.histogram(masses_sampled,bins_mass_auto)[0]
    f_m_ind_auto = np.nonzero(f_m_counts_auto)[0]
    f_m_ind_auto = np.arange(f_m_ind_auto[0],f_m_ind_auto[-1]+1)
    

    bins_mass_ind_auto = np.append(f_m_ind_auto, f_m_ind_auto[-1]+1)
    bins_mass_auto = bins_mass_auto[bins_mass_ind_auto]
    
    bins_rad_auto = bins_rad_auto[bins_mass_ind_auto]
    bins_rad_log_auto = np.log(bins_rad_auto)
    bins_mass_width_auto = (bins_mass_auto[1:]-bins_mass_auto[:-1])
    bins_rad_width_log_auto = (bins_rad_log_auto[1:]-bins_rad_log_auto[:-1])
    
    ### approximate the functions f_m, f_lnR = 3*m*f_m, g_lnR=3*m^2*f_m
    # estimate f_m(m) by binning:
    # DNC_i = f_m(m_i) * dm_i = droplet number conc in bin i with size dm_i
    f_m_num_sampled_auto = np.histogram(masses_sampled,bins_mass_auto,
                                   weights=xis_sampled)[0]
    g_m_num_sampled_auto = np.histogram(masses_sampled,bins_mass_auto,
                                   weights=xis_sampled*masses_sampled)[0]
    
    f_m_num_sampled_auto = f_m_num_sampled_auto\
                           / (bins_mass_width_auto * dV * no_sims)
    g_m_num_sampled_auto = g_m_num_sampled_auto\
                           / (bins_mass_width_auto * dV * no_sims)
    
    # build g_ln_r = 3*m*g_m DIRECTLY from data
    g_ln_r_num_sampled_auto = np.histogram(radii_sampled,
                                      bins_rad_auto,
                                      weights=xis_sampled*masses_sampled)[0]
    g_ln_r_num_sampled_auto = g_ln_r_num_sampled_auto \
                         / (bins_rad_width_log_auto * dV * no_sims)
    
    # define centers on lin scale
    bins_mass_center_lin_auto =\
        0.5 * (bins_mass_auto[:-1] + bins_mass_auto[1:])
    bins_rad_center_lin_auto =\
        0.5 * (bins_rad_auto[:-1] + bins_rad_auto[1:])
    
    # define centers on the logarithmic scale
    bins_mass_center_log_auto = bins_mass_auto[:-1] * np.sqrt(bin_factor)
    bins_rad_center_log_auto = bins_rad_auto[:-1] * np.sqrt(bin_factor)
    
    # define the center of mass for each bin and set it as the "bin center"
    bins_mass_center_COM_auto = g_m_num_sampled_auto/f_m_num_sampled_auto
    bins_rad_center_COM_auto =\
        mp.compute_radius_from_mass_vec(bins_mass_center_COM_auto*1.0E18,
                                        mass_density)
    
    # set the bin "mass centers" at the right spot such that
    # f_avg_i in bin in = f(mm_i), where mm_i is the "mass center"
    if dist == "expo":
        m_avg = LWC0_over_DNC0
    elif dist == "lognormal":
        m_avg = moments_an[1] / dist_par[0]
        
    bins_mass_center_exact_auto = bins_mass_auto[:-1] \
                             + m_avg * np.log(bins_mass_width_auto\
          / (m_avg * (1-np.exp(-bins_mass_width_auto/m_avg))))
    bins_rad_center_exact_auto =\
        mp.compute_radius_from_mass_vec(bins_mass_center_exact_auto*1.0E18,
                                        mass_density)
    
    bins_mass_centers_auto = np.array((bins_mass_center_lin_auto,
                                  bins_mass_center_log_auto,
                                  bins_mass_center_COM_auto,
                                  bins_mass_center_exact_auto))
    bins_rad_centers_auto = np.array((bins_rad_center_lin_auto,
                                  bins_rad_center_log_auto,
                                  bins_rad_center_COM_auto,
                                  bins_rad_center_exact_auto))

    ###########################################################################
    ### STATISTICAL ANALYSIS OVER no_sim runs given bins
    # get f(m_i) curve for each "run" with same bins for all ensembles
    f_m_num = []
    g_m_num = []
    g_ln_r_num = []
    
    for i,mass in enumerate(masses):
        f_m_num.append(np.histogram(mass,bins_mass,weights=xis[i])[0] \
                   / (bins_mass_width * dV))
        g_m_num.append(np.histogram(mass,bins_mass,
                                       weights=xis[i]*mass)[0] \
                   / (bins_mass_width * dV))
    
        # build g_ln_r = 3*m*g_m DIRECTLY from data
        g_ln_r_num.append(np.histogram(radii[i],
                                          bins_rad,
                                          weights=xis[i]*mass)[0] \
                     / (bins_rad_width_log * dV))
    
    f_m_num = np.array(f_m_num)
    g_m_num = np.array(g_m_num)
    g_ln_r_num = np.array(g_ln_r_num)
    
    f_m_num_avg = np.average(f_m_num, axis=0)
    f_m_num_std = np.std(f_m_num, axis=0, ddof=1) / np.sqrt(no_sims)
    g_m_num_avg = np.average(g_m_num, axis=0)
    g_m_num_std = np.std(g_m_num, axis=0, ddof=1) / np.sqrt(no_sims)
    g_ln_r_num_avg = np.average(g_ln_r_num, axis=0)
    g_ln_r_num_std = np.std(g_ln_r_num, axis=0, ddof=1) / np.sqrt(no_sims)
    
    ###########################################################################
    ### STATISTICAL ANALYSIS OVER no_sim runs AUTO BINS
    # get f(m_i) curve for each "run" with same bins for all ensembles
    f_m_num_auto = []
    g_m_num_auto = []
    g_ln_r_num_auto = []
    
    for i,mass in enumerate(masses):
        f_m_num_auto.append(np.histogram(mass,bins_mass_auto,weights=xis[i])[0]
                   / (bins_mass_width_auto * dV))
        g_m_num_auto.append(np.histogram(mass,bins_mass_auto,
                                       weights=xis[i]*mass)[0] \
                   / (bins_mass_width_auto * dV))
    
        # build g_ln_r = 3*m*g_m DIRECTLY from data
        g_ln_r_num_auto.append(np.histogram(radii[i],
                                          bins_rad_auto,
                                          weights=xis[i]*mass)[0] \
                     / (bins_rad_width_log_auto * dV))
    
    f_m_num_auto = np.array(f_m_num_auto)
    g_m_num_auto = np.array(g_m_num_auto)
    g_ln_r_num_auto = np.array(g_ln_r_num_auto)
    
    f_m_num_avg_auto = np.average(f_m_num_auto, axis=0)
    f_m_num_std_auto = np.std(f_m_num_auto, axis=0, ddof=1) / np.sqrt(no_sims)
    g_m_num_avg_auto = np.average(g_m_num_auto, axis=0)
    g_m_num_std_auto = np.std(g_m_num_auto, axis=0, ddof=1) / np.sqrt(no_sims)
    g_ln_r_num_avg_auto = np.average(g_ln_r_num_auto, axis=0)
    g_ln_r_num_std_auto =\
        np.std(g_ln_r_num_auto, axis=0, ddof=1) / np.sqrt(no_sims)

    ###########################################################################
    ### generate f_m, g_m and mass centers with my hist bin method
    LWC0 = moments_an[1]
    f_m_num_avg_my_ext, f_m_num_std_my_ext, g_m_num_avg_my, g_m_num_std_my, \
    h_m_num_avg_my, h_m_num_std_my, \
    bins_mass_my, bins_mass_width_my, \
    bins_mass_centers_my, bins_mass_center_lin_my, lin_par, aa = \
        generate_myHisto_SIP_ensemble_np(masses, xis, m_min, m_max,
                                         dV, DNC0, LWC0,
                                         no_bins, no_sims,
                                         bin_mode, spread_mode,
                                         shift_factor, overflow_factor,
                                         scale_factor)
        
    f_m_num_avg_my = f_m_num_avg_my_ext[1:-1]
    f_m_num_std_my = f_m_num_std_my_ext[1:-1]
    
    ###########################################################################
    ### analytical reference data    
    m_ = np.logspace(np.log10(bins_mass[0]), np.log10(bins_mass[-1]), 1000)
    R_ = mp.compute_radius_from_mass_vec(m_*1.0E18, mass_density)
    f_m_ana_ = conc_per_mass_np(m_, *dist_par)
    g_m_ana_ = m_ * f_m_ana_
    g_ln_r_ana_ = 3 * m_ * g_m_ana_ * 1000.0    

    if act_plot_ensembles:
        plot_ensemble_data(kappa, mass_density, eta, r_critmin,
            dist, dist_par, no_sims, no_bins,
            bins_mass, bins_rad, bins_rad_log, 
            bins_mass_width, bins_rad_width, bins_rad_width_log, 
            bins_mass_centers, bins_rad_centers,
            bins_mass_centers_auto, bins_rad_centers_auto,
            masses, xis, radii, f_m_counts, f_m_ind,
            f_m_num_sampled, g_m_num_sampled, g_ln_r_num_sampled, 
            m_, R_, f_m_ana_, g_m_ana_, g_ln_r_ana_, 
            f_m_num_avg, f_m_num_std, g_m_num_avg, g_m_num_std, 
            g_ln_r_num_avg, g_ln_r_num_std, 
            f_m_num_avg_auto, f_m_num_std_auto, g_m_num_avg_auto,
            g_m_num_std_auto, 
            g_ln_r_num_avg_auto, g_ln_r_num_std_auto, 
            m_min, m_max, R_min, R_max, no_SIPs_avg, 
            moments_sampled, moments_sampled_avg_norm,moments_sampled_std_norm,
            moments_an, lin_par,
            f_m_num_avg_my_ext,
            f_m_num_avg_my, f_m_num_std_my, g_m_num_avg_my, g_m_num_std_my, 
            h_m_num_avg_my, h_m_num_std_my, 
            bins_mass_my, bins_mass_width_my, 
            bins_mass_centers_my, bins_mass_center_lin_my,
            ensemble_dir)
        
### LEAVE THIS: MAY NEED TO RETURN FOR OTHER APPLICATIONS
#    return masses, xis, radii, \
#           m_min, m_max, R_min, R_max, no_SIPs_avg, \
#           m_, R_, f_m_ana_, g_m_ana_, g_ln_r_ana_, \
#           bins_mass, bins_rad, bins_rad_log, \
#           bins_mass_width, bins_rad_width, bins_rad_width_log, \
#           bins_mass_centers, bins_rad_centers, \
#           f_m_counts, f_m_ind,\
#           f_m_num_sampled, g_m_num_sampled, g_ln_r_num_sampled,\
#           f_m_num_avg, f_m_num_std, g_m_num_avg, g_m_num_std, \
#           g_ln_r_num_avg, g_ln_r_num_std, \
#           bins_mass_auto, bins_rad_auto, bins_rad_log_auto, \
#           bins_mass_width_auto, bins_rad_width_auto, bins_rad_width_log_auto,\
#           bins_mass_centers_auto, bins_rad_centers_auto, \
#           f_m_counts_auto, f_m_ind_auto,\
#           f_m_num_sampled_auto, g_m_num_sampled_auto, g_ln_r_num_sampled_auto,\
#           f_m_num_avg_auto, f_m_num_std_auto, g_m_num_avg_auto,\
#           g_m_num_std_auto,\
#           g_ln_r_num_avg_auto, g_ln_r_num_std_auto, \
#           moments_sampled, moments_sampled_avg_norm,moments_sampled_std_norm,\
#           moments_an, \
#           f_m_num_avg_my_ext, \
#           f_m_num_avg_my, f_m_num_std_my, \
#           g_m_num_avg_my, g_m_num_std_my, \
#           h_m_num_avg_my, h_m_num_std_my, \
#           bins_mass_my, bins_mass_width_my, \
#           bins_mass_centers_my, bins_mass_center_lin_my, lin_par, aa    
    
#%% ANALYSIS OF SIM DATA

# the simulation yields masses in unit 1E-18 kg
# to compare moments etc. with other authors, masses are converted to kg
def analyze_sim_data(kappa, mass_density, dV, no_sims,
                     start_seed, no_bins, load_dir):
    save_times = np.load(load_dir + f"save_times_{start_seed}.npy")
    
    seed_list = np.arange(start_seed, start_seed+no_sims*2, 2)
    
    masses_vs_time = []
    xis_vs_time = []
    for seed in seed_list:
        # convert to kg
        masses_vs_time.append(1E-18 *
                              np.load(load_dir + f"masses_vs_time_{seed}.npy"))
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
    g_ln_r_num_avg_vs_time = np.zeros((no_times, no_bins), dtype = np.float64)
    g_ln_r_num_std_vs_time = np.zeros((no_times, no_bins), dtype = np.float64)
    
    bins_mass_vs_time = np.zeros((no_times,no_bins+1),dtype=np.float64)
    bins_mass_width_vs_time = np.zeros((no_times,no_bins),dtype=np.float64)
    bins_rad_width_log_vs_time = np.zeros((no_times,no_bins),dtype=np.float64)
    bins_mass_centers = []
    bins_rad_centers = []
    
    m_max_vs_time = np.zeros(no_times,dtype=np.float64)
    m_min_vs_time = np.zeros(no_times,dtype=np.float64)
    bin_factors_vs_time = np.zeros(no_times,dtype=np.float64)
    
    moments_vs_time = np.zeros((no_times,4,no_sims),dtype=np.float64)
    
    # factors for numerical stability (s.b.)
    last_bin_factor = 1.0001
    first_bin_factor = 0.9999
    
    for time_n, masses in enumerate(masses_vs_time_T):
        xis = xis_vs_time_T[time_n]
        masses_sampled = np.concatenate(masses)
        xis_sampled = np.concatenate(xis)
    
        m_min = masses_sampled.min()
        m_max = masses_sampled.max()
        
        # convert to microns
        R_min = mp.compute_radius_from_mass(1E18 * m_min, mass_density)
        R_max = mp.compute_radius_from_mass(1E18 * m_max, mass_density)

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
        bins_mass = np.zeros(no_bins + 1,dtype=np.float64)
        bins_mass[0] = m_min
        # bins_mass[0] = m_min / bin_factor
        for bin_n in range(1, no_bins + 1):
            bins_mass[bin_n] = bins_mass[bin_n-1] * bin_factor
        # the factors are for numerical stability:
        # to ensure that
        # m_min does not contribute to a bin smaller than the first bin and
        # m_max does not contribute to a bin larger than the last bin
        bins_mass[0] *= first_bin_factor
        bins_mass[-1] *= last_bin_factor
    
        bins_mass_vs_time[time_n] = bins_mass
        # convert to microns
        bins_rad = mp.compute_radius_from_mass_vec(1E18 * bins_mass, mass_density)
        bins_mass_log = np.log(bins_mass)
        bins_rad_log = np.log(bins_rad)
    
        bins_mass_width = (bins_mass[1:]-bins_mass[:-1])
        bins_rad_width_log = (bins_rad_log[1:]-bins_rad_log[:-1])
        bins_mass_width_vs_time[time_n] = bins_mass_width
        bins_rad_width_log_vs_time[time_n] = bins_rad_width_log
        
        # different definitions of the "bin_centers"
        # define centers on lin scale
        bins_mass_center_lin = 0.5 * (bins_mass[:-1] + bins_mass[1:])
        bins_rad_center_lin = 0.5 * (bins_rad[:-1] + bins_rad[1:])
    
        # define centers on the logarithmic scale
        bins_mass_center_log = np.exp(0.5 *
                                      (bins_mass_log[:-1] + bins_mass_log[1:]))
        bins_rad_center_log = np.exp(0.5 *
                                     (bins_rad_log[:-1] + bins_rad_log[1:]))
    
        # set the bin "mass centers" at the right spot such that
        # f_avg_i in bin in = f(mm_i), where mm_i is the "mass center"
        m_avg = masses_sampled.sum() / xis_sampled.sum()
        bins_mass_center_exact = bins_mass[:-1] \
                                 + m_avg * np.log(bins_mass_width\
              / (m_avg * (1-np.exp(-bins_mass_width/m_avg))))
        # convert to microns
        bins_rad_center_exact =\
            mp.compute_radius_from_mass_vec(1E18*bins_mass_center_exact,
                                         mass_density)
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
            # convert to microns
            rad = mp.compute_radius_from_mass_vec(1E18*mass, mass_density)
            f_m_num.append(np.histogram(mass, bins_mass,
                                        weights=xis[sim_n])[0] \
                           / (bins_mass_width * dV))
            g_m_num.append(np.histogram(mass, bins_mass,
                                        weights=xis[sim_n]*mass)[0] \
                           / (bins_mass_width * dV))
    
            # build g_ln_r = 3 * m * g_m DIRECTLY from data
            g_ln_r_num.append( np.histogram(rad, bins_rad,
                                            weights=xis[sim_n]*mass)[0] \
                               / (bins_rad_width_log * dV) )
    
            moments_vs_time[time_n,0,sim_n] = xis[sim_n].sum() / dV
            for n in range(1,4):
                moments_vs_time[time_n,n,sim_n] = \
                    np.sum(xis[sim_n]*mass**n) / dV
        
        f_m_num_avg_vs_time[time_n] = np.average(f_m_num, axis=0)
        f_m_num_std_vs_time[time_n] = \
            np.std(f_m_num, axis=0, ddof=1) / np.sqrt(no_sims)
        g_m_num_avg_vs_time[time_n] = np.average(g_m_num, axis=0)
        g_m_num_std_vs_time[time_n] = \
            np.std(g_m_num, axis=0, ddof=1) / np.sqrt(no_sims)
        g_ln_r_num_avg_vs_time[time_n] = np.average(g_ln_r_num, axis=0)
        g_ln_r_num_std_vs_time[time_n] = \
            np.std(g_ln_r_num, axis=0, ddof=1) / np.sqrt(no_sims)
    
    moments_vs_time_avg = np.average(moments_vs_time, axis=2)
    moments_vs_time_std = np.std(moments_vs_time, axis=2, ddof=1) \
                          / np.sqrt(no_sims)
    
    moments_vs_time_Unt = np.zeros_like(moments_vs_time_avg)
    
    for time_n in range(no_times):
        for n in range(4):
            moments_vs_time_Unt[time_n,n] =\
                math.log(bin_factors_vs_time[time_n]) / 3.0 \
                * np.sum( g_ln_r_num_avg_vs_time[time_n]
                                    * (bins_mass_centers[time_n][1])**(n-1) )
    
    fn_ending = f"no_sims_{no_sims}_no_bins_{no_bins}"
    
    np.save(load_dir + f"moments_vs_time_avg_" + fn_ending + ".npy",
            moments_vs_time_avg)
    np.save(load_dir + f"moments_vs_time_std_" + fn_ending + ".npy",
            moments_vs_time_std)
    np.save(load_dir + f"f_m_num_avg_vs_time_" + fn_ending + ".npy",
            f_m_num_avg_vs_time)
    np.save(load_dir + f"f_m_num_std_vs_time_" + fn_ending + ".npy",
            f_m_num_std_vs_time)
    np.save(load_dir + f"g_m_num_avg_vs_time_" + fn_ending + ".npy",
            g_m_num_avg_vs_time)
    np.save(load_dir + f"g_m_num_std_vs_time_" + fn_ending + ".npy",
            g_m_num_std_vs_time)
    np.save(load_dir + f"g_ln_r_num_avg_vs_time_" + fn_ending + ".npy",
            g_ln_r_num_avg_vs_time)
    np.save(load_dir + f"g_ln_r_num_std_vs_time_" + fn_ending + ".npy",
            g_ln_r_num_std_vs_time)
    np.save(load_dir + f"bins_mass_centers_" + fn_ending + ".npy",
            bins_mass_centers)
    np.save(load_dir + f"bins_rad_centers_" + fn_ending + ".npy",
            bins_rad_centers)

#%% PLOT moments vs time and (f_m, g_m, g_ln_R) vs time for given kappa
    
def plot_for_given_kappa(kappa, eta, eta_threshold,
                         dt, no_sims, start_seed, no_bins,
                         kernel_name, kernel_method, gen_method, bin_method,
                         moments_ref, times_ref, load_dir):
    save_times = np.load(load_dir + f"save_times_{start_seed}.npy")
    no_times = len(save_times)
    
    bins_mass_centers = \
        np.load(load_dir + f"bins_mass_centers_"
                + f"no_sims_{no_sims}_no_bins_{no_bins}.npy")
    bins_rad_centers = np.load(load_dir + f"bins_rad_centers_" +
                               f"no_sims_{no_sims}_no_bins_{no_bins}.npy")

    f_m_num_avg_vs_time = np.load(load_dir + f"f_m_num_avg_vs_time_"
                                  + f"no_sims_{no_sims}_no_bins_{no_bins}.npy")
    g_m_num_avg_vs_time = np.load(load_dir + f"g_m_num_avg_vs_time_"
                                  + f"no_sims_{no_sims}_no_bins_{no_bins}.npy")
    g_ln_r_num_avg_vs_time = \
        np.load(load_dir + f"g_ln_r_num_avg_vs_time_"
                + f"no_sims_{no_sims}_no_bins_{no_bins}.npy")

    moments_vs_time_avg = np.load(load_dir + f"moments_vs_time_avg_"
                                  + f"no_sims_{no_sims}_no_bins_{no_bins}.npy")

    fig_name = "fm_gm_glnr_vs_t"
    fig_name += f"_kappa_{kappa}_dt_{int(dt)}_"
    fig_name += f"no_sims_{no_sims}_no_bins_{no_bins}.png"
    no_rows = 3
    fig, axes = plt.subplots(nrows=no_rows, figsize=(10,8*no_rows))
    
    time_every = 4
    ax = axes[0]
    for time_n in range(no_times)[::time_every]:
        ax.plot(bins_mass_centers[time_n][0], f_m_num_avg_vs_time[time_n],
                label = f"t={save_times[time_n]:.0f}")
        ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("mass (kg)")
    ax.set_ylabel(r"$f_m$ $\mathrm{(kg^{-1} \, m^{-3})}$")
    if kernel_name == "Golovin":
        ax.set_xticks( np.logspace(-15,-5,11) )
        ax.set_yticks( np.logspace(5,21,17) )
    ax.grid()
    
    ax = axes[1]
    for time_n in range(no_times)[::time_every]:
        ax.plot(bins_mass_centers[time_n][0], g_m_num_avg_vs_time[time_n])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("mass (kg)")
    ax.set_ylabel(r"$g_m$ $\mathrm{(m^{-3})}$")
    if kernel_name == "Golovin":
        ax.set_xticks( np.logspace(-15,-5,11) )
        ax.set_yticks( np.logspace(-2,9,12) )
    ax.grid()
    
    ax = axes[2]
    for time_n in range(no_times)[::time_every]:
        ax.plot(bins_rad_centers[time_n][0],
                g_ln_r_num_avg_vs_time[time_n]*1000.0)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("radius $\mathrm{(\mu m)}$")
    ax.set_ylabel(r"$g_{\ln(r)}$ $\mathrm{(g \; m^{-3})}$")
    if kernel_name == "Golovin":
        ax.set_xticks( np.logspace(0,3,4) )
        ax.set_xlim([1.0,2.0E3])
        ax.set_ylim([1.0E-4,10.0])
    elif kernel_name == "Long_Bott":
        ax.set_xlim([1.0,5.0E3])
        ax.set_ylim([1.0E-4,10.0])
    ax.grid(which="major")
    
    for ax in axes:
        ax.tick_params(which="both", bottom=True, top=True,
                       left=True, right=True)

    fig.suptitle(f"dt={dt}, kappa={kappa}, eta={eta:.0e} ({eta_threshold}), "
                 + f"no_sims={no_sims}, no_bins={no_bins}\n"
                 + f"gen_method={gen_method}, kernel={kernel_name}, "
                 + f"kernel_method={kernel_method}, bin_method={bin_method}")
    fig.tight_layout()
    plt.subplots_adjust(top=0.95)
    fig.savefig(load_dir + fig_name)
    
    ### plot moments vs time
    fig_name = "moments_vs_time"
    fig_name += f"_kappa_{kappa}_dt_{int(dt)}_"
    fig_name += f"no_sims_{no_sims}_no_bins_{no_bins}.png"
    no_rows = 4
    fig, axes = plt.subplots(nrows=no_rows, figsize=(10,5*no_rows))
    for i,ax in enumerate(axes):
        ax.plot(save_times/60, moments_vs_time_avg[:,i],"x-")
        if i != 1:
            ax.set_yscale("log")
        ax.grid()
        ax.set_xticks(save_times/60)
        ax.set_xlim([save_times[0]/60, save_times[-1]/60])
        ax.tick_params(which="both", bottom=True, top=True,
                       left=True, right=True)
    if kernel_name == "Golovin":
        axes[0].set_yticks( [1.0E6,1.0E7,1.0E8,1.0E9] )
        axes[2].set_yticks( np.logspace(-15,-9,7) )
        axes[3].set_yticks( np.logspace(-26,-15,12) )
    
    axes[3].set_xlabel("time (min)")

    axes[0].set_ylabel("moment 0 (DNC)")
    axes[1].set_ylabel("moment 1 (LMC)")
    axes[2].set_ylabel("moment 2")
    axes[3].set_ylabel("moment 3")
    
    for mom_n in range(4):
        axes[mom_n].plot(times_ref/60, moments_ref[mom_n], "o")
    fig.suptitle(f"dt={dt}, kappa={kappa}, eta={eta:.0e} ({eta_threshold}), "
                 + f"no_sims={no_sims}, no_bins={no_bins}\n"
                 + f"gen_method={gen_method}, kernel={kernel_name}, "
                 + f"kernel_method={kernel_method}, bin_method={bin_method}")        
    fig.tight_layout()
    plt.subplots_adjust(top=0.95)
    fig.savefig(load_dir + fig_name)
    plt.close("all")

#%% PLOT MOMENTS VS TIME for several kappa

# TTFS, LFS, TKFS: title, labels, ticks font size
def plot_moments_vs_time_kappa_var(kappa_list, eta, dt, no_sims, no_bins,
                                   kernel_name, gen_method,
                                   dist, start_seed,
                                   moments_ref, times_ref,
                                   sim_data_path,
                                   result_path_add,
                                   fig_dir, TTFS, LFS, TKFS):
    no_kappas = len(kappa_list)
    
    fig_name = f"moments_vs_time_kappa_var_{no_kappas}"
    fig_name += f"_dt_{int(dt)}_no_sims_{no_sims}.pdf"
    no_rows = 4
    
    fig, axes = plt.subplots(nrows=no_rows, figsize=(10,6*no_rows),
                             sharex=True)
    
    for kappa_n,kappa in enumerate(kappa_list):
        load_dir = sim_data_path + result_path_add \
                   + f"kappa_{kappa}/dt_{int(dt)}/"
        save_times = np.load(load_dir + f"save_times_{start_seed}.npy")
        moments_vs_time_avg = np.load(load_dir +
                                      f"moments_vs_time_avg_no_sims_"
                                      + f"{no_sims}_no_bins_{no_bins}.npy")
        
        if kappa_n < 10: fmt = "x-"
        else: fmt = "x--"            
        
        for i,ax in enumerate(axes):
            ax.plot(save_times/60, moments_vs_time_avg[:,i],
                    fmt, label=f"{kappa}")

    for i,ax in enumerate(axes):
        ax.plot(times_ref/60, moments_ref[i],
                "o", c = "k",fillstyle='none', markersize = 8,
                mew=1.0, label="Wang")
        if i != 1:
            ax.set_yscale("log")
        ax.grid()
        if i != 1:
            ax.legend(fontsize=TKFS)
        if i == 1:
            ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.05),
                      fontsize=TKFS)
        ax.set_xticks(save_times/60)
        ax.set_xlim([save_times[0]/60, save_times[-1]/60])
        ax.tick_params(which="both", bottom=True, top=True,
                       left=True, right=True
                       )
        ax.tick_params(axis='both', which='major', labelsize=TKFS,
                       width=2, size=10)
        ax.tick_params(axis='both', which='minor', labelsize=TKFS,
                       width=1, size=6)
    axes[-1].set_xlabel("time (min)",fontsize=LFS)
    axes[0].set_ylabel(r"$\lambda_0$ = DNC $(\mathrm{m^{-3}})$ ",
                       fontsize=LFS)
    axes[1].set_ylabel(r"$\lambda_1$ = LWC $(\mathrm{kg \, m^{-3}})$ ",
                       fontsize=LFS)
    axes[2].set_ylabel(r"$\lambda_2$ $(\mathrm{kg^2 \, m^{-3}})$ ",
                       fontsize=LFS)
    axes[3].set_ylabel(r"$\lambda_3$ $(\mathrm{kg^3 \, m^{-3}})$ ",
                       fontsize=LFS)
    if kernel_name == "Golovin":
        axes[0].set_yticks([1.0E6,1.0E7,1.0E8,1.0E9])
        axes[2].set_yticks( np.logspace(-15,-9,7) )
        axes[3].set_yticks( np.logspace(-26,-15,12) )
    elif kernel_name == "Long_Bott":
        axes[0].set_yticks([1.0E6,1.0E7,1.0E8])
        axes[0].set_ylim([1.0E6,4.0E8])
        axes[2].set_yticks( np.logspace(-15,-7,9) )
        axes[2].set_ylim([1.0E-15,1.0E-7])
        axes[3].set_yticks( np.logspace(-26,-11,16) )
        axes[3].set_ylim([1.0E-26,1.0E-11])
        if len(times_ref) > 10:
            axes[3].set_xticks(save_times[::2]/60)
    elif kernel_name == "Hall_Bott":
        axes[0].set_yticks([1.0E8])
        axes[0].set_ylim([6.0E7,4.0E8])
        axes[2].set_yticks( np.logspace(-15,-7,9) )
        axes[2].set_ylim([1.0E-15,1.0E-7])
        axes[3].set_yticks( np.logspace(-26,-11,16) )
        axes[3].set_ylim([1.0E-26,1.0E-11])
        if len(times_ref) > 10:
            axes[3].set_xticks(save_times[::2]/60)
    title=\
        f"Moments of the distribution for various $\kappa$ (see legend)\n" \
        + f"dt={dt:.1e}, eta={eta:.0e}, r_critmin=0.6, no_sims={no_sims}, " \
        + f"gen_method={gen_method}, kernel={kernel_name}"
    fig.suptitle(title, fontsize=TTFS, y = 0.997)
    fig.tight_layout()
    plt.subplots_adjust(top=0.965)
    fig.savefig(fig_dir + fig_name)
    plt.close("all")

#%% plot moments 0, 2 and 3 of the mass distribution vs time for several kappa
# in the same plot. formatting as in the GMD publication    
    
def plot_moments_vs_time_kappa_var_paper(kappa_list, eta, dt, no_sims, no_bins,
                                         kernel_name, gen_method,
                                         dist, start_seed,
                                         moments_ref, times_ref,
                                         sim_data_path,
                                         result_path_add,
                                         figsize, figname,
                                         TTFS, LFS, TKFS):
    no_rows = 3
    
    fig, axes = plt.subplots(nrows=no_rows, figsize=(figsize), sharex=True)
    
    for kappa_n,kappa in enumerate(kappa_list):
        load_dir = sim_data_path \
                   + result_path_add + f"kappa_{kappa}/dt_{int(dt)}/"
        save_times = np.load(load_dir + f"save_times_{start_seed}.npy")
        moments_vs_time_avg = \
            np.load(load_dir
                    + f"moments_vs_time_avg_"
                    + f"no_sims_{no_sims}_no_bins_{no_bins}.npy")
        # estimated errorbars can also be shown
        # load moments_vs_time_std in this case
#        moments_vs_time_std = \
#            np.load(load_dir
#                    + f"moments_vs_time_std_"
#                    + f"no_sims_{no_sims}_no_bins_{no_bins}.npy")
        
        if kappa_n < 10: fmt = "-"
        else: fmt = "x--"            
        
        for ax_n,i in enumerate((0,2,3)):
            if kappa*5 < 100000:
                lab = f"{kappa*5}"
            else:
                lab = f"{float(kappa*5):.2}"
            axes[ax_n].plot(save_times/60, moments_vs_time_avg[:,i],fmt,
                            label=f"\\num{{{lab}}}",
                            lw=1.2,
                            ms=5,                        
                            mew=0.8,
                            zorder=97)
            
            ### estimated errorbars can also be shown
            ### load moments_vs_time_std in this case (above)
#            axes[ax_n].errorbar(save_times/60,
#                                moments_vs_time_avg[:,i],
#                                moments_vs_time_std[:,i],
#                                fmt=fmt,
#                                label=f"\\num{{{lab}}}",
#                                lw=1.2,
#                                ms=5,                        
#                                mew=0.5,
#                                elinewidth=1.0,
#                                zorder=98)
 
    if kernel_name == "Golovin":
        DNC = 296799076.3
        LWC = 1E-3
        bG = 1.5
            
    for ax_n,i in enumerate((0,2,3)):
        ax = axes[ax_n]
        if kernel_name == "Golovin":
            
            fmt = "o"
            # load times_ref from 'collision/...'
            # calc moments directly from analytic function
            moments_ref_i = compute_moments_Golovin(times_ref, i, DNC, LWC, bG)
        else:
            fmt = "o"
            moments_ref_i = moments_ref[i]
        ax.plot(times_ref/60, moments_ref_i,
                fmt, c = "k",
                fillstyle='none',
                linewidth = 2,
                markersize = 3, mew=0.4,
                label="Ref",
                zorder=99)
        if i != 1:
            ax.set_yscale("log")
        if kernel_name == "Long_Bott":
            ax.grid(which="major")
        else:
            ax.grid(which="both")
        if i == 2:
            if kernel_name == "Golovin":
                ax.legend(
                          ncol=2, handlelength=0.8, handletextpad=0.2,
                          columnspacing=0.5, borderpad=0.2, loc="lower left",
                          bbox_to_anchor=(0.0, 0.4)).set_zorder(100)          
            if kernel_name == "Long_Bott":
                ax.legend(
                          ncol=2, handlelength=0.8, handletextpad=0.2,
                          columnspacing=0.5, borderpad=0.2, loc="lower left",
                          bbox_to_anchor=(0.0, 0.6)).set_zorder(100)            
            if kernel_name == "Hall_Bott":
                ax.legend(
                          ncol=2, handlelength=0.8, handletextpad=0.2,
                          columnspacing=0.5, borderpad=0.2, loc="lower left",
                          bbox_to_anchor=(0.0, 0.5)).set_zorder(100)            
        ax.set_xticks(save_times[::2]/60)
        ax.set_xlim([save_times[0]/60, save_times[-1]/60])
        ax.tick_params(which="both", bottom=True, top=True,
                       left=True, right=True
                       )
        ax.tick_params(axis='both', which='major', labelsize=TKFS,
                       width=0.8, size=3)
        ax.tick_params(axis='both', which='minor', labelsize=TKFS,
                       width=0.6, size=2, labelleft=False)
    axes[-1].set_xlabel("Time (min)",fontsize=LFS)
    axes[0].set_ylabel(r"$\lambda_0$ = DNC $(\mathrm{m^{-3}})$ ",
                       fontsize=LFS)
    axes[1].set_ylabel(r"$\lambda_2$ $(\mathrm{kg^2 \, m^{-3}})$ ",
                       fontsize=LFS)
    axes[2].set_ylabel(r"$\lambda_3$ $(\mathrm{kg^3 \, m^{-3}})$ ",
                       fontsize=LFS)
    if kernel_name == "Golovin":
        axes[0].set_yticks([1.0E6,1.0E7,1.0E8,1.0E9])
        axes[0].set_yticks([5.0E6,5.0E7,5.0E8], minor=True)
        axes[1].set_yticks( np.logspace(-15,-9,7) )
        axes[2].set_yticks( np.logspace(-26,-15,12)[::2] )
        axes[2].set_yticks( np.logspace(-26,-15,12)[1::2], minor=True )
    elif kernel_name == "Long_Bott":
        axes[0].set_yticks([1.0E6,1.0E7,1.0E8])
        axes[0].set_yticks(
                np.concatenate((
                        np.linspace(2E6,9E6,8),
                        np.linspace(2E7,9E7,8),
                        np.linspace(2E8,3E8,2),
                                )),
                minor=True)
        axes[0].set_ylim([1.0E6,4.0E8])
        axes[1].set_yticks( np.logspace(-15,-7,9) )
        axes[1].set_ylim([1.0E-15,1.0E-7])
        axes[2].set_yticks( np.logspace(-26,-11,16)[::2])
        axes[2].set_yticks( np.logspace(-26,-11,16)[1::2],minor=True)
        axes[2].set_ylim([1.0E-26,1.0E-11])
    elif kernel_name == "Hall_Bott":
        axes[0].set_yticks([7E7,8E7,9E7,1E8,2.0E8,3E8])

        axes[0].set_ylim([7.0E7,4.0E8])
        axes[0].yaxis.set_ticklabels(
                [r'$7\times10^7$','','',r'$1\times10^8$',
                r'$2\times 10^8$',r'$3\times10^8$','','','','',
                r'$2\times 10^5$','','',r'$5\times10^5$','','','','',
                ])        
        axes[1].set_yticks( np.logspace(-15,-8,8) )
        axes[1].set_ylim([1.0E-15,1.0E-8])
        axes[2].set_yticks( np.logspace(-26,-12,8) )
        axes[2].set_ylim([1.0E-26,1.0E-12])
    
    if kernel_name == "Hall_Bott":
        xpos_ = -0.19
        ypos_ = 0.86
    elif kernel_name == "Long_Bott":
        xpos_ = -0.14
        ypos_ = 0.86
    elif kernel_name == "Golovin":
        xpos_ = -0.14
        ypos_ = 0.86
    fig.text(xpos_, ypos_ , r"\textbf{(c)}", fontsize=LFS)    
    
    fig.savefig(figname,
                bbox_inches = 'tight',
                pad_inches = 0.04
                )      

    plt.close("all")        

#%% Plot g_lnR vs time for a single kappa
    
def plot_g_ln_R_for_given_kappa(kappa,
                                eta, dt, no_sims, start_seed, no_bins,
                                DNC0, m_mean,
                                kernel_name, gen_method, time_idx,
                                load_dir,
                                figsize, figname,
                                LFS):
    fn_end = f"no_sims_{no_sims}_no_bins_{no_bins}.npy"
    bG = 1.5 # K = b * (m1 + m2) # b in m^3/(fg s)
    save_times = np.load(load_dir + f"save_times_{start_seed}.npy")
    bins_mass_centers = np.load(load_dir
                                + f"bins_mass_centers_" + fn_end)
    bins_rad_centers = np.load(load_dir
                               + f"bins_rad_centers_" + fn_end)

    add_masses = 4
    
    g_ln_r_num_avg_vs_time = np.load(load_dir
                                     + f"g_ln_r_num_avg_vs_time_" + fn_end)
    g_ln_r_num_std_vs_time = np.load(load_dir
                                     + f"g_ln_r_num_std_vs_time_" + fn_end)

    no_rows = 1
    fig, axes = plt.subplots(nrows=no_rows, figsize=figsize)
    
    ax = axes
    ax.set_xscale("log", nonposx="mask")    
    ax.set_yscale("log", nonposy="mask")    
    for time_n in time_idx:
        mask = g_ln_r_num_avg_vs_time[time_n]*1000.0 > 1E-6
        ax.plot(bins_rad_centers[time_n][0][mask],
                g_ln_r_num_avg_vs_time[time_n][mask]*1000.0,
                label = f"{int(save_times[time_n]//60)}", zorder=50)
        
        above_curve = g_ln_r_num_avg_vs_time[time_n]*1000.0\
                      + g_ln_r_num_std_vs_time[time_n]*1000.0
        above_curve = \
            np.where(above_curve <= 1E-4, 1E-4, above_curve)
        
        below_curve = g_ln_r_num_avg_vs_time[time_n]*1000.0\
                      - g_ln_r_num_std_vs_time[time_n]*1000.0
        below_curve = \
            np.where(below_curve <= 1E-4, 1E-4, below_curve)
        
        ax.fill_between(bins_rad_centers[time_n][0][mask],
                        below_curve[mask],
                        above_curve[mask],
                        alpha=0.4, lw=1
                        )              
    ax.set_prop_cycle(None)
    for j,time_n in enumerate(time_idx):
        if kernel_name == "Golovin":
            scale_g=1000.      
            no_bins_ref = 2*no_bins
            ref_masses = np.zeros(no_bins_ref + add_masses)
            
            bin_factor = np.sqrt(bins_mass_centers[time_n][0][-1] \
                                 /bins_mass_centers[time_n][0][-2])
            ref_masses[0] = bins_mass_centers[time_n][0][0]

            for n in range(1,no_bins_ref + add_masses):
                ref_masses[n] = ref_masses[n-1]*bin_factor
            f_m_golo = dist_vs_time_golo_exp(ref_masses,
                                             save_times[time_n],m_mean,DNC0,bG)
            g_ln_r_golo = f_m_golo * 3 * ref_masses**2
            g_ln_r_ref = g_ln_r_golo
            ref_radii = 1E6 * (3. * ref_masses / (4. * math.pi * 1E3))**(1./3.)
            
        else:
            scale_g = 1.
            dpref = f"collision/ref_data/{kernel_name}/"
            ref_radii = \
                np.loadtxt(dpref + "Wang_2007_radius_bin_centers.txt")[j][::5]
            g_ln_r_ref = \
                np.loadtxt(dpref + "Wang_2007_g_ln_R.txt")[j][::5]
        ax.plot(ref_radii,
                g_ln_r_ref*scale_g,
                "o",
                fillstyle='none',
                linewidth = 2,
                markersize = 3, mew=0.4)                
    
    ax.set_xlabel("Radius ($\si{\micro\meter}$)")
    ax.set_ylabel(r"$g_{\ln(R)}$ $\mathrm{(g \; m^{-3})}$")
    if kernel_name == "Golovin":
        ax.set_xticks( np.logspace(0,3,4) )
        ax.set_xlim([1.0,2.0E3])
        ax.set_ylim([1.0E-4,10.0])
    elif kernel_name == "Long_Bott":
        ax.set_xlim([1.0,5.0E3])
        ax.set_ylim([1.0E-4,10.0])
    elif kernel_name == "Hall_Bott":
        ax.set_xlim([1.0,5.0E3])
        ax.set_yticks( np.logspace(-4,2,7) )
        ax.set_ylim([1.0E-4,10.0])        
    ax.grid(which="major")
    ax.legend(ncol=7, handlelength=0.8, handletextpad=0.2,
              columnspacing=0.8, borderpad=0.2, loc="upper center")
    ax.tick_params(which="both", bottom=True, top=True, left=True, right=True)

    fig.savefig(figname,
                bbox_inches = 'tight',
                pad_inches = 0.065
                )    
    plt.close("all")    

#%% Plot g_ln_R vs time for two chosen kappa 
    
def plot_g_ln_R_kappa_compare(kappa1, kappa2,
                              eta, dt, no_sims, start_seed, no_bins,
                              DNC0, m_mean,
                              kernel_name, gen_method, time_idx,
                              load_dir_k1, load_dir_k2,
                              figsize, figname_compare, LFS):
    bG = 1.5 # K = b * (m1 + m2) # b in m^3/(fg s)
    save_times = np.load(load_dir_k1 + f"save_times_{start_seed}.npy")

    no_rows = 2
    fig, axes = plt.subplots(nrows=no_rows, figsize=figsize)
    
    load_dirs = (load_dir_k1, load_dir_k2)
    
    fn_end = f"no_sims_{no_sims}_no_bins_{no_bins}.npy"
    
    for n,kappa in enumerate((kappa1,kappa2)):
        load_dir = load_dirs[n]
        bins_mass_centers = np.load(load_dir + f"bins_mass_centers_" + fn_end)
        bins_rad_centers = np.load(load_dir + f"bins_rad_centers_"  + fn_end)
    
        add_masses = 4
        g_ln_r_num_avg_vs_time = \
            np.load(load_dir + f"g_ln_r_num_avg_vs_time_" + fn_end)
        g_ln_r_num_std_vs_time = \
            np.load(load_dir + f"g_ln_r_num_std_vs_time_"  + fn_end)
        ax = axes[n]
        
        ax.set_xscale("log", nonposx="mask")    
        ax.set_yscale("log", nonposy="mask")                
    
        for time_n in time_idx:
            mask = g_ln_r_num_avg_vs_time[time_n]*1000.0 > 1E-6
            ax.plot(bins_rad_centers[time_n][0][mask],
                    g_ln_r_num_avg_vs_time[time_n][mask]*1000.0,
                    label = f"{int(save_times[time_n]//60)}", zorder=50)
            
            above_curve = g_ln_r_num_avg_vs_time[time_n]*1000.0\
                          + g_ln_r_num_std_vs_time[time_n]*1000.0
            above_curve = \
                np.where(above_curve <= 1E-4, 1E-4, above_curve)
            
            below_curve = g_ln_r_num_avg_vs_time[time_n]*1000.0\
                          - g_ln_r_num_std_vs_time[time_n]*1000.0
            below_curve = \
                np.where(below_curve <= 1E-4, 1E-4, below_curve)
            
            ax.fill_between(bins_rad_centers[time_n][0][mask],
                            below_curve[mask],
                            above_curve[mask],
                            alpha=0.4, lw=1
                            )                          
        ax.set_prop_cycle(None)
        for j,time_n in enumerate(time_idx):
            if kernel_name == "Golovin":
                scale_g=1000.      
                no_bins_ref = 2*no_bins
                ref_masses = np.zeros(no_bins_ref+add_masses)
                
                bin_factor = np.sqrt(bins_mass_centers[time_n][0][-1] \
                                     /bins_mass_centers[time_n][0][-2])
                ref_masses[0] = bins_mass_centers[time_n][0][0]
    
                for n in range(1,no_bins_ref+add_masses):
                    ref_masses[n] = ref_masses[n-1]*bin_factor
                f_m_golo = \
                    dist_vs_time_golo_exp(ref_masses, save_times[time_n],
                                          m_mean, DNC0, bG)
                g_ln_r_golo = f_m_golo * 3 * ref_masses**2
                g_ln_r_ref = g_ln_r_golo
                ref_radii = 1E6 * (3. * ref_masses \
                                   / (4. * math.pi * 1E3))**(1./3.)                
            else:
                scale_g=1.
                dpref = f"collision/ref_data/{kernel_name}/"
                ref_radii = \
                    np.loadtxt(dpref
                               + "Wang_2007_radius_bin_centers.txt")[j][::5]
                g_ln_r_ref = np.loadtxt(dpref + "Wang_2007_g_ln_R.txt")[j][::5]
                
            fmt="o"    
            ax.plot(ref_radii, g_ln_r_ref*scale_g,
                    fmt,
                    fillstyle='none',
                    linewidth = 2,
                    markersize = 2.3,
                    mew=0.3, zorder=40)             
        if kernel_name == "Golovin":
            ax.set_xticks( np.logspace(0,3,4) )
            ax.set_yticks( np.logspace(-4,0,5) )
            ax.set_xlim([1.0,2.0E3])
            ax.set_ylim([1.0E-4,10.0])
        elif kernel_name == "Long_Bott":
            ax.set_xlim([1.0,5.0E3])
            ax.set_yticks( np.logspace(-4,2,7) )
            ax.set_ylim([1.0E-4,10.0])
        elif kernel_name == "Hall_Bott":
            ax.set_xlim([1.0,5.0E3])
            ax.set_yticks( np.logspace(-4,2,7) )
            ax.set_ylim([1.0E-4,10.0])
        ax.grid(which="major")
        
    axes[1].set_xlabel("Radius ($\si{\micro\meter}$)")
    axes[0].set_ylabel(r"$g_{\ln(R)}$ $\mathrm{(g \; m^{-3})}$")            
    axes[1].set_ylabel(r"$g_{\ln(R)}$ $\mathrm{(g \; m^{-3})}$")            
    
    axes[0].legend(ncol=7, handlelength=0.8, handletextpad=0.2,
                      columnspacing=0.8, borderpad=0.15, loc="upper center",
                      bbox_to_anchor=(0.5,1.02))
    axes[0].tick_params(which="both", bottom=True, top=True,
                       left=True, right=True, labelbottom=False)
    axes[1].tick_params(which="both", bottom=True, top=True,
                       left=True, right=True)
    xpos_ = -0.054
    ypos_ = 0.86
    fig.text(xpos_, ypos_ , r"\textbf{(a)}", fontsize=LFS)    
    fig.text(xpos_, ypos_*0.51, r"\textbf{(b)}", fontsize=LFS)    
    fig.savefig(figname_compare,
                bbox_inches = 'tight',
                pad_inches = 0.05
                )    
    plt.close("all")        