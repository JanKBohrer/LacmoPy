#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 16:42:15 2019

@author: jdesk
"""

#%% IMPORTS AND DEFS

import os
import math
import numpy as np
from numba import njit

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import constants as c

from microphysics import compute_radius_from_mass
from microphysics import compute_radius_from_mass_jit

from AON_Unt_algo_Jan import simulate_collisions
from AON_Unt_algo_Jan import simulate_collisions_kernel_grid


#%% AON Method from Unterstrasser

OS = "LinuxDesk"
# OS = "MacOS"

kernel_method = "grid"
#kernel_method = "exact"

#args = [1,0,0,0]
args = [0,1,1,1]
# args = [1,1,1,1]

# simulate = True
# analyze = True
# plotting = True
# plot_moments_kappa_var = True

simulate = bool(args[0])
analyze = bool(args[1])
plotting = bool(args[2])
plot_moments_kappa_var = bool(args[3])

dist = "expo"

# kappa = 40

# kappa = 800

#kappa_list=[400]
#kappa_list=[5]
#kappa_list=[5,10,20,40]
# kappa_list=[5,10,20,40,60,100,200]
# kappa_list=[5,10,20,40,60,100,200,400]
kappa_list=[5,10,20,40,60,100,200,400]
# kappa_list=[5,10,20,40,60,100,200,400,600,800]
# kappa_list=[600]
# kappa_list=[800]

eta = 1.0E-9

# no_sims = 163
# no_sims = 450
no_sims = 500
# no_sims = 50
# no_sims = 250

# no_sims = 94

start_seed = 3711

# start_seed = 4523

# start_seed = 4127
# start_seed = 4107
# start_seed = 4385
# start_seed = 3811

no_bins = 50

gen_method = "SinSIP"
# kernel = "Golovin"
kernel = "Long_Bott"

bin_method = "auto_bin"

# dt = 1.0
dt = 10.0
# dt = 20.0
dV = 1.0
# dt_save = 40.0
dt_save = 300.0
# t_end = 200.0
t_end = 3600.0

if OS == "MacOS":
    sim_data_path = "/Users/bohrer/sim_data/"
elif OS == "LinuxDesk":
    sim_data_path = "/mnt/D/sim_data_my_kernel_grid_strict_thresh/"
#    sim_data_path = "/mnt/D/sim_data/"

# ensemble_dir =\
#     f"/mnt/D/sim_data/col_box_mod/ensembles/{gen_method}/kappa_{kappa}/"
    # f"/mnt/D/sim_data/col_box_mod/ensembles/{gen_method}/eta_{eta:.0e}/"

# seed = start_seed
# radii = compute_radius_from_mass(masses*1.0E18,
#                                  c.mass_density_water_liquid_NTP)

if kernel_method == "grid":
    mass_grid = np.load( sim_data_path + f"col_box_mod/results/{dist}/{kernel}/kernel_data/mass_grid_out.npy" )
    kernel_grid = np.load( sim_data_path + f"col_box_mod/results/{dist}/{kernel}/kernel_data/kernel_grid.npy" )
    m_kernel_low = mass_grid[0]
    bin_factor_m = mass_grid[1] / mass_grid[0]

### SIMULATE COLLISIONS
if simulate:
    for kappa in kappa_list:
        print("simulation for kappa =", kappa)
        # SIP ensembles are already stored in directory
        # LINUX desk
        ensemble_dir =\
        sim_data_path + f"col_box_mod/ensembles/{dist}/{gen_method}/eta_{eta:.0e}/kappa_{kappa}/"
        save_dir =\
        sim_data_path + f"col_box_mod/results/{dist}/{kernel}/{gen_method}/eta_{eta:.0e}/kappa_{kappa}/dt_{int(dt)}/"
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
            if kernel_method == "grid":
                simulate_collisions_kernel_grid(
                    xis, masses, dt, dV, t_end, dt_save, seed, save_dir,
                    kernel, kernel_grid, m_kernel_low, bin_factor_m)
            elif kernel_method == "exact":
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
    print("kappa, time_n, {xi_max/xi_min:.3e}, no_SIPS_avg, R_min, R_max")
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
        sim_data_path + f"col_box_mod/results/{dist}/{kernel}/{gen_method}/eta_{eta:.0e}/kappa_{kappa}/dt_{int(dt)}/"
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
        sim_data_path + f"col_box_mod/results/{dist}/{kernel}/{gen_method}/eta_{eta:.0e}/kappa_{kappa}/dt_{int(dt)}/"

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
        plt.close("all")
#%% PLOT MOMENTS VS TIME for several kappa
# plot_moments_kappa_var = True

mom0_last_time_Unt = np.array([1.0E7,5.0E6,1.8E6,1.0E6,8.0E5,
                               5.0E5,5.0E5,5.0E5,5.0E5,5.0E5])

TTFS, LFS, TKFS = 14,14,12

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

    # Wang 2007: s = 16 -> kappa = 53.151 = s * log_2(10)
    t_Wang = np.linspace(0,60,7)
    moments_vs_time_Wang = np.loadtxt(
        sim_data_path + f"col_box_mod/results/{dist}/{kernel}/Wang2007_results2.txt")
    moments_vs_time_Wang = np.reshape(moments_vs_time_Wang,(4,7)).T
    moments_vs_time_Wang[:,0] *= 1.0E6
    moments_vs_time_Wang[:,2] *= 1.0E-6
    moments_vs_time_Wang[:,3] *= 1.0E-12

    fig_dir = \
    sim_data_path + f"col_box_mod/results/{dist}/{kernel}/{gen_method}/eta_{eta:.0e}/"
    # fig_dir = \
    # f"/mnt/D/sim_data/col_box_mod/results/{kernel}/{gen_method}/"
    
    # save_times = np.load(load_dir + f"save_times_{start_seed}.npy")
    
    fig_name = "moments_vs_time_kappa_var"
    # fig_name += f"_dt_{int(dt)}_no_sims_{no_sims}.png"
    fig_name += f"_dt_{int(dt)}_no_sims_{no_sims}.png"
    no_rows = 4
    
    fig, axes = plt.subplots(nrows=no_rows, figsize=(10,6*no_rows), sharex=True)
    
    mom0_last_time = np.zeros(len(kappa_list),dtype=np.float64)
    
    for kappa_n,kappa in enumerate(kappa_list):
        load_dir = \
        sim_data_path + f"col_box_mod/results/{dist}/{kernel}/{gen_method}/eta_{eta:.0e}/kappa_{kappa}/dt_{int(dt)}/"
        # load_dir = \
        # f"/mnt/D/sim_data/col_box_mod/results/{kernel}/{gen_method}/kappa_{kappa}/dt_{int(dt)}/"
        save_times = np.load(load_dir + f"save_times_{start_seed}.npy")
        moments_vs_time_avg = np.load(load_dir + f"moments_vs_time_avg_no_sims_{no_sims}_no_bins_{no_bins}.npy")
        moments_vs_time_avg[:,1] *= 1.0E3
        mom0_last_time[kappa_n] = moments_vs_time_avg[-1,0]
        for i,ax in enumerate(axes):
            ax.plot(save_times/60, moments_vs_time_avg[:,i],"x-",label=f"{kappa}")

    for i,ax in enumerate(axes):
        ax.plot(t_Wang, moments_vs_time_Wang[:,i],
                "o", c = "k",fillstyle='none', markersize = 10, mew=3.5, label="Wang")
        if i != 1:
            ax.set_yscale("log")
        ax.grid()
        # if i ==0: ax.legend()
        if i != 1:
            ax.legend(fontsize=TKFS)
        if i == 1:
            ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.05),
                      fontsize=TKFS)
        ax.set_xticks(save_times/60)
        ax.set_xlim([save_times[0]/60, save_times[-1]/60])
        # ax.tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=False,
        #                  bottom=True, top=False, left=False, right=False)
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
    axes[1].set_ylabel(r"$\lambda_1$ = LWC $(\mathrm{g \, m^{-3}})$ ",
                       fontsize=LFS)
    axes[2].set_ylabel(r"$\lambda_2$ $(\mathrm{kg^2 \, m^{-3}})$ ",
                       fontsize=LFS)
    axes[3].set_ylabel(r"$\lambda_3$ $(\mathrm{kg^3 \, m^{-3}})$ ",
                       fontsize=LFS)
    if kernel == "Golovin":
        axes[0].set_yticks([1.0E6,1.0E7,1.0E8,1.0E9])
        axes[2].set_yticks( np.logspace(-15,-9,7) )
        axes[3].set_yticks( np.logspace(-26,-15,12) )
    elif kernel == "Long_Bott":
        # axes[0].set_yticks([1.0E6,1.0E7,1.0E8,3.0E8,4.0E8])
        axes[0].set_yticks([1.0E6,1.0E7,1.0E8])
        axes[0].set_ylim([1.0E6,4.0E8])
        # axes[1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
        axes[2].set_yticks( np.logspace(-15,-7,9) )
        axes[2].set_ylim([1.0E-15,1.0E-7])
        axes[3].set_yticks( np.logspace(-26,-11,16) )
        axes[3].set_ylim([1.0E-26,1.0E-11])
    # axes[0].plot(t_Unt,lam0_Unt, "o", c = "k")
    # axes[2].plot(t_Unt2,lam2_Unt, "o", c = "k")
        
#    for mom0_last in mom0_last_time:
#    print(mom0_last_time/mom0_last_time.min())
    print(mom0_last_time/mom0_last_time[-2])
    print(mom0_last_time_Unt/mom0_last_time_Unt.min())
    print()
    title=\
f"Moments of the distribution for various $\kappa$ (see legend)\n\
dt={dt:.1e}, eta={eta:.0e}, r_critmin=0.6, no_sims={no_sims}, \
gen_method={gen_method}, kernel=LONG"
#     title=\
# f"Moments of the distribution for various $\kappa$ (see legend)\n\
# dt={dt}, eta={eta:.0e}, no_sims={no_sims}, \
# gen_method={gen_method}, kernel=LONG"
    fig.suptitle(title, fontsize=TTFS, y = 0.997)
    fig.tight_layout()
    # fig.subplots_adjust()
    plt.subplots_adjust(top=0.965)
    fig.savefig(fig_dir + fig_name)

plt.close("all")