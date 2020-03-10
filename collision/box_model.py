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
Shima et al. 2009, Q. J. R. Meteorol. Soc. 135: 1307–1320 and
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

from microphysics import compute_radius_from_mass
from microphysics import compute_radius_from_mass_vec

import collision.kernel as ker
import collision.all_or_nothing as aon

from golovin import compute_moments_Golovin, dist_vs_time_golo_exp

#%% DEFINITIONS

### SIMULATION
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
    
### ANALYSIS OF SIM DATA

# for given kappa:
# the simulation yields masses in unit 1E-18 kg
# to compare moments etc with Unterstrasser, masses are converted to kg
def analyze_sim_data(kappa, mass_density, dV, no_sims, start_seed, no_bins, load_dir):
    # f"/mnt/D/sim_data/col_box_mod/results/{kernel_name}/{gen_method}/kappa_{kappa}/dt_{int(dt)}/"
    # f"/mnt/D/sim_data/col_box_mod/results/{kernel_name}/{gen_method}/kappa_{kappa}/dt_{int(dt)}/perm/"
    
    save_times = np.load(load_dir + f"save_times_{start_seed}.npy")
    
    seed_list = np.arange(start_seed, start_seed+no_sims*2, 2)
    
    masses_vs_time = []
    xis_vs_time = []
    for seed in seed_list:
        # convert to kg
        masses_vs_time.append(1E-18*np.load(load_dir + f"masses_vs_time_{seed}.npy"))
#        masses_vs_time.append(np.load(load_dir + f"masses_vs_time_{seed}.npy"))
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
        
        # convert to microns
        R_min = compute_radius_from_mass(1E18*m_min, mass_density)
        R_max = compute_radius_from_mass(1E18*m_max, mass_density)

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
        # convert to microns
        bins_rad = compute_radius_from_mass_vec(1E18*bins_mass, mass_density)
        bins_mass_log = np.log(bins_mass)
        bins_rad_log = np.log(bins_rad)
    
        bins_mass_width = (bins_mass[1:]-bins_mass[:-1])
#        bins_rad_width = (bins_rad[1:]-bins_rad[:-1])
        bins_rad_width_log = (bins_rad_log[1:]-bins_rad_log[:-1])
        bins_mass_width_vs_time[time_n] = bins_mass_width
        bins_rad_width_log_vs_time[time_n] = bins_rad_width_log
    
#        f_m_counts = np.histogram(masses_sampled,bins_mass)[0]
    
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
        # convert to microns
        bins_rad_center_exact =\
            compute_radius_from_mass_vec(1E18*bins_mass_center_exact,
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
            rad = compute_radius_from_mass_vec(1E18*mass, mass_density)
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
    # convert to microns
#    R_min_vs_time = compute_radius_from_mass_vec(1E18*m_min_vs_time,
#                                                 mass_density)
#    R_max_vs_time = compute_radius_from_mass_vec(1E18*m_max_vs_time,
#                                                 mass_density)
    
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
    np.save(load_dir + f"f_m_num_std_vs_time_no_sims_{no_sims}_no_bins_{no_bins}.npy", f_m_num_std_vs_time)
    np.save(load_dir + f"g_m_num_avg_vs_time_no_sims_{no_sims}_no_bins_{no_bins}.npy", g_m_num_avg_vs_time)
    np.save(load_dir + f"g_m_num_std_vs_time_no_sims_{no_sims}_no_bins_{no_bins}.npy", g_m_num_std_vs_time)
    np.save(load_dir + f"g_ln_r_num_avg_vs_time_no_sims_{no_sims}_no_bins_{no_bins}.npy", g_ln_r_num_avg_vs_time)
    np.save(load_dir + f"g_ln_r_num_std_vs_time_no_sims_{no_sims}_no_bins_{no_bins}.npy", g_ln_r_num_std_vs_time)
    np.save(load_dir + f"bins_mass_centers_{no_sims}_no_bins_{no_bins}.npy", bins_mass_centers)
    np.save(load_dir + f"bins_rad_centers_{no_sims}_no_bins_{no_bins}.npy", bins_rad_centers)    


# PLOTTING
def plot_for_given_kappa(kappa, eta, dt, no_sims, start_seed, no_bins,
                         kernel_name, gen_method, bin_method,
                         moments_ref, times_ref, load_dir):
    save_times = np.load(load_dir + f"save_times_{start_seed}.npy")
    no_times = len(save_times)
    # bins_mass_centers =
    bins_mass_centers = np.load(load_dir + f"bins_mass_centers_{no_sims}_no_bins_{no_bins}.npy")
    bins_rad_centers = np.load(load_dir + f"bins_rad_centers_{no_sims}_no_bins_{no_bins}.npy")

    f_m_num_avg_vs_time = np.load(load_dir + f"f_m_num_avg_vs_time_no_sims_{no_sims}_no_bins_{no_bins}.npy")
    g_m_num_avg_vs_time = np.load(load_dir + f"g_m_num_avg_vs_time_no_sims_{no_sims}_no_bins_{no_bins}.npy")
    g_ln_r_num_avg_vs_time = np.load(load_dir + f"g_ln_r_num_avg_vs_time_no_sims_{no_sims}_no_bins_{no_bins}.npy")

    moments_vs_time_avg = np.load(load_dir + f"moments_vs_time_avg_no_sims_{no_sims}_no_bins_{no_bins}.npy")

    fig_name = "fm_gm_glnr_vs_t"
    fig_name += f"_kappa_{kappa}_dt_{int(dt)}_no_sims_{no_sims}_no_bins_{no_bins}.png"
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
        ax.tick_params(which="both", bottom=True, top=True, left=True, right=True)

    fig.suptitle(
f"dt={dt}, kappa={kappa}, eta={eta:.0e}, no_sims={no_sims}, no_bins={no_bins}\n\
gen_method={gen_method}, kernel={kernel_name}, bin_method={bin_method}")
    fig.tight_layout()
    plt.subplots_adjust(top=0.95)
    fig.savefig(load_dir + fig_name)
    
    ### plot moments vs time
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
        ax.tick_params(which="both", bottom=True, top=True, left=True, right=True)
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
    fig.suptitle(
f"dt={dt}, kappa={kappa}, eta={eta:.0e}, no_sims={no_sims}, no_bins={no_bins}\n\
gen_method={gen_method}, kernel={kernel_name}, bin_method={bin_method}")
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
    
    fig, axes = plt.subplots(nrows=no_rows, figsize=(10,6*no_rows), sharex=True)
    
    for kappa_n,kappa in enumerate(kappa_list):
        load_dir = sim_data_path + result_path_add + f"kappa_{kappa}/dt_{int(dt)}/"
        save_times = np.load(load_dir + f"save_times_{start_seed}.npy")
        moments_vs_time_avg = np.load(load_dir + f"moments_vs_time_avg_no_sims_{no_sims}_no_bins_{no_bins}.npy")
        
        if kappa_n < 10: fmt = "x-"
        else: fmt = "x--"            
        
        for i,ax in enumerate(axes):
            ax.plot(save_times/60, moments_vs_time_avg[:,i],fmt,label=f"{kappa}")

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
f"Moments of the distribution for various $\kappa$ (see legend)\n\
dt={dt:.1e}, eta={eta:.0e}, r_critmin=0.6, no_sims={no_sims}, \
gen_method={gen_method}, kernel={kernel_name}"
    fig.suptitle(title, fontsize=TTFS, y = 0.997)
    fig.tight_layout()
    plt.subplots_adjust(top=0.965)
    fig.savefig(fig_dir + fig_name)
    plt.close("all")

#%% plot moments 0, 2 and 3 of the mass distribution vs time for several kappa in
# the same plot. formatting as in the GMD publication    
    
### IN WORK: INSERT PLOT DICTIONARIES ETC. FROM FILE run_box_model_plots_MA.py    
### IN WORK: INSERT PLOT DICTIONARIES ETC. FROM FILE run_box_model_plots_MA.py    
### IN WORK: INSERT PLOT DICTIONARIES ETC. FROM FILE run_box_model_plots_MA.py    
def plot_moments_vs_time_kappa_var_paper(kappa_list, eta, dt, no_sims, no_bins,
                                         kernel_name, gen_method,
                                         dist, start_seed,
                                         moments_ref, times_ref,
                                         sim_data_path,
                                         result_path_add,
                                         figsize, figname, figsize2, figname2,
                                         figsize3, figname3, figsize4, figname4,
                                         TTFS, LFS, TKFS):
    no_rows = 3
    
    fig, axes = plt.subplots(nrows=no_rows, figsize=(figsize), sharex=True)
    
    for kappa_n,kappa in enumerate(kappa_list):
        load_dir = sim_data_path + result_path_add + f"kappa_{kappa}/dt_{int(dt)}/"
        save_times = np.load(load_dir + f"save_times_{start_seed}.npy")
        moments_vs_time_avg = np.load(load_dir + f"moments_vs_time_avg_no_sims_{no_sims}_no_bins_{no_bins}.npy")
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
                            zorder=98)
    if kernel_name == "Golovin":
        DNC = 296799076.3
        LWC = 1E-3
        bG = 1.5
            
    for ax_n,i in enumerate((0,2,3)):
        ax = axes[ax_n]
        if kernel_name == "Golovin":
            
            fmt = "o"
            times_ref = np.linspace(0.,3600.,19)
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
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()      
            
            if kernel_name == "Golovin":
                ax.legend(
                        np.reshape(handles, (2,4)).T.flatten(),
                        np.reshape(labels, (2,4)).T.flatten(),
                          ncol=4, handlelength=0.8, handletextpad=0.2,
                          columnspacing=0.5, borderpad=0.2, loc="upper right",
                          bbox_to_anchor=(1.015, 1.04))            
        if i == 2:
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
    
    no_rows = len(kappa_list)

### REFERENCE REL DEV PLOTS 
    
    fig, axes = plt.subplots(nrows=no_rows, ncols=2, figsize=(figsize2), sharex=True)    
    
    for kappa_n,kappa in enumerate(kappa_list):
        load_dir = sim_data_path + result_path_add + f"kappa_{kappa}/dt_{int(dt)}/"
        save_times = np.load(load_dir + f"save_times_{start_seed}.npy")
        
        if kernel_name == "Golovin":
            moments_vs_time_avg = np.load(load_dir + f"moments_vs_time_avg_no_sims_{no_sims}_no_bins_{no_bins}.npy")    
            moments_vs_time_std = np.load(load_dir + f"moments_vs_time_std_no_sims_{no_sims}_no_bins_{no_bins}.npy")    
        else:
            moments_vs_time_avg = np.load(load_dir + f"moments_vs_time_avg_no_sims_{no_sims}_no_bins_{no_bins}.npy")[::2,:]
            moments_vs_time_std = np.load(load_dir + f"moments_vs_time_std_no_sims_{no_sims}_no_bins_{no_bins}.npy")[::2,:]
        
        for cnt_m,mom_n in enumerate((0,2,3)):
            if kernel_name == "Golovin":
                moments_ref_i = compute_moments_Golovin(save_times, mom_n, DNC, LWC, bG)
                save_times0 = save_times
            else:
                moments_ref_i = moments_ref[mom_n][::3]
                save_times0 = save_times[::2]
                
            
            rel_dev = np.abs(moments_vs_time_avg[:,mom_n]-moments_ref_i)/moments_ref_i
            
            rel_err = moments_vs_time_std[:,mom_n] / moments_ref_i
            
            ax = axes[kappa_n,0]
            ax.errorbar(save_times0/60, rel_dev, rel_err, label=str(mom_n))
            ax.grid()

            ax = axes[kappa_n,1]
            ax.plot(save_times0/60, rel_err, label=str(mom_n))
            ax.grid()
        axes[kappa_n,1].annotate(
                            r"$N_\mathrm{{SIP}} = {}$".format(kappa*5),
                            (0.01,0.82),
                            xycoords="axes fraction")                 
        
        ax = axes[kappa_n,0]
        ax.set_yscale("log")
        if kernel_name == "Golovin":
            ax.set_yticks(np.logspace(-5,0,6))
        else:
            ax.set_yticks(np.logspace(-4,1,6))

        ax = axes[kappa_n,1]
        ax.set_yscale("log")
        if kernel_name == "Golovin":
            ax.set_yticks(np.logspace(-5,0,6))
        else:
            ax.set_yticks(np.logspace(-6,0,4))
        
    axes[0,0].legend(ncol=3)
    axes[0,1].legend(ncol=3)
    
    axes[0,0].set_title("rel. dev. $(\lambda-\lambda_\mathrm{ref})/\lambda_\mathrm{ref}$")
    axes[0,1].set_title("rel. error $\mathrm{SD}(\lambda)/\lambda_\mathrm{ref}$ ")
    
    
    axes[-1,0].set_xticks(np.linspace(0,60,7))
    axes[-1,1].set_xticks(np.linspace(0,60,7))
    axes[-1,0].set_xlim((0,60))
    axes[-1,1].set_xlim((0,60))
    
    axes[-1,0].set_xlabel("Time (min)")
    axes[-1,1].set_xlabel("Time (min)")
    
    fig.savefig(figname2,
                bbox_inches = 'tight',
                pad_inches = 0.04
                )   
    
### CONVERGENCE REL DEV PLOTS 
    no_rows = 1
    fig, axes = plt.subplots(nrows=no_rows, ncols=2,
                             figsize=(figsize3)
                             )    
    
    kappa = kappa_list[-1]
    load_dir = sim_data_path + result_path_add + f"kappa_{kappa}/dt_{int(dt)}/"
    moments_vs_time_avg_ref = np.load(load_dir + f"moments_vs_time_avg_no_sims_{no_sims}_no_bins_{no_bins}.npy")
#    moments_vs_time_std_ref = np.load(load_dir + f"moments_vs_time_std_no_sims_{no_sims}_no_bins_{no_bins}.npy")    
    
    kappa = kappa_list[-2]
    load_dir = sim_data_path + result_path_add + f"kappa_{kappa}/dt_{int(dt)}/"
    save_times = np.load(load_dir + f"save_times_{start_seed}.npy")
    moments_vs_time_avg = np.load(load_dir + f"moments_vs_time_avg_no_sims_{no_sims}_no_bins_{no_bins}.npy")    
    moments_vs_time_std = np.load(load_dir + f"moments_vs_time_std_no_sims_{no_sims}_no_bins_{no_bins}.npy")    
        
#        if kernel_name == "Golovin":
#        else:
##            moments_vs_time_avg = np.load(load_dir + f"moments_vs_time_avg_no_sims_{no_sims}_no_bins_{no_bins}.npy")    
##            moments_vs_time_std = np.load(load_dir + f"moments_vs_time_std_no_sims_{no_sims}_no_bins_{no_bins}.npy")    
#            moments_vs_time_avg = np.load(load_dir + f"moments_vs_time_avg_no_sims_{no_sims}_no_bins_{no_bins}.npy")
#            moments_vs_time_std = np.load(load_dir + f"moments_vs_time_std_no_sims_{no_sims}_no_bins_{no_bins}.npy")
##            print ("moments_vs_time_avg.shape")
##            print (moments_vs_time_avg.shape)
#        
#        for cnt_m,mom_n in enumerate((0,2,3)):
##            if kernel_name == "Golovin":
##                moments_ref_i = compute_moments_Golovin(save_times, mom_n, DNC, LWC, bG)
##                save_times0 = save_times
##            else:
##                moments_ref_i = moments_ref[mom_n][::3]
###                print("moments_ref_i.shape")
###                print(moments_ref_i.shape)
##                save_times0 = save_times[::2]
###                print(save_times)
    save_times0 = save_times
    
    for cnt_m,mom_n in enumerate((0,2,3)):
        moments_ref_i = moments_vs_time_avg_ref[:,mom_n]
        rel_dev = np.abs(moments_vs_time_avg[:,mom_n]-moments_ref_i)/moments_ref_i
        
        rel_err = moments_vs_time_std[:,mom_n] / moments_ref_i
        
        ax = axes[0]
        ax.errorbar(save_times0/60, rel_dev, rel_err, label=str(mom_n))
        ax.grid()
    
        ax.set_yscale("log")

        ax = axes[1]
        ax.plot(save_times0/60, rel_err, label=str(mom_n))
        ax.grid()
    
        ax.set_yscale("log")
        
    for ax in axes:
        ax.legend(
    #                        np.reshape(handles, (5,2)).T.flatten(),
    #                        np.reshape(labels, (5,2)).T.flatten(),
    #                    np.concatenate((handles[::2],handles[1::2]),axis=0),
    #                      np.concatenate((labels[::2],labels[1::2]),axis=0),
                  ncol=3, handlelength=0.8, handletextpad=0.2,
                  columnspacing=0.5, borderpad=0.2,
#                  loc="lower left",
#                  bbox_to_anchor=(0.0, 0.6)
                  ).set_zorder(100)           
    
    axes[0].set_title("rel. dev. $(\lambda-\lambda_\mathrm{ref})/\lambda_\mathrm{ref}$")
    axes[1].set_title("rel. error $\mathrm{SD}(\lambda)/\lambda_\mathrm{ref}$ ")
    axes[0].set_xticks(np.linspace(0,60,7))
    axes[1].set_xticks(np.linspace(0,60,7))
    axes[0].set_xlim((0,60))
    axes[1].set_xlim((0,60))
    
    axes[0].set_xlabel("Time (min)")
    axes[1].set_xlabel("Time (min)")

    fig.savefig(figname3,
#                bbox_inches = 0,
                bbox_inches = 'tight',
                pad_inches = 0.04
                )   
    
### CONVERGENCE REL DEV PLOTS 2   
    no_rows = 1
    fig, axes = plt.subplots(nrows=no_rows, ncols=2,
                             figsize=(figsize4)
                             )    
    
    kappa = kappa_list[-1]
    load_dir = sim_data_path + result_path_add + f"kappa_{kappa}/dt_{int(dt)}/"
    moments_vs_time_avg_ref = np.load(load_dir + f"moments_vs_time_avg_no_sims_{no_sims}_no_bins_{no_bins}.npy")
#    moments_vs_time_std_ref = np.load(load_dir + f"moments_vs_time_std_no_sims_{no_sims}_no_bins_{no_bins}.npy")    
    
    kappa = kappa_list[1]
    load_dir = sim_data_path + result_path_add + f"kappa_{kappa}/dt_{int(dt)}/"
    save_times = np.load(load_dir + f"save_times_{start_seed}.npy")
    moments_vs_time_avg = np.load(load_dir + f"moments_vs_time_avg_no_sims_{no_sims}_no_bins_{no_bins}.npy")    
    moments_vs_time_std = np.load(load_dir + f"moments_vs_time_std_no_sims_{no_sims}_no_bins_{no_bins}.npy")    
        
    save_times0 = save_times
    
    for cnt_m,mom_n in enumerate((0,2,3)):
        moments_ref_i = moments_vs_time_avg_ref[:,mom_n]
        rel_dev = np.abs(moments_vs_time_avg[:,mom_n]-moments_ref_i)/moments_ref_i
        
        rel_err = moments_vs_time_std[:,mom_n] / moments_ref_i
        
        ax = axes[0]
        ax.errorbar(save_times0/60, rel_dev, rel_err, label=str(mom_n))
        ax.grid()
    
        ax.set_yscale("log")

        ax = axes[1]
        ax.plot(save_times0/60, rel_err, label=str(mom_n))
        ax.grid()
    
        ax.set_yscale("log")
        
    for ax in axes:
        ax.legend(
    #                        np.reshape(handles, (5,2)).T.flatten(),
    #                        np.reshape(labels, (5,2)).T.flatten(),
    #                    np.concatenate((handles[::2],handles[1::2]),axis=0),
    #                      np.concatenate((labels[::2],labels[1::2]),axis=0),
                  ncol=3, handlelength=0.8, handletextpad=0.2,
                  columnspacing=0.5, borderpad=0.2,
#                  loc="lower left",
#                  bbox_to_anchor=(0.0, 0.6)
                  ).set_zorder(100)           
    
    axes[0].set_title("rel. dev. $(\lambda-\lambda_\mathrm{ref})/\lambda_\mathrm{ref}$")
    axes[1].set_title("rel. error $\mathrm{SD}(\lambda)/\lambda_\mathrm{ref}$ ")
    
    
    axes[0].set_xticks(np.linspace(0,60,7))
    axes[1].set_xticks(np.linspace(0,60,7))
    axes[0].set_xlim((0,60))
    axes[1].set_xlim((0,60))
    
    axes[0].set_xlabel("Time (min)")
    axes[1].set_xlabel("Time (min)")
    
    fig.savefig(figname4,
                bbox_inches = 'tight',
                pad_inches = 0.04
                )   
    
    plt.close("all")        

#%% Plot g_lnR vs time for a single kappa
# or for two specific kappa one below the other
    
def plot_g_ln_R_for_given_kappa(kappa, kappa1, kappa2,
                                eta, dt, no_sims, start_seed, no_bins,
                                DNC0, m_mean,
                                kernel_name, gen_method, bin_method, time_idx,
                                load_dir, load_dir_k1, load_dir_k2,
                                figsize, figname,
                                plot_compare,
                                figsize2, figname_compare, LFS):
    bG = 1.5 # K = b * (m1 + m2) # b in m^3/(fg s)
    save_times = np.load(load_dir + f"save_times_{start_seed}.npy")
#    no_times = len(save_times)
    bins_mass_centers = np.load(load_dir + f"bins_mass_centers_{no_sims}_no_bins_{no_bins}.npy")
    bins_rad_centers = np.load(load_dir + f"bins_rad_centers_{no_sims}_no_bins_{no_bins}.npy")

    add_masses = 4
    
#    f_m_num_avg_vs_time = np.load(load_dir + f"f_m_num_avg_vs_time_no_sims_{no_sims}_no_bins_{no_bins}.npy")
#    g_m_num_avg_vs_time = np.load(load_dir + f"g_m_num_avg_vs_time_no_sims_{no_sims}_no_bins_{no_bins}.npy")
    g_ln_r_num_avg_vs_time = np.load(load_dir + f"g_ln_r_num_avg_vs_time_no_sims_{no_sims}_no_bins_{no_bins}.npy")
    g_ln_r_num_std_vs_time = np.load(load_dir + f"g_ln_r_num_std_vs_time_no_sims_{no_sims}_no_bins_{no_bins}.npy")

#    moments_vs_time_avg = np.load(load_dir + f"moments_vs_time_avg_no_sims_{no_sims}_no_bins_{no_bins}.npy")
#    moments_vs_time_std = np.load(load_dir + f"moments_vs_time_std_no_sims_{no_sims}_no_bins_{no_bins}.npy")

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
            
            bin_factor = np.sqrt(bins_mass_centers[time_n][0][-1]/bins_mass_centers[time_n][0][-2])
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
            dp = f"collision/ref_data/{kernel_name}/"
            ref_radii = np.loadtxt(dp + "Wang_2007_radius_bin_centers.txt")[j][::5]
            g_ln_r_ref = np.loadtxt(dp + "Wang_2007_g_ln_R.txt")[j][::5]
#        fmt="o"    
#        print(j)
#        print(ref_radii.shape)
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
    
def plot_g_ln_R_kappa_compare(kappa1, kappa2,
                              eta, dt, no_sims, start_seed, no_bins,
                              DNC0, m_mean,
                              kernel_name, gen_method, bin_method, time_idx,
                              load_dir_k1, load_dir_k2,
                              figsize, figname_compare, LFS):
    bG = 1.5 # K = b * (m1 + m2) # b in m^3/(fg s)
    save_times = np.load(load_dir_k1 + f"save_times_{start_seed}.npy")

    no_rows = 2
    fig, axes = plt.subplots(nrows=no_rows, figsize=figsize)
    
    load_dirs = (load_dir_k1, load_dir_k2)
    
    for n,kappa in enumerate((kappa1,kappa2)):
        load_dir = load_dirs[n]
        bins_mass_centers = np.load(load_dir + f"bins_mass_centers_{no_sims}_no_bins_{no_bins}.npy")
        bins_rad_centers = np.load(load_dir + f"bins_rad_centers_{no_sims}_no_bins_{no_bins}.npy")
    
        add_masses = 4
#            f_m_num_avg_vs_time = np.load(load_dir + f"f_m_num_avg_vs_time_no_sims_{no_sims}_no_bins_{no_bins}.npy")
#            g_m_num_avg_vs_time = np.load(load_dir + f"g_m_num_avg_vs_time_no_sims_{no_sims}_no_bins_{no_bins}.npy")
        g_ln_r_num_avg_vs_time = np.load(load_dir + f"g_ln_r_num_avg_vs_time_no_sims_{no_sims}_no_bins_{no_bins}.npy")
        g_ln_r_num_std_vs_time = np.load(load_dir + f"g_ln_r_num_std_vs_time_no_sims_{no_sims}_no_bins_{no_bins}.npy")
#            moments_vs_time_avg = np.load(load_dir + f"moments_vs_time_avg_no_sims_{no_sims}_no_bins_{no_bins}.npy")
#            moments_vs_time_std = np.load(load_dir + f"moments_vs_time_std_no_sims_{no_sims}_no_bins_{no_bins}.npy")        
        
#            LW_compare = 1.0
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
                
                bin_factor = np.sqrt(bins_mass_centers[time_n][0][-1]/bins_mass_centers[time_n][0][-2])
                ref_masses[0] = bins_mass_centers[time_n][0][0]
    
                for n in range(1,no_bins_ref+add_masses):
                    ref_masses[n] = ref_masses[n-1]*bin_factor
                f_m_golo = dist_vs_time_golo_exp(ref_masses,
                                                 save_times[time_n],m_mean,DNC0,bG)
                g_ln_r_golo = f_m_golo * 3 * ref_masses**2
                g_ln_r_ref = g_ln_r_golo
                ref_radii = 1E6 * (3. * ref_masses / (4. * math.pi * 1E3))**(1./3.)                
            else:
                scale_g=1.
                dp = f"collision/ref_data/{kernel_name}/"
                ref_radii = np.loadtxt(dp + "Wang_2007_radius_bin_centers.txt")[j][::5]
                g_ln_r_ref = np.loadtxt(dp + "Wang_2007_g_ln_R.txt")[j][::5]
                
            fmt="o"    
            ax.plot(ref_radii,
                    g_ln_r_ref*scale_g,
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