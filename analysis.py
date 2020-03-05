#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TROPOS LAGRANGIAN CLOUD MODEL
Super-Droplet method in two-dimensional kinematic framework
(Test Case 1 ICMW 2012)
Author: Jan Bohrer (bohrer@tropos.de)
Further contact: Oswald Knoth (knoth@tropos.de)

FUNCTIONS FOR SIMULATION DATA ANALYSIS AND DATA PROCESSING FOR PLOTTING

basic units:
particle mass, water mass, solute mass in femto gram = 10^-18 kg
particle radius in micro meter ("mu")
all other quantities in SI units
"""

#%% MODULE IMPORTS

import math
import numpy as np
from numba import njit
import timeit

import constants as c
import microphysics as mp
from distributions import conc_per_mass_expo_np, conc_per_mass_lognormal_np
from file_handling import load_grid_scalar_fields, load_particle_data_all

from plotting import plot_ensemble_data

#%% UPDATE FUNCTIONS

# in unit (kg/kg)
@njit()
def update_mixing_ratio(mixing_ratio, m_w, xi, cells, mass_dry_inv, 
                        id_list, mask):
    mixing_ratio.fill(0.0)
    for ID in id_list[mask]:
        mixing_ratio[cells[0,ID],cells[1,ID]] += m_w[ID] * xi[ID]
    mixing_ratio *= 1.0E-18 * mass_dry_inv  

# in unit (1/kg)
@njit()
def update_number_concentration_per_dry_mass(conc, xi, cells, mass_dry_inv, 
                        id_list, mask):
    conc.fill(0.0)
    for ID in id_list[mask]:
        conc[cells[0,ID],cells[1,ID]] += xi[ID]
    conc *= mass_dry_inv 

@njit()
def update_T_p(grid_temp, cells, T_p):
    for ID in range(len(T_p)):
        T_p[ID] = grid_temp[cells[0,ID],cells[1,ID]] 

#%% RUNTIME OF FUNCTIONS

# INPUT:
# functions: list of strings,
# e.g. ["compute_r_l_grid_field", "compute_r_l_grid_field_np"]
# pars: string,
# e.g. "m_w, xi, cells, grid.mixing_ratio_water_liquid, grid.mass_dry_inv"
# rs: list of repeats (int)
# ns: number of exec per repeat (int)
# example:
# funcs = ["compute_r_l_grid_field_np", "compute_r_l_grid_field"]
# pars = "m_w, xi, cells, grid.mixing_ratio_water_liquid, grid.mass_dry_inv"
# rs = [5,5,5]
# ns = [100,10000,1000]
# compare_functions_run_time(funcs, pars, rs, ns, globals_=globals())
# NOTE that we need to call with globals_=globals() explicitly
# a default argument for globals_ cannot be given in the function definition..
# because in that case, the globals are taken from module "analysis.py" and
# not from the environment of the executed program
def compare_functions_run_time(functions, pars, rs, ns, globals_):
    # print (__name__)
    t = []
    for i,func in enumerate(functions):
        print(func + ": repeats =", rs[i], "no reps = ", ns[i])
    # print(globals_)
    for i,func in enumerate(functions):
        statement = func + "(" + pars + ")"
        t_ = timeit.repeat(statement, repeat=rs[i],
                           number=ns[i], globals=globals_)
        t.append(t_)
        print("best = ", f"{min(t_)/ns[i]*1.0E6:.4}", "us;",
              "worst = ", f"{max(t_)/ns[i]*1.0E6:.4}", "us;",
              "mean =", f"{np.mean(t_)/ns[i]*1.0E6:.4}",
              "+-", f"{np.std(t_, ddof = 1)/ns[i]*1.0E6:.3}", "us" )

#%% BINNING OF SIPs:

def auto_bin_SIPs(masses, xis, no_bins, dV, no_sims, xi_min=1):

    ind = np.nonzero(xis)
    m_sort = masses[ind]
    xi_sort = xis[ind]
    
    ind = np.argsort(m_sort)
    m_sort = m_sort[ind]
    xi_sort = xi_sort[ind]
    
    ### merge particles with xi < xi_min
    for i in range(len(xi_sort)-1):
        if xi_sort[i] < xi_min:
            xi = xi_sort[i]
            m = m_sort[i]
            xi_left = 0
            j = i
            while(j > 0 and xi_left==0):
                j -= 1
                xi_left = xi_sort[j]
            if xi_left != 0:
                m1 = m_sort[j]
                dm_left = m-m1
            else:
                dm_left = 1.0E18
            m2 = m_sort[i+1]
            if m2-m < dm_left:
                j = i+1
                # assign to m1 since distance is smaller
                # i.e. increase number of xi[i-1],
                # then reweight mass to total mass
            m_sum = m*xi + m_sort[j]*xi_sort[j]
            xi_sort[i] = 0           
    
    if xi_sort[-1] < xi_min:
        i = -1
        xi = xi_sort[i]
        m = m_sort[-1]
        xi_left = 0
        j = i
        while(xi_left==0):
            j -= 1
            xi_left = xi_sort[j]
        
        m_sum = m*xi + m_sort[j]*xi_sort[j]
        xi_sort[j] += xi_sort[i]
        m_sort[j] = m_sum / xi_sort[j]
        xi_sort[i] = 0           
    
    ind = np.nonzero(xi_sort)
    xi_sort = xi_sort[ind]
    m_sort = m_sort[ind]
    
    ind = np.argsort(m_sort)
    m_sort = m_sort[ind]
    xi_sort = xi_sort[ind]
    
    ### merge particles, which have masses or xis < m_lim, xi_lim
    no_bins0 = no_bins
    no_bins *= 10
    
    no_spc = len(xi_sort)
    n_save = int(no_spc//1000)
    if n_save < 2: n_save = 2
    
    no_rpc = np.sum(xi_sort)
    total_mass = np.sum(xi_sort*m_sort)
    xi_lim = no_rpc / no_bins
    m_lim = total_mass / no_bins
    
    bin_centers = []
    m_bin = []
    xi_bin = []
    
    n_left = no_rpc
    
    i = 0
    while(n_left > 0 and i < len(xi_sort)-n_save):
        bin_mass = 0.0
        bin_xi = 0
        while(bin_mass < m_lim and bin_xi < xi_lim and n_left > 0
              and i < len(xi_sort)-n_save):
            bin_xi += xi_sort[i]
            bin_mass += xi_sort[i] * m_sort[i]
            n_left -= xi_sort[i]
            i += 1
        bin_centers.append(bin_mass / bin_xi)
        m_bin.append(bin_mass)
        xi_bin.append(bin_xi)
            
    xi_bin = np.array(xi_bin)
    bin_centers = np.array(bin_centers)
    m_bin = np.array(m_bin)
    
    ### merge particles, whose masses are close together in log space:
    bin_size_log =\
        (np.log10(bin_centers[-1]) - np.log10(bin_centers[0])) / no_bins0
    
    i = 0
    while(i < len(xi_bin)-1):
        m_next_bin = bin_centers[i] * 10**bin_size_log
        m = bin_centers[i]
        j = i
        while (m < m_next_bin and j < len(xi_bin)-1):
            j += 1
            m = bin_centers[j]
        if m >= m_next_bin:
            j -= 1
        if (i != j):
            m_sum = 0.0
            xi_sum = 0
            for k in range(i,j+1):
                m_sum += m_bin[k]
                xi_sum += xi_bin[k]
                if k > i:
                    xi_bin[k] = 0
            bin_centers[i] = m_sum / xi_sum
            xi_bin[i] = xi_sum
            m_bin[i] = m_sum
        i = j+1            
    
    
    ind = np.nonzero(xi_bin)
    xi_bin = xi_bin[ind]
    bin_centers = bin_centers[ind]        
    m_bin = m_bin[ind]
    
    radii = mp.compute_radius_from_mass_vec(bin_centers,
                                            c.mass_density_water_liquid_NTP)
    
    # find the midpoints between the masses/radii
    # midpoints = 0.5 * ( m_sort[:-1] + m_sort[1:] )
    # m_left = 2.0 * m_sort[0] - midpoints[0]
    # m_right = 2.0 * m_sort[-1] - midpoints[-1]
    bins = 0.5 * ( radii[:-1] + radii[1:] )
    # add missing bin borders for m_min and m_max:
    R_left = 2.0 * radii[0] - bins[0]
    R_right = 2.0 * radii[-1] - bins[-1]
    
    bins = np.hstack([R_left, bins, R_right])
    bins_log = np.log(bins)
    
    m_bin = np.array(m_bin)
    g_ln_R = m_bin * 1.0E-15 / no_sims / (bins_log[1:] - bins_log[0:-1]) / dV
    
    return g_ln_R, radii, bins, xi_bin, bin_centers

#%% PARTICLE SIZE SPECTRA
        
# active ids not necessary: choose target cell and no_cells_x
# such that the region is included in the valid domain
@njit()
def sample_masses(m_w, m_s, xi, cells, id_list, grid_temperature,
                  target_cell, no_cells_x, no_cells_z):
    
    dx = no_cells_x // 2
    dz = no_cells_z // 2
    
    i_low = target_cell[0] - dx
    i_high = target_cell[0] + dx
    j_low = target_cell[0] - dz
    j_high = target_cell[0] + dz
    
    mask =   ((cells[0] >= i_low) & (cells[0] <= i_high)) \
           & ((cells[1] >= j_low) & (cells[1] <= j_high))
    
    no_masses = mask.sum()
    
    m_s_out = np.zeros(no_masses, dtype = np.float64)
    m_w_out = np.zeros(no_masses, dtype = np.float64)
    xi_out = np.zeros(no_masses, dtype = np.float64)
    T_p = np.zeros(no_masses, dtype = np.float64)
    
    for cnt, ID in enumerate(id_list[mask]):
        m_s_out[cnt] = m_s[ID]
        m_w_out[cnt] = m_w[ID]
        xi_out[cnt] = xi[ID]
        T_p[cnt] = grid_temperature[ cells[0,ID], cells[1,ID] ]
    
    return m_w_out, m_s_out, xi_out, T_p


# active ids not necessary: choose target cell and no_cells_x
# such that the region is included in the valid domain
# weights_out = xi/mass_dry_inv (in that respective cell) 
# weights_out in number/kg_dry_air
@njit()
def sample_masses_per_m_dry(m_w, m_s, xi, cells, id_list, grid_temperature,
                            grid_mass_dry_inv,
                  target_cell, no_cells_x, no_cells_z):
    
    dx = no_cells_x // 2
    dz = no_cells_z // 2
    
    i_low = target_cell[0] - dx
    i_high = target_cell[0] + dx
    j_low = target_cell[1] - dz
    j_high = target_cell[1] + dz
    
    no_cells_eval = (dx * 2 + 1) * (dz * 2 + 1)
    
    mask =   ((cells[0] >= i_low) & (cells[0] <= i_high)) \
           & ((cells[1] >= j_low) & (cells[1] <= j_high))
    
    no_masses = mask.sum()
    
    m_s_out = np.zeros(no_masses, dtype = np.float64)
    m_w_out = np.zeros(no_masses, dtype = np.float64)
    xi_out = np.zeros(no_masses, dtype = np.float64)
    weights_out = np.zeros(no_masses, dtype = np.float64)
    T_p = np.zeros(no_masses, dtype = np.float64)
    
    for cnt, ID in enumerate(id_list[mask]):
        m_s_out[cnt] = m_s[ID]
        m_w_out[cnt] = m_w[ID]
        xi_out[cnt] = xi[ID]
        weights_out[cnt] = xi[ID] * grid_mass_dry_inv[cells[0,ID], cells[1,ID]]
        T_p[cnt] = grid_temperature[ cells[0,ID], cells[1,ID] ]
    
    return m_w_out, m_s_out, xi_out, weights_out, T_p, no_cells_eval



# we always assume the only quantities stored are m_s, m_w, xi
def sample_radii(m_w, m_s, xi, cells, solute_type, id_list,
                 grid_temperature, target_cell, no_cells_x, no_cells_z):
    m_w_out, m_s_out, xi_out, T_p = sample_masses(m_w, m_s, xi, cells, id_list,
                                                  grid_temperature,
                                                  target_cell, no_cells_x,
                                                  no_cells_z)
    if solute_type == "AS":
        mass_density_dry = c.mass_density_AS_dry
        compute_R_p_w_s_rho_p = mp.compute_R_p_w_s_rho_p_AS
    elif solute_type == "NaCl":
        mass_density_dry = c.mass_density_NaCl_dry
        compute_R_p_w_s_rho_p = mp.compute_R_p_w_s_rho_p_NaCl
    
    R_s = mp.compute_radius_from_mass_vec(m_s_out, mass_density_dry)
    R_p, w_s, rho_p = compute_R_p_w_s_rho_p(m_w_out, m_s_out, T_p)
    
    return R_p, R_s, xi_out        

# weights_out in number/kg_dry_air
# we always assume the only quantities stored are m_s, m_w, xi
def sample_radii_per_m_dry(m_w, m_s, xi, cells, solute_type, id_list,
                 grid_temperature, grid_mass_dry_inv,
                 target_cell, no_cells_x, no_cells_z):
    m_w_out, m_s_out, xi_out, weights_out, T_p, no_cells_eval = \
        sample_masses_per_m_dry(m_w, m_s, xi, cells, id_list, grid_temperature,
                                grid_mass_dry_inv,
                                target_cell, no_cells_x, no_cells_z)
    if solute_type == "AS":
        mass_density_dry = c.mass_density_AS_dry
        compute_R_p_w_s_rho_p = mp.compute_R_p_w_s_rho_p_AS
    elif solute_type == "NaCl":
        mass_density_dry = c.mass_density_NaCl_dry
        compute_R_p_w_s_rho_p = mp.compute_R_p_w_s_rho_p_NaCl
    
    R_s = mp.compute_radius_from_mass_vec(m_s_out, mass_density_dry)
    R_p, w_s, rho_p = compute_R_p_w_s_rho_p(m_w_out, m_s_out, T_p)
    
    return R_p, R_s, xi_out, weights_out, no_cells_eval        

#%%
def avg_moments_over_boxes(
        moments_vs_time_all_seeds, no_seeds, idx_t, no_moments,
        target_cells_x, target_cells_z,
        no_cells_per_box_x, no_cells_per_box_z):
    no_times_eval = len(idx_t)
    no_target_cells_x = len(target_cells_x)
    no_target_cells_z = len(target_cells_z)
    di_cell = no_cells_per_box_x // 2
    dj_cell = no_cells_per_box_z // 2    
    moments_at_boxes_all_seeds = np.zeros( (no_seeds,no_times_eval,no_moments,
                                  no_target_cells_x,no_target_cells_z),
                                 dtype = np.float64)

    for seed_n in range(no_seeds):
        for time_n, time_ind in enumerate(idx_t):
            for mom_n in range(no_moments):
                for box_n_x, tg_cell_x in enumerate(target_cells_x):
                    for box_n_z , tg_cell_z in enumerate(target_cells_z):
                        moment_box = 0.
                        i_tg_corner = tg_cell_x - di_cell
                        j_tg_corner = tg_cell_z - dj_cell
                        cells_box_x = np.arange(i_tg_corner,
                                                i_tg_corner+no_cells_per_box_x)
                        cells_box_z = np.arange(j_tg_corner,
                                                j_tg_corner+no_cells_per_box_z)
                        MG = np.meshgrid(cells_box_x, cells_box_z)
                        
                        cells_box_x = MG[0].flatten()
                        cells_box_z = MG[1].flatten()
                        
                        moment_box = moments_vs_time_all_seeds[seed_n, time_n,
                                                               mom_n,
                                                               cells_box_x,
                                                               cells_box_z]
                        
                        moment_box = np.average(moment_box)
                        moments_at_boxes_all_seeds[seed_n, time_n, mom_n,
                                                   box_n_x, box_n_z] = \
                            moment_box
    return moments_at_boxes_all_seeds 

#%% DATA ANALYSIS

@njit()
# V0 = volume grid cell
def compute_moment_R_grid(n, R_p, xi, V0,
                          cells, active_ids, id_list, no_cells):
    moment = np.zeros( (no_cells[0], no_cells[1]), dtype = np.float64 )
    if n == 0:
        for ID in id_list[active_ids]:
            moment[cells[0,ID],cells[1,ID]] += xi[ID]
    else:
        for ID in id_list[active_ids]:
            moment[cells[0,ID],cells[1,ID]] += xi[ID] * R_p[ID]**n
    return moment / V0



# possible field indices:
#0: r_v
#1: r_l
#2: Theta
#3: T
#4: p
#5: S
# possibe derived indices:
# 0: r_aero
# 1: r_cloud
# 2: r_rain     
# 3: n_aero
# 4: n_c
# 5: n_r 
# 6: R_avg
# 7: R_1/2 = 2nd moment / 1st moment
# 8: R_eff = 3rd moment/ 2nd moment of R-distribution
# NOTE that time_indices must be an array because of indexing below     
def generate_field_frame_data_avg(load_path_list,
                                  field_indices, time_indices,
                                  derived_indices,
                                  mass_dry_inv, grid_volume_cell,
                                  no_cells, solute_type):
    # output: fields_with_time = [ [time0: all fields[] ],
    #                              [time1],
    #                              [time2], .. ]
    # for collected fields:
    # unit_list
    # name_list
    # scales_list
    # save_times
    
    V0 = grid_volume_cell
    
    if solute_type == "AS":
        compute_R_p_w_s_rho_p = mp.compute_R_p_w_s_rho_p_AS
    elif solute_type == "NaCl":
        compute_R_p_w_s_rho_p = mp.compute_R_p_w_s_rho_p_NaCl
    
    bins_R_p_drop_classif = [0.5, 25.]
    
    field_names_orig = ["r_v", "r_l", "\Theta", "T", "p", "S"]
    scales_orig = [1000., 1000., 1, 1, 0.01, 1]
    units_orig = ["g/kg", "g/kg", "K", "K", "hPa", "-"]
    
    field_names_deri = ["r_\mathrm{aero}", "r_c", "r_r",
                       "n_\mathrm{aero}", "n_c", "n_r",
                       r"R_\mathrm{avg}", r"R_{2/1}", r"R_\mathrm{eff}"]
    units_deri = ["g/kg", "g/kg", "g/kg", "1/mg", "1/mg", "1/mg",
                  r"$\mathrm{\mu m}$", r"$\mathrm{\mu m}$", r"$\mathrm{\mu m}$"]
    scales_deri = [1000., 1000., 1000., 1E-6, 1E-6, 1E-6, 1., 1., 1.]    
    
    no_seeds = len(load_path_list)
    no_times = len(time_indices)
    no_fields_orig = len(field_indices)
    no_fields_derived = len(derived_indices)
    no_fields = no_fields_orig + no_fields_derived
    
    fields_with_time = np.zeros( (no_times, no_fields,
                                  no_cells[0], no_cells[1]),
                                dtype = np.float64)
    fields_with_time_sq = np.zeros( (no_times, no_fields,
                                  no_cells[0], no_cells[1]),
                                dtype = np.float64)
    
    
    load_path = load_path_list[0]
    frame_every, no_grid_frames, dump_every = \
        np.load(load_path+"data_saving_paras.npy")
    grid_save_times = np.load(load_path+"grid_save_times.npy")
    
    save_times_out = np.zeros(no_times, dtype = np.int64)
    
    field_names_out = []
    units_out = []
    scales_out = []
    
    for cnt in range(no_fields_orig):
        idx_f = field_indices[cnt]
        field_names_out.append(field_names_orig[idx_f])
        units_out.append(units_orig[idx_f])
        scales_out.append(scales_orig[idx_f])
    
    for cnt in range(no_fields_derived):
        idx_f = derived_indices[cnt]
        field_names_out.append(field_names_deri[idx_f])
        units_out.append(units_deri[idx_f])
        scales_out.append(scales_deri[idx_f])         
    
    for time_n in range(no_times):
        idx_t = time_indices[time_n]
        save_times_out[time_n] = grid_save_times[idx_t]
    
    for seed_n, load_path in enumerate(load_path_list):
        
        fields = load_grid_scalar_fields(load_path, grid_save_times)
        vec_data, cells_with_time, scal_data, xi_with_time, active_ids_with_time =\
            load_particle_data_all(load_path, grid_save_times)
        m_w_with_time = scal_data[:,0]
        m_s_with_time = scal_data[:,1]
        
        for cnt in range(no_fields_orig):
            idx_f = field_indices[cnt]
            fields_with_time[:,cnt] += fields[time_indices,idx_f]
            fields_with_time_sq[:,cnt] += \
                fields[time_indices,idx_f]*fields[time_indices,idx_f]
        
        for time_n in range(no_times):
            idx_t = time_indices[time_n]
            
            no_SIPs = len(xi_with_time[idx_t])
            T_p = np.zeros(no_SIPs, dtype = np.float64)
            id_list = np.arange(no_SIPs)
            update_T_p(fields[idx_t, 3], cells_with_time[idx_t], T_p)
            R_p, w_s, rho_p = \
                compute_R_p_w_s_rho_p(m_w_with_time[idx_t],
                                      m_s_with_time[idx_t], T_p)
            idx_R_p = np.digitize(R_p, bins_R_p_drop_classif)
            idx_classification = np.arange(3).reshape((3,1))
            
            masks_R_p = idx_classification == idx_R_p
                   
            fields_derived = np.zeros((no_fields_derived, no_cells[0],
                                       no_cells[1]),
                                       dtype = np.float64)
            
            mom0 = compute_moment_R_grid(0, R_p, xi_with_time[idx_t], V0,
                                         cells_with_time[idx_t],
                                         active_ids_with_time[idx_t],
                                         id_list, no_cells)
            mom1 = compute_moment_R_grid(1, R_p, xi_with_time[idx_t], V0,
                                         cells_with_time[idx_t],
                                         active_ids_with_time[idx_t],
                                         id_list, no_cells)
#            mom2 = compute_moment_R_grid(2, R_p, xi_with_time[idx_t], V0,
#                                         cells_with_time[idx_t],
#                                         active_ids_with_time[idx_t],
#                                         id_list, no_cells)
#            mom3 = compute_moment_R_grid(3, R_p, xi_with_time[idx_t], V0,
#                                         cells_with_time[idx_t],
#                                         active_ids_with_time[idx_t],
#                                         id_list, no_cells)
            
            # calculate R_eff only from cloud range (as Arabas 2015)
            mom1_cloud = compute_moment_R_grid(
                             1,
                             R_p[masks_R_p[1]],
                             xi_with_time[idx_t][masks_R_p[1]], V0,
                             cells_with_time[idx_t][:,masks_R_p[1]],
                             active_ids_with_time[idx_t][masks_R_p[1]],
                             id_list, no_cells)
            mom2_cloud = compute_moment_R_grid(
                             2,
                             R_p[masks_R_p[1]],
                             xi_with_time[idx_t][masks_R_p[1]], V0,
                             cells_with_time[idx_t][:,masks_R_p[1]],
                             active_ids_with_time[idx_t][masks_R_p[1]],
                             id_list, no_cells)
            mom3_cloud = compute_moment_R_grid(
                             3,
                             R_p[masks_R_p[1]],
                             xi_with_time[idx_t][masks_R_p[1]], V0,
                             cells_with_time[idx_t][:,masks_R_p[1]],
                             active_ids_with_time[idx_t][masks_R_p[1]],
                             id_list, no_cells)

            for cnt in range(no_fields_derived):
                idx_f = derived_indices[cnt]
                if idx_f < 6:
                    mask = np.logical_and(masks_R_p[idx_f%3],
                                          active_ids_with_time[idx_t])
                    if idx_f in range(3):
                        update_mixing_ratio(fields_derived[cnt],
                                            m_w_with_time[idx_t],
                                            xi_with_time[idx_t],
                                            cells_with_time[idx_t],
                                            mass_dry_inv, 
                                            id_list, mask)   
                    elif idx_f in range(3,6):
                        update_number_concentration_per_dry_mass(
                                fields_derived[cnt],
                                xi_with_time[idx_t],
                                cells_with_time[idx_t],
                                mass_dry_inv, 
                                id_list, mask)
                elif idx_f == 6:
                    # R_mean
                    fields_derived[cnt] = np.where(mom0 == 0.0, 0.0, mom1/mom0)
                elif idx_f == 7:
                    # R_2/1
                    fields_derived[cnt] = np.where(mom1_cloud == 0.0, 0.0,
                                                   mom2_cloud/mom1_cloud)
                elif idx_f == 8:
                    # R_eff
                    fields_derived[cnt] = np.where(mom2_cloud == 0.0, 0.0,
                                                   mom3_cloud/mom2_cloud)
            
            fields_with_time[time_n,no_fields_orig:no_fields] += \
                fields_derived
            fields_with_time_sq[time_n,no_fields_orig:no_fields] += \
                fields_derived * fields_derived
    
    fields_with_time /= no_seeds
    
    fields_with_time_std = np.sqrt((fields_with_time_sq \
                                  - no_seeds*fields_with_time*fields_with_time) \
                           / (no_seeds * (no_seeds-1)) )
    
    return fields_with_time, fields_with_time_std, \
           save_times_out, field_names_out, units_out, \
           scales_out 

def generate_moments_avg_std(load_path_list,
                             no_moments, time_indices,
                             grid_volume_cell,
                             no_cells, solute_type):

    if solute_type == "AS":
        compute_R_p_w_s_rho_p = mp.compute_R_p_w_s_rho_p_AS
    elif solute_type == "NaCl":
        compute_R_p_w_s_rho_p = mp.compute_R_p_w_s_rho_p_NaCl

    no_seeds = len(load_path_list)
    no_times = len(time_indices)
    
    moments_vs_time_all_seeds = np.zeros( (no_seeds, no_times, no_moments,
                                  no_cells[0], no_cells[1]),
                                dtype = np.float64)
    
    load_path = load_path_list[0]
    frame_every, no_grid_frames, dump_every = \
        np.load(load_path+"data_saving_paras.npy")
    grid_save_times = np.load(load_path+"grid_save_times.npy")
    
    save_times_out = np.zeros(no_times, dtype = np.int64)
    
    for time_n in range(no_times):
        idx_t = time_indices[time_n]        
        save_times_out[time_n] = grid_save_times[idx_t]
    
    V0 = grid_volume_cell
    
    for seed_n, load_path in enumerate(load_path_list):
        
        fields = load_grid_scalar_fields(load_path, grid_save_times)
        vec_data,cells_with_time,scal_data,xi_with_time,active_ids_with_time =\
            load_particle_data_all(load_path, grid_save_times)
        m_w_with_time = scal_data[:,0]
        m_s_with_time = scal_data[:,1]
        
        for time_n in range(no_times):
            idx_t = time_indices[time_n]
            
            no_SIPs = len(xi_with_time[idx_t])
            T_p = np.zeros(no_SIPs, dtype = np.float64)
            id_list = np.arange(no_SIPs)
            update_T_p(fields[idx_t, 3], cells_with_time[idx_t], T_p)
            R_p, w_s, rho_p = \
                compute_R_p_w_s_rho_p(m_w_with_time[idx_t],
                                      m_s_with_time[idx_t], T_p)
            
            for mom_n in range(no_moments):
            
                moments_vs_time_all_seeds[seed_n, time_n, mom_n] =\
                    compute_moment_R_grid(mom_n, R_p, xi_with_time[idx_t], V0,
                                             cells_with_time[idx_t],
                                             active_ids_with_time[idx_t],
                                             id_list, no_cells)

    return moments_vs_time_all_seeds, save_times_out

    
# for one seed only for now...
# load_path_list = [[load_path0]] 
# target_cell_list = [ [tgc1], [tgc2], ... ]; tgc1 = [i1, j1]
# ind_time = [it1, it2, ..] = ind. of save times belonging to tgc1, tgc2, ...
# -> to create one cycle cf with particle trajectories
# grid scalar fields have been saved in this order on hard disc
# 0 = r_v
# 1 = r_l
# 2 = Theta    
# 3 = T
# 4 = p
# 5 = S    
def generate_size_spectra_R_Arabas(load_path_list,
                                   ind_time,
                                   grid_mass_dry_inv,
                                   grid_no_cells,
                                   solute_type,
                                   target_cell_list,
                                   no_cells_x, no_cells_z,
                                   no_bins_R_p, no_bins_R_s):                                   

    no_seeds = len(load_path_list)
    no_times = len(ind_time)
    no_tg_cells = len(target_cell_list[0])
    
    load_path = load_path_list[0]
    frame_every, no_grid_frames, dump_every = \
        np.load(load_path+"data_saving_paras.npy")
    grid_save_times = np.load(load_path+"grid_save_times.npy")
    
    save_times_out = np.zeros(no_times, dtype = np.int64)
    
    R_p_list = []
    R_s_list = []
    weights_list = []
    R_min_list =[]    
    R_max_list =[]    
    
    grid_r_l_list = np.zeros( (no_times, grid_no_cells[0], grid_no_cells[1]),
                        dtype = np.float64)
    
    for tg_cell_n in range(no_tg_cells):
        R_p_list.append([])
        R_s_list.append([])
        weights_list.append([])
        save_times_out[tg_cell_n] = grid_save_times[ind_time[tg_cell_n]]
        
    for seed_n, load_path in enumerate(load_path_list):
        fields = load_grid_scalar_fields(load_path, grid_save_times)
        grid_temperature_with_time = fields[:,3]
        
        grid_r_l_with_time = fields[:,1]
        
        vec_data, cells_with_time, scal_data,\
        xi_with_time, active_ids_with_time =\
            load_particle_data_all(load_path, grid_save_times)
        m_w_with_time = scal_data[:,0]
        m_s_with_time = scal_data[:,1]    
        
        for tg_cell_n in range(no_tg_cells):
            target_cell = target_cell_list[:,tg_cell_n]
            idx_t = ind_time[tg_cell_n]
            
            
            id_list = np.arange(len(xi_with_time[idx_t]))
            
            R_p_tg, R_s_tg, xi_tg, weights_tg, no_cells_eval = \
                sample_radii_per_m_dry(m_w_with_time[idx_t],
                                       m_s_with_time[idx_t],
                                       xi_with_time[idx_t],
                                       cells_with_time[idx_t],
                                       solute_type, id_list,
                                       grid_temperature_with_time[idx_t],
                                       grid_mass_dry_inv,
                                       target_cell, no_cells_x, no_cells_z)
                
            R_p_list[tg_cell_n].append(R_p_tg)
            R_s_list[tg_cell_n].append(R_s_tg)
            weights_list[tg_cell_n].append(weights_tg)
            
            grid_r_l_list[tg_cell_n] += grid_r_l_with_time[idx_t]
    
    grid_r_l_list /= no_seeds
        
    f_R_p_list = np.zeros( (no_tg_cells, no_seeds, no_bins_R_p),
                          dtype = np.float64 )
    f_R_s_list = np.zeros( (no_tg_cells, no_seeds, no_bins_R_s),
                          dtype = np.float64 )
    bins_R_p_list = np.zeros( (no_tg_cells, no_bins_R_p+1),
                             dtype = np.float64 )
    bins_R_s_list = np.zeros( (no_tg_cells, no_bins_R_s+1),
                             dtype = np.float64 )
    
    for tg_cell_n in range(no_tg_cells):
            
        R_p_min = np.amin(np.concatenate(R_p_list[tg_cell_n]))
        R_p_max = np.amax(np.concatenate(R_p_list[tg_cell_n]))
        
        R_min_list.append(R_p_min)
        R_max_list.append(R_p_max)
        
        R_s_min = np.amin(np.concatenate(R_s_list[tg_cell_n]))
        R_s_max = np.amax(np.concatenate(R_s_list[tg_cell_n]))
        
        R_min_factor = 0.5
        R_max_factor = 2.
        
        bins_R_p = np.logspace(np.log10(R_p_min*R_min_factor),
                               np.log10(R_p_max*R_max_factor), no_bins_R_p+1 )
        
        bins_R_p_list[tg_cell_n] = np.copy(bins_R_p )
        
        bins_width_R_p = bins_R_p[1:] - bins_R_p[:-1]

        bins_R_s = np.logspace(np.log10(R_s_min*R_min_factor),
                               np.log10(R_s_max*R_max_factor), no_bins_R_s+1 )
    
        bins_width_R_s = bins_R_s[1:] - bins_R_s[:-1]

        bins_R_s_list[tg_cell_n] = np.copy(bins_R_s)
        
        for seed_n in range(no_seeds):
            R_p_tg = R_p_list[tg_cell_n][seed_n]
            R_s_tg = R_s_list[tg_cell_n][seed_n]
            weights_tg = weights_list[tg_cell_n][seed_n]
    
            h_p, b_p = np.histogram(R_p_tg, bins_R_p, weights= weights_tg)

            # convert from 1/(kg*micrometer) to unit 1/(milligram * micro_meter)
            f_R_p_list[tg_cell_n, seed_n] =\
                1E-6 * h_p / bins_width_R_p / no_cells_eval
        
            h_s, b_s = np.histogram(R_s_tg, bins_R_s, weights= weights_tg)
            
            # convert from 1/(kg*micrometer) to unit 1/(milligram * micro_meter)
            f_R_s_list[tg_cell_n, seed_n] =\
                1E-6 * h_s / bins_width_R_s / no_cells_eval
    
    return f_R_p_list, f_R_s_list, bins_R_p_list, bins_R_s_list, \
           save_times_out, grid_r_l_list, R_min_list, R_max_list

#%% ANALYZE ENSEMBLE DATA

def moments_analytical_expo(n, DNC, DNC_over_LWC):
    if n == 0:
        return DNC
    else:
        LWC_over_DNC = 1.0 / DNC_over_LWC
        return math.factorial(n) * DNC * LWC_over_DNC**n


def moments_analytical_lognormal_m(n, DNC, mu_m_log, sigma_m_log):
    if n == 0:
        return DNC
    else:
        return DNC * np.exp(n * mu_m_log + 0.5 * n*n * sigma_m_log*sigma_m_log)

def moments_analytical_lognormal_R(n, DNC, mu_R_log, sigma_R_log):
    if n == 0:
        return DNC
    else:
        return DNC * np.exp(n * mu_R_log + 0.5 * n*n * sigma_R_log*sigma_R_log)

# masses is a list of [masses0, masses1, ..., masses_no_sims]
# where masses[i] = array of masses of a spec. SIP ensemble
# use moments_an[1] for LWC0
def generate_myHisto_SIP_ensemble_np(masses, xis, m_min, m_max,
                                     dV, DNC0, LWC0,
                                     no_bins, no_sims,
                                     bin_mode, spread_mode,
                                     shift_factor, overflow_factor,
                                     scale_factor):
    # g_m_num = []
    # g_ln_r_num = []
    if bin_mode == 1:
            bin_factor = (m_max/m_min)**(1.0/no_bins)
            bin_log_dist = np.log(bin_factor)
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
            bins_mass[-1] *= 1.0001
            # the factor 0.99 is for numerical stability: to be sure
            # that m_min does not contribute to a bin smaller than the
            # 0-th bin
            bins_mass[0] *= 0.9999
            # m_0 = m_min / np.sqrt(bin_factor)
            bins_mass_log = np.log(bins_mass)

    bins_mass_width = np.zeros(no_bins+2,dtype=np.float64)
    bins_mass_width[1:-1] = bins_mass[1:]-bins_mass[:-1]
    # modify for overflow bins
    bins_mass_width[0] = bins_mass_width[1]
    bins_mass_width[-1] = bins_mass_width[-2]
    dm0 = 0.5*bins_mass_width[0]
    dmN = 0.5*bins_mass_width[-1]
    # dm0 = 0.5*(bins_mass[0] - bins_mass[0] / bin_factor)
    # dmN = 0.5*(bins_mass[-1] * bin_factor - bins_mass[-1])

    f_m_num = np.zeros( (no_sims,no_bins+2), dtype=np.float64 )
    g_m_num = np.zeros( (no_sims,no_bins), dtype=np.float64 )
    h_m_num = np.zeros( (no_sims,no_bins), dtype=np.float64 )

    for i,mass in enumerate(masses):
        histo = np.zeros(no_bins+2, dtype=np.float64)
        histo_g = np.zeros(no_bins+2, dtype=np.float64)
        histo_h = np.zeros(no_bins+2, dtype=np.float64)
        mass_log = np.log(mass)
        for n,m_ in enumerate(mass):
            xi = xis[i][n]
            bin_n = np.nonzero(np.histogram(m_, bins=bins_mass)[0])[0][0]
            # print(bin_n)

            # smear functions depending on weight of data point in the bin
            # on a lin base
            if spread_mode == 0:
                # norm_dist = (mass[n] - bins_mass[bin_n])/bins_mass_width[bin_n]
                # NEW: start from right side
                norm_dist = (bins_mass[bin_n+1] - mass[n]) \
                            / bins_mass_width[bin_n]
            # on a log base
            elif spread_mode == 1:
                # norm_dist = (mass_log[n] - bins_mass_log[bin_n])/bin_log_dist
                norm_dist = (bins_mass_log[bin_n] - mass_log[n])/bin_log_dist
            if norm_dist < 0.5:
                s = 0.5 + norm_dist

                # +1 because we have overflow bins left and right in "histo"-array
                bin_n += 1
                # print(n,s,"right")
                histo[bin_n+1] += (1.0-s)*xi
                histo_g[bin_n+1] += (1.0-s)*xi*m_
                histo_h[bin_n+1] += (1.0-s)*xi*m_*m_
                # if in last bin: no outflow,
                # just EXTRAPOLATION to overflow bin!
                if bin_n == no_bins:
                    histo[bin_n] += xi
                    histo_g[bin_n] += xi*m_
                    histo_h[bin_n] += xi*m_*m_
                else:
                    histo[bin_n] += s*xi
                    histo_g[bin_n] += s*xi*m_
                    histo_h[bin_n] += s*xi*m_*m_
            elif spread_mode == 0:
                # now left side of bin
                norm_dist = (mass[n] - bins_mass[bin_n]) \
                            / bins_mass_width[bin_n-1]
                # +1 because we have overflow bins left and right in "histo"-array
                bin_n += 1
                # print(n,norm_dist, "left")
                if norm_dist < 0.5:
                    s = 0.5 + norm_dist
                    # print(n,s,"left")
                    histo[bin_n-1] += (1.0-s)*xi
                    histo_g[bin_n-1] += (1.0-s)*xi*m_
                    histo_h[bin_n-1] += (1.0-s)*xi*m_*m_
                    # if in first bin: no outflow,
                    # just EXTRAPOLATION to overflow bin!
                    if bin_n == 1:
                        histo[bin_n] += xi
                        histo_g[bin_n] += xi*m_
                        histo_h[bin_n] += xi*m_*m_
                    else:
                        histo[bin_n] += s*xi
                        histo_g[bin_n] += s*xi*m_
                        histo_h[bin_n] += s*xi*m_*m_
                else:
                    histo[bin_n] += xi
                    histo_g[bin_n] += xi*m_
                    histo_h[bin_n] += xi*m_*m_
            elif spread_mode == 1:
                # +1 because we have overflow bins left and right in "histo"-array
                bin_n += 1
                s = 1.5 - norm_dist
                histo[bin_n] += s*xi
                histo[bin_n-1] += (1.0-s)*xi
                histo_g[bin_n] += s*xi*m_
                histo_g[bin_n-1] += (1.0-s)*xi*m_
                histo_h[bin_n] += s*xi*m_*m_
                histo_h[bin_n-1] += (1.0-s)*xi*m_*m_

            # on a log base
            # log_dist = mass_log[n] - bins_mass_log[bin_n]
            # if log_dist < bin_log_dist_half:
            #     s = 0.5 + log_dist/bin_log_dist
            #     # print(n,s,"left")
            #     histo[bin_n] += s*xi
            #     histo[bin_n-1] += (1.0-s)*xi
            #     histo_g[bin_n] += s*xi*m_
            #     histo_g[bin_n-1] += (1.0-s)*xi*m_
            # else:
            #     s = 1.5 - log_dist/bin_log_dist
            #     # print(n,s,"right")
            #     histo[bin_n] += s*xi
            #     histo[bin_n+1] += (1.0-s)*xi
            #     histo_g[bin_n] += s*xi*m_
            #     histo_g[bin_n+1] += (1.0-s)*xi*m_

        f_m_num[i,1:-1] = histo[1:-1] / (bins_mass_width[1:-1] * dV)

        # multiply the overflow-bins by factor to get an estimation of
        # f_m at the position m_0 - dm0/2
        # f_m at the position m_no_bins + dmN/2, where
        # dm0 = 0.5*(bins_mass[0] - bins_mass[0] / bin_factor)
        # dmN = 0.5*(bins_mass[-1] * bin_factor - bins_mass[-1])
        f_m_num[i,0] = overflow_factor * histo[0] / (dm0 * dV)
        f_m_num[i,-1] = overflow_factor * histo[-1] / (dmN * dV)

        g_m_num[i] = histo_g[1:-1] / (bins_mass_width[1:-1] * dV)
        h_m_num[i] = histo_h[1:-1] / (bins_mass_width[1:-1] * dV)


    f_m_num_avg = np.average(f_m_num, axis=0)
    f_m_num_std = np.std(f_m_num, axis=0, ddof=1) / np.sqrt(no_sims)
    g_m_num_avg = np.average(g_m_num, axis=0)
    g_m_num_std = np.std(g_m_num, axis=0, ddof=1) / np.sqrt(no_sims)
    h_m_num_avg = np.average(h_m_num, axis=0)
    h_m_num_std = np.std(h_m_num, axis=0, ddof=1) / np.sqrt(no_sims)
    
    # define centers on lin scale
    bins_mass_center_lin = np.zeros(no_bins+2, dtype=np.float64)
    bins_mass_center_lin[1:-1] = 0.5 * (bins_mass[:-1] + bins_mass[1:])
    # add dummy bin centers for quadratic approx
    bins_mass_center_lin[0] = bins_mass[0] - 0.5*dm0
    bins_mass_center_lin[-1] = bins_mass[-1] + 0.5*dmN

    # define centers on the logarithmic scale
    bins_mass_center_log = bins_mass[:-1] * np.sqrt(bin_factor)
    
    # define the center of mass for each bin and set it as the "bin center"
    bins_mass_center_COM = g_m_num_avg / f_m_num_avg[1:-1]

    # def as 2nd moment/1st moment
    bins_mass_center_h_g = h_m_num_avg / g_m_num_avg

    ### LINEAR APPROX OF f_m
    # to get an idea of the shape
    # for bin n take f[n-1], f[n], f[n+1]
    # make linear approx from n-1 to n and from n to n+1
    # to get idea of shape of function
    # lin fct: f = a0 + a1*m
    # a1 = (f[n+1]-f[n])/(m[n+1] - m[n])
    # a0 = f[n] - a1*m[n]
    # bins_mass_centers_lin_fit = np.zeros(no_bins, dtype = np.float64)
    lin_par0 = np.zeros(no_bins+1, dtype = np.float64)
    lin_par1 = np.zeros(no_bins+1, dtype = np.float64)

    lin_par1 = (f_m_num_avg[1:] - f_m_num_avg[:-1]) \
               / (bins_mass_center_lin[1:] - bins_mass_center_lin[:-1])
    lin_par0 = f_m_num_avg[:-1] - lin_par1 * bins_mass_center_lin[:-1]

    f_bin_border = lin_par0 + lin_par1 * bins_mass
    # f_bin_border_delta_left = np.zeros(no_bins+1, dtype = np.float64)
    # f_bin_border_delta_left = np.abs(f_m_num_avg[1:-1]-f_bin_border[:-1])
    # f_bin_border_delta_right = np.abs(f_bin_border[1:] - f_m_num_avg[1:-1])

    ### FIRST CORRECTION:
    # by my method of spreading over several bins the bins with higher f_avg
    # "loose" counts to bins with smaller f_avg
    # by a loss/gain analysis, one can estimate the lost counts
    # using the linear approximation of f_m(m) calc. above

    # delta of counts (estimated)
    delta_N = np.zeros(no_bins, dtype=np.float64)

    delta_N[1:-1] = 0.25 * bins_mass_width[1:-3] \
                    * ( f_m_num_avg[1:-3] - f_bin_border[1:-2] ) \
                    + 0.25 * bins_mass_width[2:-2] \
                      * ( -f_m_num_avg[2:-2] + f_bin_border[2:-1] ) \
                    + 0.083333333 \
                      * ( lin_par1[1:-2] * bins_mass_width[1:-3]**2
                          - lin_par1[2:-1] * bins_mass_width[2:-2]**2)
    # first bin: only exchange with the bin to the right
    delta_N[0] = 0.25 * bins_mass_width[1] \
                 * ( -f_m_num_avg[1] + f_bin_border[1] ) \
                 - 0.083333333 \
                   * ( lin_par1[1] * bins_mass_width[1]**2 )
    # last bin: only exchange with the bin to the left
    # bin_n = no_bins-1
    delta_N[no_bins-1] = 0.25 * bins_mass_width[no_bins-1] \
                         * (f_m_num_avg[no_bins-1] - f_bin_border[no_bins-1]) \
                         + 0.083333333 \
                           * ( lin_par1[no_bins-1]
                               * bins_mass_width[no_bins-1]**2 )
    scale = delta_N / (f_m_num_avg[1:-1] * bins_mass_width[1:-1])
    scale = np.where(scale < -0.9,
                     -0.9,
                     scale)
    scale *= scale_factor
    print("scale")
    print(scale)
    f_m_num_avg[1:-1] = f_m_num_avg[1:-1] / (1.0 + scale)
    f_m_num_avg[0] = f_m_num_avg[0] / (1.0 + scale[0])
    f_m_num_avg[-1] = f_m_num_avg[-1] / (1.0 + scale[-1])

    ## REPEAT LIN APPROX AFTER FIRST CORRECTION
    lin_par0 = np.zeros(no_bins+1, dtype = np.float64)
    lin_par1 = np.zeros(no_bins+1, dtype = np.float64)

    lin_par1 = (f_m_num_avg[1:] - f_m_num_avg[:-1]) \
                / (bins_mass_center_lin[1:] - bins_mass_center_lin[:-1])
    lin_par0 = f_m_num_avg[:-1] - lin_par1 * bins_mass_center_lin[:-1]

    f_bin_border = lin_par0 + lin_par1 * bins_mass

    ### SECOND CORRECTION:
    # try to estimate the position of m in the bin where f(m) = f_avg (of bin)
    # bin avg based on the linear approximations
    # NOTE that this is just to get an idea of the function FORM
    # f_bin_border_delta_left = np.zeros(no_bins+1, dtype = np.float64)
    f_bin_border_delta_left = np.abs(f_m_num_avg[1:-1]-f_bin_border[:-1])
    f_bin_border_delta_right = np.abs(f_bin_border[1:] - f_m_num_avg[1:-1])

    bins_mass_centers_lin_fit = np.zeros(no_bins, dtype = np.float64)

    f_avg2 = 0.25 * (f_bin_border[:-1] + f_bin_border[1:]) \
             + 0.5 * f_m_num_avg[1:-1]

    for bin_n in range(no_bins):
        if f_bin_border_delta_left[bin_n] >= f_bin_border_delta_right[bin_n]:
            m_c = (f_avg2[bin_n] - lin_par0[bin_n]) / lin_par1[bin_n]
        else:
            m_c = (f_avg2[bin_n] - lin_par0[bin_n+1]) / lin_par1[bin_n+1]

        # if f_bin_border_abs[bin_n] >= f_bin_border_abs[bin_n+1]:
        #     # take left side of current bin
        #     m_c = 0.5 * ( (bins_mass[bin_n] + 0.25*bins_mass_width[bin_n]) \
        #           + lin_par1[bin_n+1]/lin_par1[bin_n] \
        #             * (bins_mass[bin_n+1] - 0.25*bins_mass_width[bin_n]) \
        #           + (lin_par0[bin_n+1] - lin_par0[bin_n]))
        # else:
        #     m_c = 0.5 * ( lin_par1[bin_n]/lin_par1[bin_n+1] \
        #                   * (bins_mass[bin_n] + 0.25*bins_mass_width[bin_n]) \
        #                   + (bins_mass[bin_n+1] - 0.25*bins_mass_width[bin_n])\
        #                   + (lin_par0[bin_n] - lin_par0[bin_n+1]) )
        # add additional shift because of two effects:
        # 1) adding xi-"mass" to bins with smaller f_avg
        # 2) wrong setting of "center" if f_avg[n] > f_avg[n+1]

        m_c = shift_factor * m_c \
              + bins_mass_center_lin[bin_n+1] * (1.0 - shift_factor)

        if m_c < bins_mass[bin_n]:
            m_c = bins_mass[bin_n]
        elif m_c > bins_mass[bin_n+1]:
            m_c = bins_mass[bin_n+1]

        bins_mass_centers_lin_fit[bin_n] = m_c
        # shift more to center: -> is covered by shift_factor=0.5
        # bins_mass_centers_lin_fit[bin_n] = \
        #     0.5 * (m_c + bins_mass_center_lin[bin_n+1])


    ### bin mass center quad approx: -->>> BIG ISSUES: no monoton. interpol.
    # possible for three given points with quadr. fct.
    # for every bin:
    # assume that the coordinate pairs are right with
    # (m_center_lin, f_avg)
    # approximate the function f_m(m) locally with a parabola to get
    # an estimate of the form of the function
    # assume this parabola in the bin and calculate bin_center_exact


    D_10 = bins_mass_center_lin[1:-1] - bins_mass_center_lin[0:-2]
    D_20 = bins_mass_center_lin[2:] - bins_mass_center_lin[0:-2]
    D_21 = bins_mass_center_lin[2:] - bins_mass_center_lin[1:-1]

    CD_10 = (bins_mass_center_lin[1:-1] + bins_mass_center_lin[0:-2])*D_10
    CD_20 = (bins_mass_center_lin[2:] + bins_mass_center_lin[0:-2])*D_20
    CD_21 = (bins_mass_center_lin[2:] + bins_mass_center_lin[1:-1])*D_21

    a2 = f_m_num_avg[2:]/(D_21*D_20) - f_m_num_avg[1:-1]/(D_21*D_10) \
         + f_m_num_avg[:-2]/(D_10*D_20)
    a1_a2 = (-f_m_num_avg[0:-2]*CD_21 + f_m_num_avg[1:-1]*CD_20
             - f_m_num_avg[2:]*CD_10  ) \
            / (f_m_num_avg[0:-2]*D_21 - f_m_num_avg[1:-1]*D_20
               + f_m_num_avg[2:]*D_10  )
    a1 = a2 * a1_a2
    a0 = f_m_num_avg[1:-1] - a1*bins_mass_center_lin[1:-1] \
         - a2*bins_mass_center_lin[1:-1]**2

    bins_mass_sq = bins_mass*bins_mass

    bins_mass_centers_qfit =\
        -0.5*a1_a2 \
        + np.sqrt( 0.25*(a1_a2)**2
                   + 0.5*a1_a2 * (bins_mass[:-1] + bins_mass[1:])
                   + 0.33333333 * (bins_mass_sq[:-1]
                                   + bins_mass[:-1]*bins_mass[1:]
                                   + bins_mass_sq[1:]) )

    bins_mass_center_lin2 = bins_mass_center_lin[1:-1]

    bins_mass_width = bins_mass_width[1:-1]
    
    # set the bin "mass centers" at the right spot for exponential dist
    # such that f_avg_i in bin in = f(mm_i), where mm_i is the "mass center"
    # use moments_an[1] for LWC0 if not given (e.g. for lognormal distr.)
    m_avg = LWC0 / DNC0
    bins_mass_center_exact = bins_mass[:-1]\
                             + m_avg * np.log(bins_mass_width\
          / (m_avg * (1-np.exp(-bins_mass_width/m_avg))))

    bins_mass_centers = np.array((bins_mass_center_lin2,
                                  bins_mass_center_log,
                                  bins_mass_center_COM,
                                  bins_mass_center_exact,
                                  bins_mass_centers_lin_fit,
                                  bins_mass_centers_qfit,
                                  bins_mass_center_h_g))

    return f_m_num_avg, f_m_num_std, g_m_num_avg, g_m_num_std,\
           h_m_num_avg, h_m_num_std, \
           bins_mass, bins_mass_width, \
           bins_mass_centers, bins_mass_center_lin, \
           np.array((lin_par0,lin_par1)), np.array((a0,a1,a2))
# generate_myHisto_SIP_ensemble = njit()(generate_myHisto_SIP_ensemble_np)

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
        moments_analytical = moments_analytical_expo
    elif dist =="lognormal":
        conc_per_mass_np = conc_per_mass_lognormal_np
        dV, DNC0, mu_m_log, sigma_m_log, mass_density, r_critmin, \
        kappa, eta, no_sims00, start_seed = \
            tuple(np.load(ensemble_dir + "ensemble_parameters.npy"))
        dist_par = (DNC0, mu_m_log, sigma_m_log)
        moments_analytical = moments_analytical_lognormal_m

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
        
    print(f"######## kappa {kappa} ########")    
    print("moments_an: ", moments_an)    
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
    
#    no_SIPs_avg_auto = f_m_counts_auto.sum()/no_sims

    bins_mass_ind_auto = np.append(f_m_ind_auto, f_m_ind_auto[-1]+1)
    
    bins_mass_auto = bins_mass_auto[bins_mass_ind_auto]
    
    bins_rad_auto = bins_rad_auto[bins_mass_ind_auto]
    bins_rad_log_auto = np.log(bins_rad_auto)
    bins_mass_width_auto = (bins_mass_auto[1:]-bins_mass_auto[:-1])
#    bins_rad_width_auto = (bins_rad_auto[1:]-bins_rad_auto[:-1])
    bins_rad_width_log_auto = (bins_rad_log_auto[1:]-bins_rad_log_auto[:-1])
    
    ### approximate the functions f_m, f_lnR = 3*m*f_m, g_lnR=3*m^2*f_m
    # estimate f_m(m) by binning:
    # DNC_i = f_m(m_i) * dm_i = droplet number conc in bin i with size dm_i
    f_m_num_sampled_auto = np.histogram(masses_sampled,bins_mass_auto,
                                   weights=xis_sampled)[0]
    g_m_num_sampled_auto = np.histogram(masses_sampled,bins_mass_auto,
                                   weights=xis_sampled*masses_sampled)[0]
    
    f_m_num_sampled_auto = f_m_num_sampled_auto / (bins_mass_width_auto * dV * no_sims)
    g_m_num_sampled_auto = g_m_num_sampled_auto / (bins_mass_width_auto * dV * no_sims)
    
    # build g_ln_r = 3*m*g_m DIRECTLY from data
    g_ln_r_num_sampled_auto = np.histogram(radii_sampled,
                                      bins_rad_auto,
                                      weights=xis_sampled*masses_sampled)[0]
    g_ln_r_num_sampled_auto = g_ln_r_num_sampled_auto \
                         / (bins_rad_width_log_auto * dV * no_sims)
    # g_ln_r_num_derived = 3 * bins_mass_center * g_m_num * 1000.0
    
    # define centers on lin scale
    bins_mass_center_lin_auto = 0.5 * (bins_mass_auto[:-1] + bins_mass_auto[1:])
    bins_rad_center_lin_auto = 0.5 * (bins_rad_auto[:-1] + bins_rad_auto[1:])
    
    # define centers on the logarithmic scale
    bins_mass_center_log_auto = bins_mass_auto[:-1] * np.sqrt(bin_factor)
    bins_rad_center_log_auto = bins_rad_auto[:-1] * np.sqrt(bin_factor)
    # bins_mass_center_log = bins_mass[:-1] * 10**(1.0/(2.0*kappa))
    # bins_rad_center_log = bins_rad[:-1] * 10**(1.0/(2.0*kappa))
    
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

    ###################################################



    ###################################################
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
    
    ###################################################
    ### STATISTICAL ANALYSIS OVER no_sim runs AUTO BINS
    # get f(m_i) curve for each "run" with same bins for all ensembles
    f_m_num_auto = []
    g_m_num_auto = []
    g_ln_r_num_auto = []
    
    for i,mass in enumerate(masses):
        f_m_num_auto.append(np.histogram(mass,bins_mass_auto,weights=xis[i])[0] \
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
    g_ln_r_num_std_auto = np.std(g_ln_r_num_auto, axis=0, ddof=1) / np.sqrt(no_sims)

##############################################################################
    
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
    
##############################################################################

    # analytical reference data    
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
            f_m_num_avg_auto, f_m_num_std_auto, g_m_num_avg_auto, g_m_num_std_auto, 
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
#           bins_mass_width_auto, bins_rad_width_auto, bins_rad_width_log_auto, \
#           bins_mass_centers_auto, bins_rad_centers_auto, \
#           f_m_counts_auto, f_m_ind_auto,\
#           f_m_num_sampled_auto, g_m_num_sampled_auto, g_ln_r_num_sampled_auto,\
#           f_m_num_avg_auto, f_m_num_std_auto, g_m_num_avg_auto, g_m_num_std_auto, \
#           g_ln_r_num_avg_auto, g_ln_r_num_std_auto, \
#           moments_sampled, moments_sampled_avg_norm,moments_sampled_std_norm,\
#           moments_an, \
#           f_m_num_avg_my_ext, \
#           f_m_num_avg_my, f_m_num_std_my, \
#           g_m_num_avg_my, g_m_num_std_my, \
#           h_m_num_avg_my, h_m_num_std_my, \
#           bins_mass_my, bins_mass_width_my, \
#           bins_mass_centers_my, bins_mass_center_lin_my, lin_par, aa

