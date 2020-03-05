#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TROPOS LAGRANGIAN CLOUD MODEL
Super-Droplet method in two-dimensional kinematic framework
(Test Case 1 ICMW 2012, Muhlbauer et al. 2013)
Author: Jan Bohrer (bohrer@tropos.de)
Further contact: Oswald Knoth (knoth@tropos.de)

DATA ANALYSIS AND PROCESSING FOR PLOT GENERATION
Provides data plotable with "plot_results.py"

basic units:
particle mass, water mass, solute mass in femto gram = 10^-18 kg
particle radius in micro meter ("mu")
all other quantities in SI units
"""

#%% MODULE IMPORTS
import os
import shutil
import sys
import numpy as np

import constants as c
from microphysics import compute_R_p_w_s_rho_p
from microphysics import compute_radius_from_mass_vec
from analysis import generate_field_frame_data_avg
from analysis import generate_size_spectra_R_Arabas
from analysis import generate_moments_avg_std
from file_handling import load_grid_and_particles_full 

#%% DATA PARENT DIRECTORY

#simdata_path = '/Users/bohrer/sim_data_cloudMP/'
simdata_path = '/vols/fs1/work/bohrer/sim_data_cloudMP/'

if len(sys.argv) > 1:
    simdata_path = sys.argv[1]

#%% GRID PARAMETERS

no_cells = np.array((75, 75))

if len(sys.argv) > 2:
    no_cells[0] = int(sys.argv[2])
if len(sys.argv) > 3:
    no_cells[1] = int(sys.argv[3])

#%% PARTICLE PARAMETERS

#solute_type = 'NaCl'
solute_type = 'AS' # 'AS' (ammon. sulf.) or 'NaCl'

if len(sys.argv) > 4:
    solute_type = sys.argv[4]

# average number of super particles per cell and mode [mode1, mode2]
no_spcm = np.array([26, 38])

if len(sys.argv) > 5:
    no_spcm[0] = int(sys.argv[5])
if len(sys.argv) > 6:
    no_spcm[1] = int(sys.argv[6])

no_seeds = 50

if len(sys.argv) > 7:
    no_seeds = int(sys.argv[7])

# first random number seed for SIP generation 
seed_SIP_gen = 9001

if len(sys.argv) > 8:
    seed_SIP_gen = int(sys.argv[8])

seed_SIP_gen_list = np.arange(seed_SIP_gen, seed_SIP_gen + no_seeds * 2, 2)

# first random number seed for particle collisions
seed_sim = 9001

if len(sys.argv) > 9:
    seed_sim = int(sys.argv[9])

seed_sim_list = np.arange(seed_sim, seed_sim + no_seeds * 2, 2)

#%% SIMULATION PARAMETERS

# options: 'spin_up', 'with_collisions', 'wo_collisions'
simulation_mode = 'with_collisions'
if len(sys.argv) > 10:
    simulation_mode = sys.argv[10]
    
spin_up_finished = True

# grid load time 
t_grid = 10800
# time interval for the data analysis
t_start = 7200
t_end = 10800

if len(sys.argv) > 11:
    t_grid = float(sys.argv[11])
if len(sys.argv) > 12:
    t_start = float(sys.argv[12])
if len(sys.argv) > 13:
    t_end = float(sys.argv[13])

dt = 1.0 # timestep of advection (seconds)

# number of condensation steps per advection step (only even integers)
no_cond_per_adv = 10

# number of collision steps per advection step
# possible values: 1, 2 OR no_cond_per_adv
no_col_per_adv = 2

if len(sys.argv) > 14:
    no_col_per_adv = int(sys.argv[14])

dt_col = dt / no_col_per_adv

#%% ANALYSIS PARAMETERS

### GRID FRAMES
# set indices of quantities, which shall be available
# possible field indices:
# 0: r_v
# 1: r_l
# 2: Theta
# 3: T
# 4: p
# 5: S
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
field_ind = np.array((2, 5, 0, 1))
field_ind_deri = np.array((0, 1, 2, 3, 4, 5, 6, 7, 8))

# "data frames" where stored every 'frame every' steps
# set indices of the "data frames" to be analyzed as list/array
#time_ind_grid = np.array((0, 2, 4, 6, 8, 10, 12))
time_ind_grid = np.arange(0, 13, 2)

### SPECTRA
# target cells for spectra analysis
# corresponding to Arabas 2015
i_tg = [16, 58]
j_tg = [27, 44, 46, 51, 72][::-1]

no_rows = len(j_tg)
no_cols = len(i_tg)

# target list from ordered mesh grid
i_list, j_list = np.meshgrid(i_tg, j_tg, indexing = 'xy')
target_cell_list = np.array([i_list.flatten(), j_list.flatten()])

# "averaging box size" for spectra:
# spectra are averaged over regions of no_cells_x * no_cells_z grid cells
# around the target cells
# please enter uneven numbers: no_cells_x = 5 =>  [x][x][tg cell][x][x]
no_cells_x = 3
no_cells_z = 3

# time indices for the spectra analysis
# time indices may be chosen individually for each spectrum
# where the cell of each spectrum is given in target_cell_list (s.a.)
#ind_time = np.array((0,2,4,6,8,10,12))
ind_time = 6 * np.ones(len(target_cell_list[0]), dtype = np.int64)

# number of bins for wet (R_p) and dry (R_s) size spectra
no_bins_R_p = 30
no_bins_R_s = 30

### EXTRACTION OF GRID DATA
grid_times = [0, 7200, 10800] # (seconds)
duration_spin_up = 7200 # (seconds)

### PARAMETERS FOR MOMENT GENERATION
# number of moments
no_moments = 4
time_ind_moments = np.arange(0, 13, 2)

#%% DERIVED AND FIX PARAMETERS

args_gen = [1,1,1,1]

act_gen_grid_frames_avg = args_gen[0]
act_gen_spectra_avg_Arabas = args_gen[1]
act_get_grid_data = args_gen[2]
act_gen_moments_all_grid_cells = args_gen[3]

#%% LOAD GRID AND PARTICLES AT TIME t_grid

grid_folder = f'{solute_type}'\
    + f'/grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm[0]}_{no_spcm[1]}/' \
    + f'{seed_SIP_gen}/'

if simulation_mode == 'spin_up':
    save_folder = 'spin_up_wo_col_wo_grav/'
elif simulation_mode == 'wo_collision':
    if spin_up_finished:
        save_folder = 'w_spin_up_wo_col/'
    else:
        save_folder = 'wo_spin_up_wo_col/'
elif simulation_mode == 'with_collisions':
    if spin_up_finished:
        save_folder = f'w_spin_up_w_col/{seed_sim}/'
    else:
        save_folder = f'wo_spin_up_w_col/{seed_sim}/'

# load grid and particles full from grid_path at time t
if int(t_grid) == 0:        
    grid_path = simdata_path + grid_folder
elif int(t_grid) <= duration_spin_up:        
    grid_path = simdata_path + grid_folder + 'spin_up_wo_col_wo_grav/'
else:    
    grid_path = simdata_path + grid_folder + save_folder

load_path = simdata_path + grid_folder + save_folder

grid, pos, cells, vel, m_w, m_s, xi, active_ids  = \
    load_grid_and_particles_full(t_grid, grid_path)

if solute_type == 'AS':
#    compute_R_p_w_s_rho_p = compute_R_p_w_s_rho_p_AS
    mass_density_dry = c.mass_density_AS_dry
elif solute_type == 'NaCl':
#    compute_R_p_w_s_rho_p = compute_R_p_w_s_rho_p_NaCl
    mass_density_dry = c.mass_density_NaCl_dry

R_p, w_s, rho_p = compute_R_p_w_s_rho_p(m_w, m_s,
                                        grid.temperature[tuple(cells)],
                                        solute_type)
R_s = compute_radius_from_mass_vec(m_s, mass_density_dry)

#%% GENERATE GRID FRAMES AVG

show_target_cells = True

if act_gen_grid_frames_avg:
    

    load_path_list = []    

    for seed_n in range(no_seeds):
        seed_SIP_gen_ = seed_SIP_gen_list[seed_n]
        seed_sim_ = seed_sim_list[seed_n]
        
        grid_folder_ =\
            f'{solute_type}' \
            + f'/grid_{no_cells[0]}_{no_cells[1]}_'\
            + f'spcm_{no_spcm[0]}_{no_spcm[1]}/' \
            + f'{seed_SIP_gen_}/'
        
        if simulation_mode == 'spin_up':
            save_folder_ = 'spin_up_wo_col_wo_grav/'
        elif simulation_mode == 'wo_collision':
            if spin_up_finished:
                save_folder_ = 'w_spin_up_wo_col/'
            else:
                save_folder_ = 'wo_spin_up_wo_col/'
        elif simulation_mode == 'with_collisions':
            if spin_up_finished:
                save_folder_ = f'w_spin_up_w_col/{seed_sim_}/'
            else:
                save_folder_ = f'wo_spin_up_w_col/{seed_sim_}/'        
                load_path_list.append()
        
        load_path_list.append(simdata_path + grid_folder_ + save_folder_)
        
    fields_with_time, fields_with_time_std, save_times_out,\
    field_names_out, units_out, scales_out = \
        generate_field_frame_data_avg(load_path_list,
                                        field_ind, time_ind_grid,
                                        field_ind_deri,
                                        grid.mass_dry_inv,
                                        grid.volume_cell,
                                        grid.no_cells,
                                        solute_type)
    ### create only plotting data output to be transfered
    output_folder = \
        f'{solute_type}' \
        + f'/grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm[0]}_{no_spcm[1]}/'\
        + f'eval_data_avg_Ns_{no_seeds}_' \
        + f'sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}/'
    
    if not os.path.exists(simdata_path + output_folder):
        os.makedirs(simdata_path + output_folder)    
    
    np.save(simdata_path + output_folder
            + 'seed_SIP_gen_list',
            seed_SIP_gen_list)
    np.save(simdata_path + output_folder
            + 'seed_sim_list',
            seed_sim_list)
    
    np.save(simdata_path + output_folder
            + f'fields_vs_time_avg_Ns_{no_seeds}_'
            + f'sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}',
            fields_with_time)
    np.save(simdata_path + output_folder
            + f'fields_vs_time_std_Ns_{no_seeds}_'
            + f'sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}',
            fields_with_time_std)
    np.save(simdata_path + output_folder
            + f'save_times_out_avg_Ns_{no_seeds}_'
            + f'sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}',
            save_times_out)
    np.save(simdata_path + output_folder
            + f'field_names_out_avg_Ns_{no_seeds}_'
            + f'sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}',
            field_names_out)
    np.save(simdata_path + output_folder
            + f'units_out_avg_Ns_{no_seeds}_'
            + f'sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}',
            units_out)
    np.save(simdata_path + output_folder
            + f'scales_out_avg_Ns_{no_seeds}_'
            + f'sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}',
            scales_out)

    print('generated average grid frames')
    print('first load path:')
    print(load_path_list[0])    
    print('time indices grid frames:')
    print(time_ind_grid)    
    
#%% GENERATE SPECTRA AVG 

if act_gen_spectra_avg_Arabas:
    
    
    load_path_list = []    
    no_seeds = len(seed_SIP_gen_list)
    for seed_n in range(no_seeds):
        seed_SIP_gen_ = seed_SIP_gen_list[seed_n]
        seed_sim_ = seed_sim_list[seed_n]
        
        grid_folder_ =\
            f'{solute_type}' \
            + f'/grid_{no_cells[0]}_{no_cells[1]}_' \
            + f'spcm_{no_spcm[0]}_{no_spcm[1]}/' \
            + f'{seed_SIP_gen_}/'
        
        if simulation_mode == 'spin_up':
            save_folder_ = 'spin_up_wo_col_wo_grav/'
        elif simulation_mode == 'wo_collision':
            if spin_up_finished:
                save_folder_ = 'w_spin_up_wo_col/'
            else:
                save_folder_ = 'wo_spin_up_wo_col/'
        elif simulation_mode == 'with_collisions':
            if spin_up_finished:
                save_folder_ = f'w_spin_up_w_col/{seed_sim_}/'
            else:
                save_folder_ = f'wo_spin_up_w_col/{seed_sim_}/'        
                load_path_list.append()
        
        load_path_list.append(simdata_path + grid_folder_ + save_folder_)
    
    f_R_p_list, f_R_s_list, bins_R_p_list, bins_R_s_list, save_times_out,\
    grid_r_l_list, R_min_list, R_max_list = \
        generate_size_spectra_R_Arabas(load_path_list,
                                       ind_time,
                                       grid.mass_dry_inv,
                                       grid.no_cells,
                                       solute_type,
                                       target_cell_list,
                                       no_cells_x, no_cells_z,
                                       no_bins_R_p, no_bins_R_s)  
    
    output_folder = \
        f'{solute_type}' \
        + f'/grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm[0]}_{no_spcm[1]}/'\
        + f'eval_data_avg_Ns_{no_seeds}_' \
        + f'sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}/'
    
    if not os.path.exists(simdata_path + output_folder):
        os.makedirs(simdata_path + output_folder)    
    
    np.save(simdata_path + output_folder
            + 'seed_SIP_gen_list',
            seed_SIP_gen_list)
    np.save(simdata_path + output_folder
            + 'seed_sim_list',
            seed_sim_list)
    
    np.save(simdata_path + output_folder
            + f'f_R_p_list_avg_Ns_{no_seeds}_'
            + f'sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}',
            f_R_p_list)    
    np.save(simdata_path + output_folder
            + f'f_R_s_list_avg_Ns_{no_seeds}_'
            + f'sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}',
            f_R_s_list)    
    np.save(simdata_path + output_folder
            + f'bins_R_p_list_avg_Ns_{no_seeds}_'
            + f'sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}',
            bins_R_p_list)    
    np.save(simdata_path + output_folder
            + f'bins_R_s_list_avg_Ns_{no_seeds}_'
            + f'sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}',
            bins_R_s_list)    
    np.save(simdata_path + output_folder
            + f'save_times_out_spectra_avg_Ns_{no_seeds}_'
            + f'sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}',
            save_times_out)    
    np.save(simdata_path + output_folder
            + f'grid_r_l_list_avg_Ns_{no_seeds}_'
            + f'sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}',
            grid_r_l_list)    
    np.save(simdata_path + output_folder
            + f'R_min_list_avg_Ns_{no_seeds}_'
            + f'sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}',
            R_min_list)    
    np.save(simdata_path + output_folder
            + f'R_max_list_avg_Ns_{no_seeds}_'
            + f'sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}',
            R_max_list)    
    np.save(simdata_path + output_folder
            + f'target_cell_list_avg_Ns_{no_seeds}_'
            + f'sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}',
            target_cell_list)    
    np.save(simdata_path + output_folder
            + f'neighbor_cells_list_avg_Ns_{no_seeds}_'
            + f'sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}',
            [no_cells_x, no_cells_z])    
    np.save(simdata_path + output_folder
            + f'no_rows_no_cols_avg_Ns_{no_seeds}_'
            + f'sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}',
            [no_rows, no_cols])    
    np.save(simdata_path + output_folder
            + f'no_bins_p_s_avg_Ns_{no_seeds}_'
            + f'sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}',
            [no_bins_R_p, no_bins_R_s])
    
    print('generated spectra in target cells')
    print('target_cell_list:')
    print(target_cell_list)
    
#%% EXTRACT GRID DATA
    
if act_get_grid_data:
    output_path0 = \
        simdata_path \
        + f'{solute_type}' \
        + f'/grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm[0]}_{no_spcm[1]}/'\
        + f'eval_data_avg_Ns_{no_seeds}_' \
        + f'sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}/grid_data/'    

    grid_path_base0 = \
        simdata_path \
        + f'{solute_type}' \
        + f'/grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm[0]}_{no_spcm[1]}/'

    no_grid_times = len(grid_times)

    output_folder = \
        f'{solute_type}' \
        + f'/grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm[0]}_{no_spcm[1]}/'\
        + f'eval_data_avg_Ns_{no_seeds}_' \
        + f'sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}/'

    np.save(simdata_path + output_folder
            + 'seed_SIP_gen_list',
            seed_SIP_gen_list)
    np.save(simdata_path + output_folder
            + 'seed_sim_list',
            seed_sim_list)
    
    for seed_n in range(no_seeds):
        s1 = f'{seed_SIP_gen_list[seed_n]}'
        s2 = f'{seed_sim_list[seed_n]}'
        grid_path_base = grid_path_base0 + f'{seed_SIP_gen_list[seed_n]}/'
        output_path = output_path0 + f'{s1}_{s2}/'
        
        if not os.path.exists(output_path):
            os.makedirs(output_path) 
            
        for gt in grid_times:
            if gt == 0:
                shutil.copy(grid_path_base + 'grid_basics_0.txt',
                            output_path)
                shutil.copy(grid_path_base + 'arr_file1_0.npy', output_path)
                shutil.copy(grid_path_base + 'arr_file2_0.npy', output_path)
            elif gt <= duration_spin_up:
                shutil.copy(grid_path_base + 'spin_up_wo_col_wo_grav/'
                             + f'grid_basics_{int(gt)}.txt',
                             output_path)
                shutil.copy(grid_path_base + 'spin_up_wo_col_wo_grav/'
                             + f'arr_file1_{int(gt)}.npy',
                             output_path)
                shutil.copy(grid_path_base + 'spin_up_wo_col_wo_grav/'
                             + f'arr_file2_{int(gt)}.npy',
                             output_path)
            elif gt > duration_spin_up:
                shutil.copy(grid_path_base
                             + f'w_spin_up_w_col/{seed_sim_list[seed_n]}/'
                             + f'grid_basics_{int(gt)}.txt',
                             output_path)
                shutil.copy(grid_path_base
                             + f'w_spin_up_w_col/{seed_sim_list[seed_n]}/'
                             + f'arr_file1_{int(gt)}.npy',
                             output_path)
                shutil.copy(grid_path_base
                             + f'w_spin_up_w_col/{seed_sim_list[seed_n]}/'
                             + f'arr_file2_{int(gt)}.npy',
                             output_path)    
    print('extracted grid data for times:')
    print(grid_times)

#%% GENERATE MOMENTS FOR ALL GRID CELLS

if act_gen_moments_all_grid_cells:
    
    
    load_path_list = []    

    for seed_n in range(no_seeds):
        seed_SIP_gen_ = seed_SIP_gen_list[seed_n]
        seed_sim_ = seed_sim_list[seed_n]
        
        grid_folder_ =\
            f'{solute_type}' \
            + f'/grid_{no_cells[0]}_{no_cells[1]}_' \
            + f'spcm_{no_spcm[0]}_{no_spcm[1]}/' \
            + f'{seed_SIP_gen_}/'
        
        if simulation_mode == 'spin_up':
            save_folder_ = 'spin_up_wo_col_wo_grav/'
        elif simulation_mode == 'wo_collision':
            if spin_up_finished:
                save_folder_ = 'w_spin_up_wo_col/'
            else:
                save_folder_ = 'wo_spin_up_wo_col/'
        elif simulation_mode == 'with_collisions':
            if spin_up_finished:
                save_folder_ = f'w_spin_up_w_col/{seed_sim_}/'
            else:
                save_folder_ = f'wo_spin_up_w_col/{seed_sim_}/'        
                load_path_list.append()
        
        load_path_list.append(simdata_path + grid_folder_ + save_folder_)

    moments_vs_time_all_seeds, save_times_out = \
        generate_moments_avg_std(load_path_list,
                               no_moments, time_ind_moments,
                               grid.volume_cell,
                               no_cells, solute_type)
    
    ### create plotting data to be transfered
    output_folder = \
        f'{solute_type}' \
        + f'/grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm[0]}_{no_spcm[1]}/'\
        + f'eval_data_avg_Ns_{no_seeds}_' \
        + f'sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}/moments/'
    
    if not os.path.exists(simdata_path + output_folder):
        os.makedirs(simdata_path + output_folder)    
    
    np.save(simdata_path + output_folder
            + 'seed_SIP_gen_list',
            seed_SIP_gen_list)
    np.save(simdata_path + output_folder
            + 'seed_sim_list',
            seed_sim_list)
    
    np.save(simdata_path + output_folder
            + f'moments_vs_time_all_seeds_Ns_{no_seeds}_'
            + f'sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}',
            moments_vs_time_all_seeds)
    
    np.save(simdata_path + output_folder
            + f'save_times_out_avg_Ns_{no_seeds}_'
            + f'sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}',
            save_times_out)

    print('generated moments in all grid cells')    
    print('first load path:')    
    print(load_path_list[0])    
    print('time indices moments:')
    print(time_ind_moments)
