#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 15:43:10 2019

@author: jdesk
"""

#%% MODULE IMPORTS & LOAD GRID
import os
import numpy as np
import matplotlib.pyplot as plt

#%% STORAGE DIRECTORIES

simdata_path = "/Users/bohrer/sim_data_cloudMP/"
home_path = "/Users/bohrer/"

#%% CHOOSE OPERATIONS

args_plot = [1,0,0,0,0]
#args_plot = [0,1,0,0,0]
#args_plot = [0,0,0,1,0]
#args_plot = [0,1,1,1,0]
#args_plot = [0,0,0,0,1]

act_plot_grid_frames_avg = args_plot[0]
act_plot_grid_frames_avg_shift = args_plot[1]
act_plot_spectra_avg_Arabas = args_plot[2]
act_plot_grid_frames_INIT = args_plot[3]
act_plot_grid_frames_avg_compare = args_plot[4]

#%% GRID PARAMETERS

# needed for filename
#no_cells = (10, 10)
#no_cells = (15, 15)
no_cells = (75, 75)

# for plots, the periodic grid can be shifted horizont. by shift_cells_x cells
shift_cells_x = 19
#shift_cells_x = 56

#%% PARTICLE PARAMETERS

# solute material: NaCl OR ammonium sulfate
#solute_type = "NaCl"
solute_type = "AS"

# no_super_particles_cell_mode = [N1,N2] is a list with
# N1 = no super part. per cell in mode 1 etc.
# with init method = SingleSIP, this is only the target value.
# the true number of particles per cell and mode will fluctuate around this
#no_spcm = np.array([2, 2])
#no_spcm = np.array([6, 10])
#no_spcm = np.array([16, 24])
#no_spcm = np.array([20, 30])
no_spcm = np.array([26, 38])
#no_spcm = np.array([52, 76])

seed_SIP_gen = 1001
seed_sim = 1001

#no_seeds = 1
#no_seeds = 10
#no_seeds = 20
#no_seeds = 30
no_seeds = 50

### WHEN COMPARING TWO SIMULATIONS:
solute_type2 = "AS"

# no_super_particles_cell_mode = [N1,N2] is a list with
# N1 = no super part. per cell in mode 1 etc.
# with init method = SingleSIP, this is only the target value.
# the true number of particles per cell and mode will fluctuate around this
#no_spcm = np.array([2, 2])
#no_spcm = np.array([6, 10])
no_spcm2 = np.array([16, 24])
#no_spcm = np.array([20, 30])
#no_spcm = np.array([26, 38])
#no_spcm = np.array([52, 76])

seed_SIP_gen2 = 2101
seed_sim2 = 2101

#no_seeds = 1
#no_seeds = 10
#no_seeds = 20
no_seeds2 = 30
#no_seeds = 50


#%% SIM PARAMETERS

#simulation_mode = "spin_up"
#simulation_mode = "wo_collision"
simulation_mode = "with_collision"

# for file names
dt_col = 0.5
#dt_col = 1.0

# grid load time
# path = simdata_path + folder_load_base
t_grid = 0
#t_grid = 7200
#t_grid = 10800
#t_grid = 14400

#t_start = 0
#t_start = 3600
t_start = 7200

#t_end = 60
#t_end = 600
#t_end = 3600
#t_end = 7200
t_end = 10800
#t_end = 14400

#%% PLOTTING PARAMETERS

### GRID FRAMES
show_target_cells = True

### SPECTRA
# if set to "None" the values are loaded from stored files
#no_rows = 5
#no_cols = 3
no_rows = None
no_cols = None

# if set to "None" the values are loaded from stored files
#no_bins_R_p = 30
#no_bins_R_s = 30
no_bins_R_p = None
no_bins_R_s = None

#%% LOAD GRID AND SET PATHS
data_generated_with_gen_plot_data = True

if data_generated_with_gen_plot_data:
    data_folder = \
        f"{solute_type}" \
        + f"/grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm[0]}_{no_spcm[1]}/"\
        + f"eval_data_avg_Ns_{no_seeds}_" \
        + f"sg_{seed_SIP_gen}_ss_{seed_sim}/"
    data_path = simdata_path + data_folder
    grid_path = simdata_path + data_folder + "grid_data/" \
                + f"{seed_SIP_gen}_{seed_sim}/"
else:        
    data_folder = \
        f"{solute_type}" \
        + f"/grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm[0]}_{no_spcm[1]}/"\
        + f"{seed_SIP_gen}/"
    data_path = simdata_path + data_folder
    grid_path = simdata_path + data_folder

from file_handling import load_grid_from_files

grid = load_grid_from_files(grid_path + f"grid_basics_{int(t_grid)}.txt",
                            grid_path + f"arr_file1_{int(t_grid)}.npy",
                            grid_path + f"arr_file2_{int(t_grid)}.npy")

grid.print_info()

seed_SIP_gen_list = np.load(data_path + "seed_SIP_gen_list.npy" )
seed_sim_list = np.load(data_path + "seed_sim_list.npy")

target_cell_list = np.load(data_path
        + f"target_cell_list_avg_Ns_{no_seeds}_"
        + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy")

no_neighbor_cells = np.load(data_path
        + f"neighbor_cells_list_avg_Ns_{no_seeds}_"
        + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
        )       
no_cells_x = no_neighbor_cells[0]
no_cells_z = no_neighbor_cells[1]

if no_cells_x % 2 == 0: no_cells_x += 1
if no_cells_z % 2 == 0: no_cells_z += 1  

no_tg_cells = len(target_cell_list[0])   

#%% PLOT INIT GRID FRAMES
if act_plot_grid_frames_INIT:
    fig_dir = data_path + f"plots_{simulation_mode}_dt_col_{dt_col}/"
#    fig_name = \
#               f"grid_init_frames_" \
#               + f"t_{t_grid}.png"
    grid.plot_thermodynamic_scalar_fields(fig_dir = fig_dir)
    print("plotted initial grid frames at t = ", t_grid)

#%% PLOT AVG GRID FRAMES

if act_plot_grid_frames_avg:
    from analysis import plot_scalar_field_frames_extend_avg    

    fields_with_time = np.load(data_path
            + f"fields_vs_time_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )
    save_times_out_fr = np.load(data_path
            + f"save_times_out_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )
    field_names_out = np.load(data_path
            + f"field_names_out_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )
    units_out = np.load(data_path
            + f"units_out_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )
    scales_out = np.load(data_path
            + f"scales_out_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )    

    fig_path = data_path + f"plots_{simulation_mode}_dt_col_{dt_col}/"
    fig_name = \
               f"scalar_fields_avg_" \
               + f"t_{save_times_out_fr[0]}_" \
               + f"{save_times_out_fr[-1]}_Nfr_{len(save_times_out_fr)}_" \
               + f"Nfie_{len(field_names_out)}_" \
               + f"Ns_{no_seeds}_sg_{seed_SIP_gen_list[0]}_" \
               + f"ss_{seed_sim_list[0]}.png"
    if not os.path.exists(fig_path):
            os.makedirs(fig_path)    
    plot_scalar_field_frames_extend_avg(grid, fields_with_time,
                                        save_times_out_fr,
                                        field_names_out,
                                        units_out,
                                        scales_out,
                                        solute_type,
                                        simulation_mode, # for time in label
                                        fig_path=fig_path+fig_name,
                                        no_ticks=[6,6], 
                                        alpha = 1.0,
                                        TTFS = 12, LFS = 10, TKFS = 10,
                                        cbar_precision = 2,
                                        show_target_cells = show_target_cells,
                                        target_cell_list = target_cell_list,
                                        no_cells_x = no_cells_x,
                                        no_cells_z = no_cells_z)     
    plt.close("all")   
    
    print("plotted ensemble-averaged grid frames")

#%% PLOT COMPARISON OF AVG GRID FRAMES OF TWO SIMULATIONS

if act_plot_grid_frames_avg_compare:
    from analysis import plot_scalar_field_frames_extend_avg    
    
    if data_generated_with_gen_plot_data:
        data_folder2 = \
            f"{solute_type2}" \
            + f"/grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm2[0]}_{no_spcm2[1]}/"\
            + f"eval_data_avg_Ns_{no_seeds2}_" \
            + f"sg_{seed_SIP_gen2}_ss_{seed_sim2}/"
        data_path2 = simdata_path + data_folder2
        grid_path2 = simdata_path + data_folder2 + "grid_data/" \
                    + f"{seed_SIP_gen2}_{seed_sim2}/"
    else:        
        data_folder2 = \
            f"{solute_type2}" \
            + f"/grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm2[0]}_{no_spcm2[1]}/"\
            + f"{seed_SIP_gen2}/"
        data_path2 = simdata_path + data_folder2
        grid_path2 = simdata_path + data_folder2
    
    seed_SIP_gen_list2 = np.load(data_path2 + "seed_SIP_gen_list.npy" )
    seed_sim_list2 = np.load(data_path2 + "seed_sim_list.npy")    
    
    
    fields_with_time1 = np.load(data_path
            + f"fields_vs_time_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )
    
    fields_with_time2 = np.load(data_path2
            + f"fields_vs_time_avg_Ns_{no_seeds2}_"
            + f"sg_{seed_SIP_gen_list2[0]}_ss_{seed_sim_list2[0]}.npy"
            )
    
    fields_with_time = fields_with_time2 - fields_with_time1
    
    save_times_out_fr = np.load(data_path
            + f"save_times_out_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )
    field_names_out = np.load(data_path
            + f"field_names_out_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )
    units_out = np.load(data_path
            + f"units_out_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )
    scales_out = np.load(data_path
            + f"scales_out_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )    

    fig_path = data_path + f"plots_{simulation_mode}_dt_col_{dt_col}/"
    fig_name = \
               f"scalar_fields_avg_DIFF_" \
               + f"t_{save_times_out_fr[0]}_" \
               + f"{save_times_out_fr[-1]}_Nfr_{len(save_times_out_fr)}_" \
               + f"Nfie_{len(field_names_out)}_" \
               + f"Ns_{no_seeds}_sg_{seed_SIP_gen_list[0]}_" \
               + f"ss_{seed_sim_list[0]}.png"
    if not os.path.exists(fig_path):
            os.makedirs(fig_path)    
    plot_scalar_field_frames_extend_avg(grid, fields_with_time,
                                        save_times_out_fr,
                                        field_names_out,
                                        units_out,
                                        scales_out,
                                        solute_type,
                                        simulation_mode, # for time in label
                                        fig_path=fig_path+fig_name,
                                        no_ticks=[6,6], 
                                        alpha = 1.0,
                                        TTFS = 12, LFS = 10, TKFS = 10,
                                        cbar_precision = 2,
                                        show_target_cells = show_target_cells,
                                        target_cell_list = target_cell_list,
                                        no_cells_x = no_cells_x,
                                        no_cells_z = no_cells_z)     
    plt.close("all")   
    
    print("plotted compared ensemble-averaged grid frames of two simulations")

#%% PLOT AVG GRID FRAMES SHIFT IN X DIRECTION

if act_plot_grid_frames_avg_shift:
    from analysis import plot_scalar_field_frames_extend_avg_shift

    fields_with_time = np.load(data_path
            + f"fields_vs_time_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )
    save_times_out_fr = np.load(data_path
            + f"save_times_out_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )
    field_names_out = np.load(data_path
            + f"field_names_out_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )
    units_out = np.load(data_path
            + f"units_out_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )
    scales_out = np.load(data_path
            + f"scales_out_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )    

    fig_path = data_path + f"plots_{simulation_mode}_dt_col_{dt_col}/"
    fig_name = \
               f"scalar_fields_avg_shift_{shift_cells_x}_" \
               + f"t_{save_times_out_fr[0]}_" \
               + f"{save_times_out_fr[-1]}_Nfr_{len(save_times_out_fr)}_" \
               + f"Nfie_{len(field_names_out)}_" \
               + f"Ns_{no_seeds}_sg_{seed_SIP_gen_list[0]}_" \
               + f"ss_{seed_sim_list[0]}.png"
    if not os.path.exists(fig_path):
            os.makedirs(fig_path)    
    plot_scalar_field_frames_extend_avg_shift(grid, fields_with_time,
                                        save_times_out_fr,
                                        field_names_out,
                                        units_out,
                                        scales_out,
                                        solute_type,
                                        simulation_mode, # for time in label
                                        fig_path=fig_path+fig_name,
                                        no_ticks=[6,6], 
                                        alpha = 1.0,
                                        TTFS = 12, LFS = 10, TKFS = 10,
                                        cbar_precision = 2,
                                        show_target_cells = show_target_cells,
                                        target_cell_list = target_cell_list,
                                        no_cells_x = no_cells_x,
                                        no_cells_z = no_cells_z,
                                        shift_cells_x = shift_cells_x)     

    print("plotted ensemble-averaged grid frames shifted horiz. by",
          shift_cells_x, "cells")

#%% PLOT SPECTRA AVG 

if act_plot_spectra_avg_Arabas:
    from analysis import plot_size_spectra_R_Arabas
    
    f_R_p_list = np.load(data_path
            + f"f_R_p_list_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )    
    f_R_s_list = np.load(data_path
            + f"f_R_s_list_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )    
    bins_R_p_list = np.load(data_path
            + f"bins_R_p_list_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )    
    bins_R_s_list = np.load(data_path
            + f"bins_R_s_list_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )    
    save_times_out_spectra = np.load(data_path
            + f"save_times_out_spectra_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )    
    grid_r_l_list = np.load(data_path
            + f"grid_r_l_list_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )    
    R_min_list = np.load(data_path
            + f"R_min_list_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )    
    R_max_list = np.load(data_path
            + f"R_max_list_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )    
    
    # if not given manually above:
    if no_rows is None:
        no_rowcol = np.load(data_path
                + f"no_rows_no_cols_avg_Ns_{no_seeds}_"
                + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
                )    
        no_rows = no_rowcol[0]
        no_cols = no_rowcol[1]
        
    
    if no_bins_R_p is None:
        no_bins_p_s = np.load(data_path
                + f"no_bins_p_s_avg_Ns_{no_seeds}_"
                + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
                )        
        no_bins_R_p = no_bins_p_s[0]
        no_bins_R_s = no_bins_p_s[1]
    
    j_low = target_cell_list[1].min()
    j_high = target_cell_list[1].max()    
    t_low = save_times_out_spectra.min()
    t_high = save_times_out_spectra.max()    


    fig_path = data_path + f"plots_{simulation_mode}_dt_col_{dt_col}/"    
    if not os.path.exists(fig_path):
        os.makedirs(fig_path) 

    fig_name =\
        f"spectra_at_tg_cells_j_from_{j_low}_to_{j_high}_" \
        + f"Ntgcells_{no_tg_cells}_Nneigh_{no_cells_x}_{no_cells_z}_" \
        + f"Nseeds_{no_seeds}_sseed_{seed_sim_list[0]}_t_{t_low}_{t_high}.pdf"
    fig_path_tg_cells =\
        fig_path \
        + f"tg_cell_posi_j_from_{j_low}_to_{j_high}_" \
        + f"Ntgcells_{no_tg_cells}_Nneigh_{no_cells_x}_{no_cells_z}_" \
        + f"Nseeds_{no_seeds}_sseed_{seed_sim_list[0]}_t_{t_low}_{t_high}.pdf"
    fig_path_R_eff =\
        fig_path \
        + f"R_eff_{j_low}_to_{j_high}_" \
        + f"Ntgcells_{no_tg_cells}_Nneigh_{no_cells_x}_{no_cells_z}_" \
        + f"Nseeds_{no_seeds}_sseed_{seed_sim_list[0]}_t_{t_low}_{t_high}.pdf"
    
    plot_size_spectra_R_Arabas(f_R_p_list, f_R_s_list,
                               bins_R_p_list, bins_R_s_list,
                               grid_r_l_list,
                               R_min_list, R_max_list,
                               save_times_out_spectra,
                               solute_type,
                               grid,
                               target_cell_list,
                               no_cells_x, no_cells_z,
                               no_bins_R_p, no_bins_R_s,
                               no_rows, no_cols,
                               TTFS=12, LFS=10, TKFS=10, LW = 2.0,
                               fig_path = fig_path + fig_name,
                               show_target_cells = True,
                               fig_path_tg_cells = fig_path_tg_cells   ,
                               fig_path_R_eff = fig_path_R_eff
                               )        
    plt.close("all")    
    
    print("plotted spectra in target cells:")
    print(target_cell_list)
