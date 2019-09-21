#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 15:43:10 2019

@author: jdesk
"""

#%% MODULE IMPORTS
import os
import numpy as np
# from datetime import datetime
# import timeit

import constants as c
from microphysics import compute_R_p_w_s_rho_p_AS
from microphysics import compute_R_p_w_s_rho_p_NaCl
from microphysics import compute_radius_from_mass_vec
from file_handling import load_grid_and_particles_full,\
                          load_grid_scalar_fields\


import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

#mpl.use("pdf")

#mpl.use("pgf")

import matplotlib.ticker as mticker
#import numpy as np

from microphysics import compute_mass_from_radius_vec
#import constants as c

#import numba as 
from plotting import cm2inch
from plotting import generate_rcParams_dict
from plotting import pgf_dict, pdf_dict

#plt.rcParams.update(pdf_dict)

#from file_handling import load_grid_and_particles_full,\
#                          load_grid_scalar_fields\

from analysis import sample_masses, sample_radii
from analysis import avg_moments_over_boxes
from analysis import sample_masses_per_m_dry , sample_radii_per_m_dry
from analysis import plot_size_spectra_R_Arabas, generate_size_spectra_R_Arabas
from plotting_fcts_MA import plot_scalar_field_frames_extend_avg_MA
from plotting_fcts_MA import plot_size_spectra_R_Arabas_MA

mpl.rcParams.update(plt.rcParamsDefault)
mpl.rcParams.update(pgf_dict)

#%%
def gen_data_paths(solute_type_var, kernel_var, seed_SIP_gen_var, seed_sim_var,
                   DNC0_var, no_spcm_var, no_seeds_var, dt_col_var):
    grid_paths = []
    data_paths = []
    
    no_variations = len(solute_type_var)
    for var_n in range(no_variations):
        solute_type = solute_type_var[var_n]
        no_cells = no_cells_var[var_n]
        no_spcm = no_spcm_var[var_n]
        no_seeds = no_seeds_var[var_n]
        seed_SIP_gen = seed_SIP_gen_var[var_n]
        seed_sim = seed_sim_var[var_n]
        
        data_folder = \
            f"{solute_type}" \
            + f"/grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm[0]}_{no_spcm[1]}/"\
            + f"eval_data_avg_Ns_{no_seeds}_" \
            + f"sg_{seed_SIP_gen}_ss_{seed_sim}/"    
        data_path = simdata_path + data_folder
        
        grid_path = simdata_path + data_folder + "grid_data/" \
                    + f"{seed_SIP_gen}_{seed_sim}/"
    
        data_paths.append(data_path)
        grid_paths.append(grid_path)

    return grid_paths, data_paths

#%% SET DEFAULT PLOT PARAMETERS
# (can be changed lateron for specific elements directly)
# TITLE, LABEL (and legend), TICKLABEL FONTSIZES
TTFS = 10
LFS = 10
TKFS = 8

#TTFS = 16
#LFS = 12
#TKFS = 12

# LINEWIDTH, MARKERSIZE
LW = 1.2
MS = 2

# raster resolution for e.g. .png
DPI = 600

mpl.rcParams.update(generate_rcParams_dict(LW, MS, TTFS, LFS, TKFS, DPI))

#%% STORAGE DIRECTORIES
my_OS = "Linux_desk"
#my_OS = "Mac"
#my_OS = "TROPOS_server"

if(my_OS == "Linux_desk"):
    home_path = '/home/jdesk/'
    simdata_path = "/mnt/D/sim_data_cloudMP/"
#    fig_path = home_path + 'Onedrive/Uni/Masterthesis/latex/Report/Figures/'
elif (my_OS == "Mac"):
    simdata_path = "/Users/bohrer/sim_data_cloudMP/"
#    fig_path = home_path \
#               + 'OneDrive - bwedu/Uni/Masterthesis/latex/Report/Figures/'
elif (my_OS == "TROPOS_server"):
    simdata_path = "/vols/fs1/work/bohrer/sim_data_cloudMP/"

#%% CHOOSE OPERATIONS

args_plot = [0,0,0,1]

act_plot_grid_frames_avg = args_plot[0]
act_plot_grid_frames_avg_shift = args_plot[1]
act_plot_spectra_avg_Arabas = args_plot[2]
act_plot_moments_vs_z = args_plot[3]

#%% GRID PARAMETERS

# needed for filename
no_cells = (75, 75)
#no_cells = (3, 3)

shift_cells_x = 18
#shift_cells_x = 56

#%% PARTICLE PARAMETERS

# solute material: NaCl OR ammonium sulfate
#solute_type = "NaCl"
solute_type = "AS"
kernel = "Long"
#kernel = "Ecol05"
seed_SIP_gen = 3711
seed_sim = 4711
DNC0 = [60,40]
#DNC0 = [30,20]
#DNC0 = [120,80]
no_spcm = np.array([20, 30])
#no_spcm = np.array([26, 38])
#no_spcm = np.array([52, 76])

# no_super_particles_cell_mode = [N1,N2] is a list with
# N1 = no super part. per cell in mode 1 etc.
# with init method = SingleSIP, this is only the target value.
# the true number of particles per cell and mode will fluctuate around this
#no_spcm = np.array([13, 19])

#no_seeds = 2
#no_seeds = 50
no_seeds = 4

dt_col = 0.5

figname_base =\
f"{solute_type}_{kernel}_dim_{no_cells[0]}_{no_cells[1]}"\
+ f"_SIP_{no_spcm[0]}_{no_spcm[1]}_Ns_{no_seeds}_DNC_{DNC0[0]}_{DNC0[1]}_dtcol_{int(dt_col*10)}"

#%% SIM PARAMETERS

#simulation_mode = "spin_up"
#simulation_mode = "wo_collision"
simulation_mode = "with_collision"

# for file names
#dt_col = 0.5
#dt_col = 1.0

# grid load time
# path = simdata_path + folder_load_base
#t_grid = 0
#t_grid = 7200
#t_grid = 10800
t_grid = 14400

#t_start = 0
t_start = 7200

#t_end = 60
#t_end = 3600
#t_end = 7200
#t_end = 10800
t_end = 14400

#%% PLOTTING PARAMETERS

figsize_spectra = cm2inch(16.8,22)
figsize_tg_cells = cm2inch(6.6,7)
figsize_scalar_fields = cm2inch(16.8,20)
#figsize_scalar_fields = cm2inch(100,60)

figpath = home_path + "Masterthesis/Figures/06TestCase/"

#figname_add = ""

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

# indices for the possible fields
#0 \Theta
#1 S
#2 r_v
#3 r_l
#4 r_\mathrm{aero}
#5 r_c
#6 r_r
#7 n_\mathrm{aero}
#8 n_c
#9 n_r
#10 R_\mathrm{avg}
#11 R_{2/1}
#12 R_\mathrm{eff}

fields_type = 1

if fields_type == 0:
    idx_fields_plot = np.array((0,2,3,7))
    fields_name_add = "Th_rv_rl_na"

if fields_type == 1:
    idx_fields_plot = np.array((5,6,9,12))
    fields_name_add = "rc_rr_nr_Reff"

# time indices for scalar field frames
# [ 7200  7800  8400  9000  9600 10200 10800]
idx_times_plot = np.array((0,2,3,6))

#%% SET PARAS FOR PLOTTING MOMENTS
no_variations = 3

no_cells_var = [[75,75],[75,75],[75,75]] 

solute_type_var = ["AS","AS","AS"]
kernel_var = ["Long","Long","Long"]
seed_SIP_gen_var = [3711,3711,3711]
seed_sim_var = [4711,4711,4711]
DNC0_var = [[60,40],[60,40],[60,40]]
no_spcm_var = [[20, 30],[20, 30],[20, 30]]
no_seeds_var = [4,4,4]
dt_col_var = [[0.5,0.5,0.5]]

grid_paths, data_paths = gen_data_paths(solute_type_var, kernel_var,
                                        seed_SIP_gen_var, seed_sim_var,
                                        DNC0_var, no_spcm_var,
                                        no_seeds_var, dt_col_var)


#%% LOAD GRID AND SET PATHS

data_folder = \
    f"{solute_type}" \
    + f"/grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm[0]}_{no_spcm[1]}/"\
    + f"eval_data_avg_Ns_{no_seeds}_" \
    + f"sg_{seed_SIP_gen}_ss_{seed_sim}/"

data_path = simdata_path + data_folder

grid_path = simdata_path + data_folder + "grid_data/" \
            + f"{seed_SIP_gen}_{seed_sim}/"

from file_handling import load_grid_from_files

grid = load_grid_from_files(grid_path + f"grid_basics_{int(t_grid)}.txt",
                            grid_path + f"arr_file1_{int(t_grid)}.npy",
                            grid_path + f"arr_file2_{int(t_grid)}.npy")

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


#%% PLOT AVG GRID FRAMES

if act_plot_grid_frames_avg:

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

    fields_with_time = fields_with_time[idx_times_plot][:,idx_fields_plot]
    save_times_out_fr = save_times_out_fr[idx_times_plot]-7200
    field_names_out = field_names_out[idx_fields_plot]
    units_out = units_out[idx_fields_plot]
    scales_out = scales_out[idx_fields_plot]

#    fig_path = data_path + f"plots_{simulation_mode}_dt_col_{dt_col}/"
    fig_path = figpath
#    fig_name = \
#               f"scalar_fields_avg_" \
#               + f"t_{save_times_out_fr[0]}_" \
#               + f"{save_times_out_fr[-1]}_Nfr_{len(save_times_out_fr)}_" \
#               + f"Nfie_{len(field_names_out)}_" \
#               + f"Ns_{no_seeds}_sg_{seed_SIP_gen_list[0]}_" \
#               + f"ss_{seed_sim_list[0]}_" \
#               + fields_name_add + ".pdf"
    
    fig_name = "fields_avg_" + figname_base + ".pdf"
               
    if not os.path.exists(fig_path):
            os.makedirs(fig_path)    
    plot_scalar_field_frames_extend_avg_MA(grid, fields_with_time,
                                        save_times_out_fr,
                                        field_names_out,
                                        units_out,
                                        scales_out,
                                        solute_type,
                                        simulation_mode, # for time in label
                                        fig_path=fig_path+fig_name,
                                        figsize = figsize_scalar_fields,
                                        no_ticks=[6,6], 
                                        alpha = 1.0,
                                        TTFS = 10, LFS = 10, TKFS = 8,
                                        cbar_precision = 2,
                                        show_target_cells = show_target_cells,
                                        target_cell_list = target_cell_list,
                                        no_cells_x = no_cells_x,
                                        no_cells_z = no_cells_z)     
    plt.close("all")   

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
#                                        show_target_cells = False,
                                        show_target_cells = show_target_cells,
                                        target_cell_list = target_cell_list,
                                        no_cells_x = no_cells_x,
                                        no_cells_z = no_cells_z,
                                        shift_cells_x = shift_cells_x)     
#    plt.close("all")   

#%% PLOT SPECTRA AVG 

if act_plot_spectra_avg_Arabas:
    grid = load_grid_from_files(grid_path + f"grid_basics_{int(t_grid)}.txt",
                            grid_path + f"arr_file1_{int(t_grid)}.npy",
                            grid_path + f"arr_file2_{int(t_grid)}.npy")
#    from analysis import plot_size_spectra_R_Arabas
    
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
        
    print(no_rows, no_cols)    
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


#    fig_path = data_path + f"plots_{simulation_mode}_dt_col_{dt_col}/"    
#    fig_path = data_path + f"plots_{simulation_mode}_dt_col_{dt_col}/"  
    fig_path = figpath
    if not os.path.exists(fig_path):
        os.makedirs(fig_path) 

    fig_path_spectra =\
        fig_path \
        + f"spectra_Nc_{no_tg_cells}_" + figname_base + ".pdf"
#        f"spectra_at_tg_cells_j_from_{j_low}_to_{j_high}_" \
#        + f"Ntgcells_{no_tg_cells}_Nneigh_{no_cells_x}_{no_cells_z}_" \
#        + f"Nseeds_{no_seeds}_sseed_{seed_sim_list[0]}_t_{t_low}_{t_high}.pdf"
    fig_path_tg_cells =\
        fig_path \
        + f"target_c_pos_Nc_{no_tg_cells}_" + figname_base + ".pdf"
#        + f"tg_cell_posi_j_from_{j_low}_to_{j_high}_" \
#        + f"Ntgcells_{no_tg_cells}_Nneigh_{no_cells_x}_{no_cells_z}_" \
#        + f"Nseeds_{no_seeds}_sseed_{seed_sim_list[0]}_t_{t_low}_{t_high}.pdf"
    fig_path_R_eff =\
        fig_path \
        + f"R_eff_Nc_{no_tg_cells}_" + figname_base + ".pdf"
#        + f"R_eff_{j_low}_to_{j_high}_" \
#        + f"Ntgcells_{no_tg_cells}_Nneigh_{no_cells_x}_{no_cells_z}_" \
#        + f"Nseeds_{no_seeds}_sseed_{seed_sim_list[0]}_t_{t_low}_{t_high}.pdf"
    
    
    plot_size_spectra_R_Arabas_MA(
            f_R_p_list, f_R_s_list,
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
            TTFS=10, LFS=10, TKFS=8, LW = 2.0, MS = 0.4,
            figsize_spectra = figsize_spectra,
            figsize_trace_traj = figsize_tg_cells,
            fig_path = fig_path_spectra,
            show_target_cells = True,
            fig_path_tg_cells = fig_path_tg_cells   ,
            fig_path_R_eff = fig_path_R_eff,
            trajectory = None,
            show_textbox=False)    
#    plot_size_spectra_R_Arabas_MA(f_R_p_list, f_R_s_list,
#                               bins_R_p_list, bins_R_s_list,
#                               grid_r_l_list,
#                               R_min_list, R_max_list,
#                               save_times_out_spectra,
#                               solute_type,
#                               grid,
#                               target_cell_list,
#                               no_cells_x, no_cells_z,
#                               no_bins_R_p, no_bins_R_s,
#                               no_rows, no_cols,
#                               TTFS=12, LFS=10, TKFS=10, LW = 2.0,
#                               fig_path = fig_path + fig_name,
#                               show_target_cells = True,
#                               fig_path_tg_cells = fig_path_tg_cells   ,
#                               fig_path_R_eff = fig_path_R_eff
#                               )        
    plt.close("all")    
    

#%% PLOT MOMENTS VS Z

if act_plot_moments_vs_z:
    print(grid_paths)
    
    no_boxes_z = 25
    no_cells_per_box_x = 3
    target_cells_x = np.array((16, 58, 66))    
    
    no_moments = 4
    idx_t = [3]
    
    no_target_cells_x = len(target_cells_x)
    no_target_cells_z = no_boxes_z

    no_rows = no_moments
    no_cols = no_target_cells_x
    
    figsize_moments = cm2inch(15,22)
    figname_moments = "moments_vs_z.pdf"
    
    fig, axes = plt.subplots(no_rows, no_cols, figsize=figsize_moments,
                             sharex=True, sharey="row")
    
    for var_n in range(no_variations):
        grid_path_ = grid_paths[var_n]
        data_path_ = data_paths[var_n]
    
        grid = load_grid_from_files(grid_path_ + f"grid_basics_{int(t_grid)}.txt",
                                grid_path_ + f"arr_file1_{int(t_grid)}.npy",
                                grid_path_ + f"arr_file2_{int(t_grid)}.npy")    
    
    #    np.save(simdata_path + output_folder
    #            + f"moments_vs_time_avg_Ns_{no_seeds}_"
    #            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
    #            moments_vs_time_avg)
    #    np.save(simdata_path + output_folder
    #            + f"moments_vs_time_std_Ns_{no_seeds}_"
    #            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
    #            moments_vs_time_std)
    #    np.save(simdata_path + output_folder
    #            + f"save_times_out_avg_Ns_{no_seeds}_"
    #            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}",
    #            save_times_out)
    
        moments_vs_time_all_seeds = np.load(data_path_ + "moments/"
                       + f"moments_vs_time_all_seeds_Ns_{no_seeds}_"
                       + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy")
    #    moments_vs_time_avg = np.load(data_path + "moments/"
    #                   + f"moments_vs_time_avg_Ns_{no_seeds}_"
    #                   + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy")
    #    moments_vs_time_std = np.load(data_path + "moments/"
    #                   + f"moments_vs_time_std_Ns_{no_seeds}_"
    #                   + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy")
        save_times_out = np.load(data_path_ + "moments/"
                       + f"save_times_out_avg_Ns_{no_seeds}_"
                       + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy")
        
        Nz = grid.no_cells[1]
        no_cells_per_box_z = Nz // no_boxes_z
        
        start_cell_z = no_cells_per_box_z // 2
        
        target_cells_z = np.arange( start_cell_z, Nz+1, no_cells_per_box_z )
        
#        print(target_cells_z)        
    

        
        no_seeds = moments_vs_time_all_seeds.shape[0]
#        no_moments = moments_vs_time_all_seeds.shape[2]
        
        moments_vs_z = np.zeros( (no_target_cells_x,
                                  no_target_cells_z,
                                  no_moments
                                  ),
                                dtype = np.float64)
        
        times = save_times_out[ np.array(idx_t) ]
        no_times_eval = len(times)
        
    #    from numba import njit
    #    @njit()    
                                
                
            
        moments_at_boxes = avg_moments_over_boxes(
                moments_vs_time_all_seeds, no_seeds, idx_t, no_moments,
                target_cells_x, target_cells_z,
                no_cells_per_box_x, no_cells_per_box_z )
        
        moments_at_boxes_avg = np.average(moments_at_boxes, axis=0)
        moments_at_boxes_std = np.std(moments_at_boxes, axis=0, ddof=1)
        


    
        for row_n in range(no_moments):
            for col_n in range(no_target_cells_x):
                ax = axes[row_n, col_n]
                z = ((target_cells_z + 0.5) * grid.steps[1]) / 1500
                x = ((target_cells_x[col_n] + 0.5) * grid.steps[0]) / 1500
                ax.errorbar( z, moments_at_boxes_avg[0,row_n,col_n],
                            yerr=moments_at_boxes_std[0,row_n,col_n],
                            fmt = "x-")
    #            ax.plot( z, moments_at_boxes_std[0,row_n,col_n] )
                if var_n == 0:
                    if row_n == no_rows - 1:
                        ax.set_xlabel("$z$ (km)")
                    if row_n == 0:
                        ax.set_title(f"$x={x:.2}$ km")
                    if col_n == 0:
                        ax.set_ylabel(f"$\\lambda_{row_n}$")
                    
    fig.savefig(figpath + figname_moments,
                bbox_inches = 'tight',
                pad_inches = 0.04,
                dpi=600
                )          
    
    