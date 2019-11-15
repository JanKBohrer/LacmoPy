#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 19:36:12 2019

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


import matplotlib.ticker as mticker
#import numpy as np

from microphysics import compute_mass_from_radius_vec
#import constants as c

#import numba as 
from plotting import cm2inch
from plotting import generate_rcParams_dict
#from plotting import pgf_dict, pdf_dict
from plotting import pgf_dict2

#plt.rcParams.update(pdf_dict)

#from file_handling import load_grid_and_particles_full,\
#                          load_grid_scalar_fields\

from analysis import sample_masses, sample_radii
from analysis import avg_moments_over_boxes
from analysis import sample_masses_per_m_dry , sample_radii_per_m_dry
from analysis import plot_size_spectra_R_Arabas, generate_size_spectra_R_Arabas

from plotting_fcts_presiMA import plot_scalar_field_frames_extend_avg_PRESI

from plotting_fcts_MA import plot_size_spectra_R_Arabas_MA
from plotting_fcts_MA import plot_scalar_field_frames_std_MA

mpl.rcParams.update(plt.rcParamsDefault)
mpl.use("pgf")
#mpl.use("pdf")
#mpl.use("agg")
mpl.rcParams.update(pgf_dict2)
#mpl.rcParams.update(pdf_dict)

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
TTFS = 8
LFS = 8
TKFS = 6
#TTFS = 10
#LFS = 10
#TKFS = 8
#TTFS = 12
#LFS = 12
#TKFS = 10

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
    home_path = '/Users/bohrer/'
    simdata_path = "/Users/bohrer/sim_data_cloudMP/"
#    fig_path = home_path \
#               + 'OneDrive - bwedu/Uni/Masterthesis/latex/Report/Figures/'
elif (my_OS == "TROPOS_server"):
    simdata_path = "/vols/fs1/work/bohrer/sim_data_cloudMP/"

#%% CHOOSE OPERATIONS

#args_plot = [0,0,0,0,0,0,0,0,1,0]
args_plot = [0,1,0,0,0,0,0,0,0,0]
#args_plot = [0,0,0,1,0,0,0,0,0,0]

#args_plot = [0,1,0,0,0,0,0,0]
#args_plot = [0,0,1,0,0,0,0,0]
#args_plot = [0,0,0,1,0,0,0,0]
#args_plot = [0,0,0,0,1,0,0,0]
#args_plot = [0,0,0,0,0,0,0,1]

act_plot_grid_frames_init = args_plot[0]
act_plot_grid_frames_avg = args_plot[1]
act_plot_grid_frames_std = args_plot[2]
act_plot_grid_frames_abs_dev = args_plot[3]
act_plot_spectra_avg_Arabas = args_plot[4]
act_plot_moments_vs_z = args_plot[5]
act_plot_moments_diff_vs_z = args_plot[6]
act_plot_moments_norm_vs_t = args_plot[7]
act_plot_SIP_convergence = args_plot[8]
act_compute_CFL = args_plot[8]

# this one is not adapted for MA plots...
#act_plot_grid_frames_avg_shift = args_plot[2]
act_plot_grid_frames_avg_shift = False

#%% Simulations done
gseedl=(3711, 3811, 3811, 3411, 3311, 3811 ,4811 ,4811 ,3711, 3711, 3811)
sseedl=(6711 ,6811 ,6811 ,6411 ,6311 ,7811 ,6811 ,7811 ,6711, 6711, 8811)

ncl=(75, 75, 75, 75, 75, 75, 75, 75, 75, 150, 75)
solute_typel=["AS", "AS" ,"AS", "AS" ,"AS" ,"AS", "AS", "AS", "NaCl","AS","AS"]

DNC1 = np.array((60,60,60,30,120,60,60,60,60,60,60))
DNC2 = DNC1 * 2 // 3
no_spcm0l=(13, 26 ,52 ,26 ,26 ,26 ,26 ,26 ,26,26,26)
no_spcm1l=(19, 38 ,76 ,38 ,38 ,38 ,38 ,38 ,38,38,38)
no_col_per_advl=(2, 2, 2 ,2 ,2 ,2 ,2 ,2 ,2,2,10)

kernel_l = ["Long","Long","Long","Long","Long",
            "Hall","NoCol","Ecol05",
            "Long", "Long", "Long"]

#%% SET SIMULATION PARAS
SIM_N = 1  # default
#SIM_N = 2 # Nsip 128
#SIM_N = 3 # pristine
#SIM_N = 4 # polluted
#SIM_N = 7 # Ecol 0.5
#SIM_N = 10 # dt_col 0.1

shift_cells_x = 18

Ns=50
#t_grid = 0.
t_grid = 14400.
#t_start=0.0
t_start=7200.0
#t_end=7200.0
t_end=14400.0
dt=1. # advection TS
simulation_mode="with_collision"

#%% SET PLOTTING PARAMETERS

figsize_spectra = cm2inch(16.8,24)
figsize_tg_cells = cm2inch(6.6,7)
figsize_scalar_fields = cm2inch(12,8)

figsize_scalar_fields_init = cm2inch(7.4,7.4)
#figsize_scalar_fields = cm2inch(100,60)

figpath = home_path + "PresentationMA/Figures/"

### GRID FRAMES
if SIM_N == 9:
    show_target_cells = False
else:
    show_target_cells = True

### SPECTRA
# if set to "None" the values are loaded from stored files
#no_rows = 5
#no_cols = 3
    
if SIM_N == 7:
    no_rows = 5
    no_cols = 2
else:    
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

#if fields_type == 0:
#    idx_fields_plot = np.array((0,2,3,7))
#    fields_name_add = "Th_rv_rl_na_"
#
#if fields_type == 1:
#    idx_fields_plot = np.array((5,6,9,12))
#    fields_name_add = "rc_rr_nr_Reff_"    
#
#if fields_type == 3:
#    idx_fields_plot = np.array((7,5,6,12))
#    fields_name_add = "Naero_rc_rr_Reff_"  
#if fields_type == 4:
#    idx_fields_plot = np.array((7,8,9,5,6))
#    fields_name_add = "Naero_Nc_Nr_rc_rr_"  

fields_types_avg = [5]
#fields_types_avg = [0,1]

#if fields_type == 0:
#    idx_fields_plot = np.array((0,2,3,7))
#    fields_name_add = "Th_rv_rl_na_"
#
#if fields_type == 1:
#    idx_fields_plot = np.array((5,6,9,12))
#    fields_name_add = "rc_rr_nr_Reff_"  
#
#if fields_type == 2:
#    idx_fields_plot = np.array((2,7,5,6))
#    fields_name_add = "rv_na_rc_rr_"  
#if fields_type == 3:
#    idx_fields_plot = np.array((7,5,6,12))
#    fields_name_add = "Naero_rc_rr_Reff_"  

fields_types_std = [3]

plot_abs = True
plot_rel = True

# time indices for scalar field frames
# [ 7200  7800  8400  9000  9600 10200 10800]
idx_times_plot = np.array((0,2,3,6))

#%% SET PARAS FOR ABSOLUTE DIFFERENCE PLOTS

#%% DERIVED PARAS
#%% DERIVED GRID PARAMETERS

# needed for filename
no_cells = (ncl[SIM_N], ncl[SIM_N])
#no_cells = (75, 75)
#no_cells = (3, 3)

#shift_cells_x = 56

#%% DERIVED PARTICLE PARAMETERS

# solute material: NaCl OR ammonium sulfate
#solute_type = "NaCl"
solute_type = solute_typel[SIM_N]
kernel = kernel_l[SIM_N]
#kernel = "Ecol05"
seed_SIP_gen = gseedl[SIM_N]
seed_sim = sseedl[SIM_N]
DNC0 = [DNC1[SIM_N], DNC2[SIM_N]]
no_spcm = np.array([no_spcm0l[SIM_N], no_spcm1l[SIM_N]])

# no_super_particles_cell_mode = [N1,N2] is a list with
# N1 = no super part. per cell in mode 1 etc.
# with init method = SingleSIP, this is only the target value.
# the true number of particles per cell and mode will fluctuate around this
#no_spcm = np.array([13, 19])

no_seeds = Ns

dt_col = dt / no_col_per_advl[SIM_N]

figname_base =\
f"{solute_type}_{kernel}_dim_{no_cells[0]}_{no_cells[1]}"\
+ f"_SIP_{no_spcm[0]}_{no_spcm[1]}_Ns_{no_seeds}_DNC_{DNC0[0]}_{DNC0[1]}_dtcol_{int(dt_col*10)}"

#figname_moments = figname_moments0 + figname_base + ".pdf"
#figname_moments_rel_dev = figname_moments_rel_dev0 + figname_base + ".pdf"
#figname_conv = figname_conv0 + figname_base + ".pdf"
#figname_conv_err = figname_conv_err0 + figname_base + ".pdf"

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

if act_compute_CFL:
    # \Delta t \, \max(|u/\Delta x| + |v/\Delta y|)
    v_abs_max_err = np.abs(grid.velocity[0]) / grid.steps[0] \
            + np.abs(grid.velocity[1]) / grid.steps[1]
    
    v_abs = np.sqrt( grid.velocity[0]**2 + grid.velocity[1]**2 )
    CFL = np.amax(v_abs_max_err) * dt
    
    print("CFL", CFL)
    print("v_abs", np.amax(v_abs))

scale_x = 1E-3    
grid.steps *= scale_x
grid.ranges *= scale_x
grid.corners[0] *= scale_x
grid.corners[1] *= scale_x
grid.centers[0] *= scale_x
grid.centers[1] *= scale_x  

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
    
    for fields_type in fields_types_avg:
    
        if fields_type == 0:
            idx_fields_plot = np.array((0,2,3,7))
            fields_name_add = "Th_rv_rl_na_"
        
        if fields_type == 1:
            idx_fields_plot = np.array((5,6,9,12))
            fields_name_add = "rc_rr_nr_Reff_"    
        
        if fields_type == 3:
            idx_fields_plot = np.array((7,5,6,12))
            fields_name_add = "Naero_rc_rr_Reff_"    

        if fields_type == 4:
            idx_fields_plot = np.array((7,8,9,5,6))
            fields_name_add = "Naero_Nc_Nr_rc_rr_" 
            show_target_cells = False
        
        if fields_type == 5:
            idx_fields_plot = np.array((7,5,6))
            fields_name_add = "Naero_Nc_Nr_rc_rr_" 
            show_target_cells = False
        
#        if fields_type == 3:
#            idx_fields_plot = np.array((7,5,6,12))
#            fields_name_add = "Naero_rc_rr_Reff_"    
        
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
        
        fig_name = "fields_avg_" + fields_name_add + figname_base + ".pdf"
    
        if not os.path.exists(fig_path):
                os.makedirs(fig_path)    
        plot_scalar_field_frames_extend_avg_PRESI(grid, fields_with_time,
                                            save_times_out_fr,
                                            field_names_out,
                                            units_out,
                                            scales_out,
                                            solute_type,
                                            simulation_mode, # for time in label
                                            fig_path=fig_path+fig_name,
                                            figsize = figsize_scalar_fields,
                                            SIM_N = SIM_N,
                                            no_ticks=[6,6], 
                                            alpha = 1.0,
                                            TTFS = TTFS, LFS = LFS, TKFS = TKFS,
                                            cbar_precision = 2,
                                            show_target_cells = show_target_cells,
                                            target_cell_list = target_cell_list,
                                            no_cells_x = no_cells_x,
                                            no_cells_z = no_cells_z)     
    plt.close("all")   

#%% PLOT STD GRID FRAMES STD

if act_plot_grid_frames_std:
    
    for fields_type in fields_types_std:
    
        if fields_type == 0:
            idx_fields_plot = np.array((0,2,3,7))
            fields_name_add = "Th_rv_rl_na_"
        
        if fields_type == 1:
            idx_fields_plot = np.array((5,6,9,12))
            fields_name_add = "rc_rr_nr_Reff_"  
        
        if fields_type == 2:
            idx_fields_plot = np.array((2,7,5,6))
            fields_name_add = "rv_na_rc_rr_"  

        if fields_type == 3:
            idx_fields_plot = np.array((7,5,6,12))
            fields_name_add = "Naero_rc_rr_Reff_"  

        fields_with_time = np.load(data_path
                + f"fields_vs_time_avg_Ns_{no_seeds}_"
                + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
                )
        fields_with_time_std = np.load(data_path
                + f"fields_vs_time_std_Ns_{no_seeds}_"
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
        fields_with_time_std = fields_with_time_std[idx_times_plot][:,idx_fields_plot]
        save_times_out_fr = save_times_out_fr[idx_times_plot]-7200
        field_names_out = field_names_out[idx_fields_plot]
        units_out = units_out[idx_fields_plot]
        scales_out = scales_out[idx_fields_plot]
    
    #    fig_name = \
    #               f"scalar_fields_avg_" \
    #               + f"t_{save_times_out_fr[0]}_" \
    #               + f"{save_times_out_fr[-1]}_Nfr_{len(save_times_out_fr)}_" \
    #               + f"Nfie_{len(field_names_out)}_" \
    #               + f"Ns_{no_seeds}_sg_{seed_SIP_gen_list[0]}_" \
    #               + f"ss_{seed_sim_list[0]}_" \
    #               + fields_name_add + ".pdf"
        
        fig_name_abs = "fields_std_ABS_ERROR_" + fields_name_add + figname_base + ".pdf"
        fig_name_rel = "fields_std_REL_ERROR_" + fields_name_add + figname_base + ".pdf"
                   
        if not os.path.exists(figpath):
                os.makedirs(figpath)    
        plot_scalar_field_frames_std_MA(grid, fields_with_time,
                                        fields_with_time_std,
                                            save_times_out_fr,
                                            field_names_out,
                                            units_out,
                                            scales_out,
                                            solute_type,
                                            simulation_mode, # for time in label
                                            fig_path_abs=figpath+fig_name_abs,
                                            fig_path_rel=figpath+fig_name_rel,
                                            figsize = figsize_scalar_fields,
                                            SIM_N = SIM_N,
                                            plot_abs=plot_abs,
                                            plot_rel=plot_rel,                                            
                                            no_ticks=[6,6], 
                                            alpha = 1.0,
                                            TTFS = 10, LFS = 10, TKFS = 8,
                                            cbar_precision = 2,
                                            show_target_cells = show_target_cells,
                                            target_cell_list = target_cell_list,
                                            no_cells_x = no_cells_x,
                                            no_cells_z = no_cells_z)     
    plt.close("all")   

#%% PLOT ABSOLUTE DEVIATIONS BETWEEN TWO GRIDS

#compare_type_abs_dev = "dt_col"
compare_type_abs_dev = "Ncell"
#compare_type_abs_dev = "solute"
#compare_type_abs_dev = "Kernel"
#compare_type_abs_dev = "Nsip"

figsize_abs_dev = figsize_scalar_fields

if act_plot_grid_frames_abs_dev:
    
    from plotting_fcts_MA import plot_scalar_field_frames_abs_dev_MA
    
    # n_aero, r_c, r_r, R_eff
#    idx_fields_plot = (7, 5, 6, 12)
    
    ###
#    idx_fields_plot = np.array((7,5,6,12))
#    fields_name_add = "Naero_rc_rr_Reff_"
    
    ###
    idx_fields_plot = np.array((7,8,9,10))
    fields_name_add = "Naero_Nc_Nr_Ravg_"
    
    idx_times_plot = np.array((0,2,3,6))
    
    if compare_type_abs_dev == "dt_col":
        SIM_Ns = [1,10]
    elif compare_type_abs_dev == "Ncell":
        SIM_Ns = [1,9]
    elif compare_type_abs_dev == "solute":
        SIM_Ns = [1,8]
    elif compare_type_abs_dev == "Kernel":
        SIM_Ns = [1,5]
    elif compare_type_abs_dev == "Nsip":
        SIM_Ns = [1,2]
#        fname_abs_dev_add = ""
    
    SIM_N = SIM_Ns[0]
    
    no_cells = (ncl[SIM_N], ncl[SIM_N])
    solute_type = solute_typel[SIM_N]
    kernel = kernel_l[SIM_N]
    seed_SIP_gen = gseedl[SIM_N]
    seed_sim = sseedl[SIM_N]
    DNC0 = [DNC1[SIM_N], DNC2[SIM_N]]
    no_spcm = np.array([no_spcm0l[SIM_N], no_spcm1l[SIM_N]])
    no_seeds = Ns
    dt_col = dt / no_col_per_advl[SIM_N]
    
#    figname_base =\
#    f"{solute_type}_{kernel}_dim_{no_cells[0]}_{no_cells[1]}"\
#    + f"_SIP_{no_spcm[0]}_{no_spcm[1]}_Ns_{no_seeds}_DNC_{DNC0[0]}_{DNC0[1]}_dtcol_{int(dt_col*10)}"
    
    data_folder = \
        f"{solute_type}" \
        + f"/grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm[0]}_{no_spcm[1]}/"\
        + f"eval_data_avg_Ns_{no_seeds}_" \
        + f"sg_{seed_SIP_gen}_ss_{seed_sim}/"
    
    grid_path = simdata_path + data_folder + "grid_data/" \
                + f"{seed_SIP_gen}_{seed_sim}/"
    
    grid = load_grid_from_files(grid_path + f"grid_basics_{int(t_grid)}.txt",
                                grid_path + f"arr_file1_{int(t_grid)}.npy",
                                grid_path + f"arr_file2_{int(t_grid)}.npy")
    scale_x = 1E-3    
    grid.steps *= scale_x
    grid.ranges *= scale_x
    grid.corners[0] *= scale_x
    grid.corners[1] *= scale_x
    grid.centers[0] *= scale_x
    grid.centers[1] *= scale_x  
    
    data_path = simdata_path + data_folder
    seed_SIP_gen_list = np.load(data_path + "seed_SIP_gen_list.npy" )
    seed_sim_list = np.load(data_path + "seed_sim_list.npy")
    
    fields_with_time1 = np.load(data_path
            + f"fields_vs_time_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )
    fields_with_time_std1 = np.load(data_path
            + f"fields_vs_time_std_Ns_{no_seeds}_"
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
    
    ####################
    SIM_N = SIM_Ns[1]
    
    no_cells = (ncl[SIM_N], ncl[SIM_N])
    solute_type = solute_typel[SIM_N]
    kernel = kernel_l[SIM_N]
    seed_SIP_gen = gseedl[SIM_N]
    seed_sim = sseedl[SIM_N]
    DNC0 = [DNC1[SIM_N], DNC2[SIM_N]]
    no_spcm = np.array([no_spcm0l[SIM_N], no_spcm1l[SIM_N]])
    no_seeds = Ns
    dt_col = dt / no_col_per_advl[SIM_N]
    
    figname_base =\
    f"{solute_type}_{kernel}_dim_{no_cells[0]}_{no_cells[1]}"\
    + f"_SIP_{no_spcm[0]}_{no_spcm[1]}_Ns_{no_seeds}_DNC_{DNC0[0]}_{DNC0[1]}_dtcol_{int(dt_col*10)}"
    
    data_folder = \
        f"{solute_type}" \
        + f"/grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm[0]}_{no_spcm[1]}/"\
        + f"eval_data_avg_Ns_{no_seeds}_" \
        + f"sg_{seed_SIP_gen}_ss_{seed_sim}/"
    
    data_path = simdata_path + data_folder
    seed_SIP_gen_list = np.load(data_path + "seed_SIP_gen_list.npy" )
    seed_sim_list = np.load(data_path + "seed_sim_list.npy")
    
    fields_with_time2 = np.load(data_path
            + f"fields_vs_time_avg_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )
    fields_with_time_std2 = np.load(data_path
            + f"fields_vs_time_std_Ns_{no_seeds}_"
            + f"sg_{seed_SIP_gen_list[0]}_ss_{seed_sim_list[0]}.npy"
            )   

    
    ####################
    
#    print(fields_with_time2.shape)
    
    fields_with_time1 = fields_with_time1[idx_times_plot][:,idx_fields_plot]
    fields_with_time_std1 = fields_with_time_std1[idx_times_plot][:,idx_fields_plot]
    fields_with_time2 = fields_with_time2[idx_times_plot][:,idx_fields_plot]
    fields_with_time_std2 = fields_with_time_std2[idx_times_plot][:,idx_fields_plot]
    
    if compare_type_abs_dev == "Ncell":
        fields_with_time2 = (  fields_with_time2[:,:,::2,::2] 
                             + fields_with_time2[:,:,1::2,::2] 
                             + fields_with_time2[:,:,::2,1::2] 
                             + fields_with_time2[:,:,1::2,1::2] ) / 4.
        fields_with_time_std2 = (  fields_with_time_std2[:,:,::2,::2] 
                                 + fields_with_time_std2[:,:,1::2,::2] 
                                 + fields_with_time_std2[:,:,::2,1::2] 
                                 + fields_with_time_std2[:,:,1::2,1::2] ) / 4.
#    print(fields_with_time_x.shape)
    
    print(field_names_out)
    
    save_times_out_fr = save_times_out_fr[idx_times_plot]-7200
    field_names_out = field_names_out[idx_fields_plot]
    units_out = units_out[idx_fields_plot]
    scales_out = scales_out[idx_fields_plot]

#    fig_path = data_path + f"plots_{simulation_mode}_dt_col_{dt_col}/"
#    fig_path = figpath
#    fig_name = \
#               f"scalar_fields_avg_" \
#               + f"t_{save_times_out_fr[0]}_" \
#               + f"{save_times_out_fr[-1]}_Nfr_{len(save_times_out_fr)}_" \
#               + f"Nfie_{len(field_names_out)}_" \
#               + f"Ns_{no_seeds}_sg_{seed_SIP_gen_list[0]}_" \
#               + f"ss_{seed_sim_list[0]}_" \
#               + fields_name_add + ".pdf"
    
    fig_name = "fields_abs_dev_" + compare_type_abs_dev + "_" \
               + fields_name_add + figname_base + ".pdf"
    fig_name_abs_err = "fields_abs_dev_ABS_ERR_" + compare_type_abs_dev + "_" \
               + fields_name_add + figname_base + ".pdf"
               
    if not os.path.exists(figpath):
            os.makedirs(figpath)    
            
    plot_scalar_field_frames_abs_dev_MA(grid,
                                        fields_with_time1,
                                        fields_with_time_std1,
                                        fields_with_time2,
                                        fields_with_time_std2,
                                        save_times_out_fr,
                                        field_names_out,
                                        units_out,
                                        scales_out,
                                        solute_type,
                                        simulation_mode, # for time in label
                                        compare_type=compare_type_abs_dev,
                                        fig_path=figpath + fig_name,
                                        fig_path_abs_err=figpath + fig_name_abs_err,
                                        figsize=figsize_abs_dev,
                                        no_ticks=[6,6],
                                        alpha = 1.0,
                                        TTFS = 10, LFS = 10, TKFS = 8,
                                        cbar_precision = 2,
                                        show_target_cells = False,
                                        target_cell_list = None,
                                        no_cells_x = 0,
                                        no_cells_z = 0
                                        )   
    ### CAN BE ACTIVATED!!!
#    if compare_type_abs_dev == "Ncell":       
#        fig_name2 = "fields_coarse_grain_" + compare_type_abs_dev + "_" \
#           + fields_name_add + figname_base + ".pdf"
#        plot_scalar_field_frames_extend_avg_MA(grid, fields_with_time2,
#                                            save_times_out_fr,
#                                            field_names_out,
#                                            units_out,
#                                            scales_out,
#                                            solute_type,
#                                            simulation_mode, # for time in label
#                                            fig_path=figpath+fig_name2,
#                                            figsize = figsize_scalar_fields,
#                                            no_ticks=[6,6], 
#                                            alpha = 1.0,
#                                            TTFS = 10, LFS = 10, TKFS = 8,
#                                            cbar_precision = 2,
#                                            show_target_cells = show_target_cells,
#                                            target_cell_list = target_cell_list,
#                                            no_cells_x = no_cells_x,
#                                            no_cells_z = no_cells_z)     
#      
#        fig_name3 = "fields_default_compare_" + compare_type_abs_dev + "_" \
#           + fields_name_add + figname_base + ".pdf"
#        plot_scalar_field_frames_extend_avg_MA(grid, fields_with_time1,
#                                            save_times_out_fr,
#                                            field_names_out,
#                                            units_out,
#                                            scales_out,
#                                            solute_type,
#                                            simulation_mode, # for time in label
#                                            fig_path=figpath+fig_name3,
#                                            figsize = figsize_scalar_fields,
#                                            no_ticks=[6,6], 
#                                            alpha = 1.0,
#                                            TTFS = 10, LFS = 10, TKFS = 8,
#                                            cbar_precision = 2,
#                                            show_target_cells = show_target_cells,
#                                            target_cell_list = target_cell_list,
#                                            no_cells_x = no_cells_x,
#                                            no_cells_z = no_cells_z)     
    plt.close("all")   


