#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TROPOS LAGRANGIAN CLOUD MODEL
Super-Droplet method in two-dimensional kinematic framework
(Test Case 1 ICMW 2012, Muhlbauer et al. 2013)
Author: Jan Bohrer (bohrer@tropos.de)
Further contact: Oswald Knoth (knoth@tropos.de)

PLOTTING FUNCTIONS FOR THE GMD PUBLICATION

basic units:
particle mass, water mass, solute mass in femto gram = 10^-18 kg
particle radius in micro meter ("mu")
all other quantities in SI units
"""

#%% MODULE IMPORTS
import os
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from file_handling import load_grid_from_files
from plotting import cm2inch, generate_rcParams_dict, pgf_dict
from plotting import plot_scalar_field_frames_avg
from plotting import plot_size_spectra_vs_R

mpl.rcParams.update(plt.rcParamsDefault)
mpl.use('pgf')
mpl.rcParams.update(pgf_dict)

#%% SET DEFAULT PLOT PARAMETERS

# fontsizes for title, label (and legend), ticklabels
TTFS = 10
LFS = 10
TKFS = 8
# linewidth, markersize
LW = 1.2
MS = 2
# raster resolution for e.g. .png
DPI = 600

mpl.rcParams.update(generate_rcParams_dict(LW, MS, TTFS, LFS, TKFS, DPI))

#%% DATA AND STORAGE DIRECTORIES

# parent directory of the simulation data
simdata_path = '/Users/bohrer/sim_data_cloudMP/'
home_path = '/Users/bohrer/'
# save figures in this directory
fig_dir = home_path \
          + '/OneDrive - bwedu/Paper_LCM_2019/TeX/figures4/'

#%% PARAMETERS OF CONDUCTED SIMULATIONS (as lists/tuples)

gen_seed_list = (6001, 2001) # (simulation series 0, sim. series 1, ...)
sim_seed_list = (6001, 2001) # (simulation series 0, sim. series 1, ...)

no_cells_x_list = (75, 75)
no_cells_z_list = (75, 75)
solute_type_list = ['AS', 'AS']

DNC1_list = np.array((60, 60)) # droplet number conc. in mode 1
DNC2_list = DNC1_list * 2 // 3 # droplet number conc. in mode 1
no_spcm_0_list = (26, 26) # number of SIPs in mode 1
no_spcm_1_list = (38, 38) # number of SIPS in mode 2
no_col_per_adv_list = (2, 2)

kernel_list = ['Hall', 'Long']

#%% SET SIMULATION PARAS
# choose simulation series (index of the lists/tuples above)
SIM_N = 0  # Hall kernel, AS

no_sims = 50
t_grid = 0
t_start = 7200
t_end = 10800
dt = 1. # advection time step
# 'with_collisions', 'wo_collisions' or 'spin_up'
simulation_mode = 'with_collisions'

#%% SET PLOTTING PARAMETERS

figsize_scalar_fields = cm2inch(13.7,22.)
figsize_spectra = cm2inch(13.3,20.)
figsize_tg_cells = cm2inch(6.6,7)

### GRID FRAMES
show_target_cells = True

### SPECTRA
# if set to "None" the values are loaded from stored files
no_rows = None
no_cols = None

# if set to "None" the values are loaded from stored files
no_bins_R_p = None
no_bins_R_s = None

plot_abs = True
plot_rel = True

# time indices for scalar field frames, the default time list is
# [ 7200  7800  8400  9000  9600 10200 10800] seconds
idx_times_plot = np.array((0,2,3))

#%% DERIVED PARAMETERS

args_plot = [1,1]
act_plot_grid_frames_avg = args_plot[0]
act_plot_spectra_avg = args_plot[1]

# needed for filename
no_cells = (no_cells_x_list[SIM_N], no_cells_z_list[SIM_N])

# solute material: NaCl OR ammonium sulfate
solute_type = solute_type_list[SIM_N]
kernel = kernel_list[SIM_N]
seed_SIP_gen = gen_seed_list[SIM_N]
seed_sim = sim_seed_list[SIM_N]
DNC0 = [DNC1_list[SIM_N], DNC2_list[SIM_N]]
no_spcm = np.array([no_spcm_0_list[SIM_N], no_spcm_1_list[SIM_N]])
no_seeds = no_sims
dt_col = dt / no_col_per_adv_list[SIM_N]

figname_base =\
    f"{solute_type}_{kernel}_dim_{no_cells[0]}_{no_cells[1]}_"\
    + f"SIP_{no_spcm[0]}_{no_spcm[1]}_Ns_{no_seeds}_"\
    + f"DNC_{DNC0[0]}_{DNC0[1]}_dtcol_{int(dt_col*10)}"

#%% LOAD GRID AND SET PATHS

data_folder = \
    f"{solute_type}" \
    + f"/grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm[0]}_{no_spcm[1]}/"\
    + f"eval_data_avg_Ns_{no_seeds}_" \
    + f"sg_{seed_SIP_gen}_ss_{seed_sim}/"

data_path = simdata_path + data_folder

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

#%% PLOT GRID FRAMES AVERAGED

fields_types_avg = [0,1]
if act_plot_grid_frames_avg:
    
    for fields_type in fields_types_avg:
    
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
        if fields_type == 0:
            idx_fields_plot = np.array((7,8,9))
            fields_name_add = "Naero_Nc_Nr_" 
            show_target_cells = True
        
        if fields_type == 1:
            idx_fields_plot = np.array((12,5,6))
            fields_name_add = "Reff_rc_rr_" 
            show_target_cells = True
        
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
        save_times_out_fr = save_times_out_fr[idx_times_plot] - 7200
        field_names_out = field_names_out[idx_fields_plot]
        units_out = units_out[idx_fields_plot]
        scales_out = scales_out[idx_fields_plot]
    
        fig_name = "fields_avg_" + fields_name_add + figname_base + ".pdf"
    
        if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)    
        plot_scalar_field_frames_avg(
                grid, fields_with_time,
                save_times_out_fr,
                field_names_out,
                units_out,
                scales_out,
                solute_type,
                simulation_mode, # for time in label
                fig_path=fig_dir+fig_name,
                figsize = figsize_scalar_fields,
                SIM_N = SIM_N,
                no_ticks=[6,6], 
                alpha = 1.0,
                TTFS = 10, LFS = 10, TKFS = 8,
                cbar_precision = 2,
                show_target_cells = show_target_cells,
                target_cell_list = target_cell_list,
                show_target_cell_labels = True,
                no_cells_x = no_cells_x,
                no_cells_z = no_cells_z)     
    plt.close("all")   

#%% PLOT SIZE SPECTRA AVERAGED

if act_plot_spectra_avg:
    
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
    
    # if not set manually above:
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
        
        
        print("no_bins_R_p")
        print(no_bins_R_p)
        print("no_bins_R_s")
        print(no_bins_R_s)
    
    j_low = target_cell_list[1].min()
    j_high = target_cell_list[1].max()    
    t_low = save_times_out_spectra.min()
    t_high = save_times_out_spectra.max()    

    no_tg_cells = no_rows * no_cols

    fig_path = fig_dir
    if not os.path.exists(fig_path):
        os.makedirs(fig_path) 

    fig_path_spectra =\
        fig_path \
        + f"spectra_Nc_{no_tg_cells}_" + figname_base + ".pdf"
    fig_path_tg_cells =\
        fig_path \
        + f"target_c_pos_Nc_{no_tg_cells}_" + figname_base + ".pdf"
    fig_path_R_eff =\
        fig_path \
        + f"R_eff_Nc_{no_tg_cells}_" + figname_base + ".pdf"
    plot_size_spectra_vs_R(
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
            SIM_N,
            TTFS=10, LFS=10, TKFS=8, LW = 2.0, MS = 0.4,
            figsize_spectra = figsize_spectra,
            figsize_trace_traj = figsize_tg_cells,
            fig_path = fig_path_spectra,
            show_target_cells = True,
            fig_path_tg_cells = fig_path_tg_cells   ,
            fig_path_R_eff = fig_path_R_eff,
            trajectory = None,
            show_textbox=False)    
    plt.close("all")    
