#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 12:37:53 2020

@author: bohrer
"""

import numpy as np
import matplotlib.pyplot as plt

import file_handling as fh

###

# needed for filename
#no_cells = (10, 10)
#no_cells = (15, 15)
no_cells = (50, 50)
#no_cells = (75, 75)

# solute material: NaCl OR ammonium sulfate
#solute_type = "NaCl"
solute_type = "AS"

# no_super_particles_cell_mode = [N1,N2] is a list with
# N1 = no super part. per cell in mode 1 etc.
# with init method = SingleSIP, this is only the target value.
# the true number of particles per cell and mode will fluctuate around this
#no_spcm = np.array([2, 2])
#no_spcm = np.array([6, 10])
no_spcm = np.array([10, 14])
#no_spcm = np.array([16, 24])
#no_spcm = np.array([20, 30])
#no_spcm = np.array([26, 38])
#no_spcm = np.array([52, 76])

seed_SIP_gen = 1005
seed_sim = 1005

#no_seeds = 1
#no_seeds = 10
#no_seeds = 20
#no_seeds = 50

t_grid = 1800

simdata_path = '/Users/bohrer/sim_data_cloudMP_ab_Jan20/'

data_path = simdata_path \
        + f"{solute_type}" \
        + f"/grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm[0]}_{no_spcm[1]}/"\
        + f"{seed_SIP_gen}/" \
        + f"w_spin_up_w_col/{seed_sim}/"
#        + f"spin_up_wo_col_wo_grav/"
        
grid, pos, cells, vel, m_w, m_s, xi, active_ids = \
        fh.load_grid_and_particles_full(t_grid, data_path)        
        
delta_r_v_w_relax = np.load(data_path + f"delta_r_v_w_relax_{t_grid}.npy")
delta_r_v_wo_relax = np.load(data_path + f"delta_r_v_wo_relax_{t_grid}.npy")
delta_Theta_w_relax = np.load(data_path + f"delta_Theta_w_relax_{t_grid}.npy")
delta_Theta_wo_relax = np.load(data_path + f"delta_Theta_wo_relax_{t_grid}.npy")


#fields = [delta_r_v_w_relax, delta_r_v_wo_relax,
#          delta_Theta_w_relax, delta_Theta_wo_relax]

#fields1 = (delta_r_v_w_relax - delta_r_v_wo_relax) / delta_r_v_wo_relax
#fields2 = (delta_Theta_w_relax - delta_Theta_wo_relax) / delta_Theta_wo_relax

fields1 = (delta_r_v_w_relax - delta_r_v_wo_relax)
fields2 = (delta_Theta_w_relax - delta_Theta_wo_relax)

#fields1 = np.where(np.abs(delta_r_v_wo_relax) > 1E-8,
#                   (delta_r_v_w_relax - delta_r_v_wo_relax) / delta_r_v_wo_relax,
#                   delta_r_v_w_relax - delta_r_v_wo_relax)
#fields2 = np.where(np.abs(delta_Theta_wo_relax) > 1E-4,
#                   (delta_Theta_w_relax - delta_Theta_wo_relax) / delta_Theta_wo_relax,
#                   delta_Theta_w_relax - delta_Theta_wo_relax)

fields = [fields1, fields2,
          delta_r_v_wo_relax, delta_Theta_wo_relax,
          grid.mixing_ratio_water_vapor, grid.potential_temperature]

tick_ranges = grid.ranges
no_ticks = [6,6]

TKFS = 8
LFS = 10

no_rows = 3
no_cols = 2
fig, axes = plt.subplots(nrows=no_rows, ncols=no_cols,
                         figsize = (6*no_cols, 5*no_rows))

for ax_cnt in range(len(fields)):
    
    ax = axes.flatten()[ax_cnt]
    field = fields[ax_cnt]
#    
    cmap = "coolwarm"
    alpha = 1.0
#   
    field_min = field.min()
    field_max = field.max()
#    
    field_extr = max(abs(field_min), abs(field_max))
#    
#    
    if ax_cnt < 4:
        CS = ax.pcolormesh(*grid.corners, field,
                           cmap=cmap, alpha=alpha,
                            edgecolor="face",
                            zorder=1,
                            vmin = -field_extr,
                            vmax = field_extr)
    else:        
        CS = ax.pcolormesh(*grid.corners, field,
                           cmap=cmap, alpha=alpha,
                            edgecolor="face",
                            zorder=1,
                            vmin = field_min,
                            vmax = field_max)
##                        norm = norm_(vmin=field_min, vmax=field_max)
##                        )
    CS.cmap.set_under("white")
    
    ax.set_xticks( np.linspace( tick_ranges[0,0],
                                     tick_ranges[0,1],
                                     no_ticks[0] ) )
    ax.set_yticks( np.linspace( tick_ranges[1,0],
                                     tick_ranges[1,1],
                                     no_ticks[1] ) )
    ax.tick_params(axis='both', which='major', labelsize=TKFS)
    ax.grid(color='gray', linestyle='dashed', zorder = 2)
    ax.set_xlabel(r'x (m)', fontsize = LFS)
    ax.set_ylabel(r'z (m)', fontsize = LFS)    

    cbar = plt.colorbar(CS, ax=ax,
#                        format=mticker.FormatStrFormatter(str_format)
                        )    
fig.tight_layout()

fig.savefig(data_path + "compare_relax_term_rel_err.png")
    
    