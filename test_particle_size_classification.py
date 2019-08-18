#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 13:44:24 2019

@author: jdesk
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
# import os
# from datetime import datetime
# import timeit

import constants as c
from microphysics import compute_R_p_w_s_rho_p_AS
from microphysics import compute_R_p_w_s_rho_p_NaCl
from microphysics import compute_radius_from_mass_vec
from file_handling import load_grid_and_particles_full,\
                          load_grid_scalar_fields\

@njit()
def update_mixing_ratio(mixing_ratio, m_w, xi, cells, mass_dry_inv, 
                        id_list, mask):
    for ID in id_list[mask]:
        mixing_ratio[cells[0,ID],cells[1,ID]] += m_w[ID] * xi[ID]
    mixing_ratio *= 1.0E-18 * mass_dry_inv     

#%% STORAGE DIRECTORIES
my_OS = "Linux_desk"
#my_OS = "Mac"

if(my_OS == "Linux_desk"):
    home_path = '/home/jdesk/'
    simdata_path = "/mnt/D/sim_data_cloudMP/"
#    fig_path = home_path + 'Onedrive/Uni/Masterthesis/latex/Report/Figures/'
elif (my_OS == "Mac"):
    simdata_path = "/Users/bohrer/sim_data_cloudMP/"
#    fig_path = home_path \
#               + 'OneDrive - bwedu/Uni/Masterthesis/latex/Report/Figures/'

#%% GRID PARAMETERS

no_cells = (75, 75)
#no_cells = (3, 3)

dx = 20.
dy = 1.
dz = 20.
dV = dx*dy*dz

#%% PARTICLE PARAMETERS

# solute material: NaCl OR ammonium sulfate
#solute_type = "NaCl"
solute_type = "AS"

# no_super_particles_cell_mode = [N1,N2] is a list with
# N1 = no super part. per cell in mode 1 etc.
# with init method = SingleSIP, this is only the target value.
# the true number of particles per cell and mode will fluctuate around this
#no_spcm = np.array([10, 10])
#no_spcm = np.array([12, 12])
no_spcm = np.array([16, 24])

# seed of the SIP generation -> needed for the right grid folder
# 3711, 3713, 3715, 3717
# 3719, 3721, 3723, 3725
#seed_SIP_gen_list = [3711, 3713, 3715, 3717]
seed_SIP_gen_list = [3711]
seed_sim_list = [4711]

simulation_mode = "spin_up"
#simulation_mode = "wo_collision"
#simulation_mode = "with_collision"

spin_up_finished = True
#spin_up_finished = False

# path = simdata_path + folder_load_base
t_grid = 0
#t = 60
#t = 3600
#t_grid = 7200
#t = 14400
# t = 10800

t_start = 0
#t_start = 7200

#t_end = 60
#t_end = 3600
t_end = 7200
# t_end = 10800
#t_end = 14400

if solute_type == "AS":
    compute_R_p_w_s_rho_p = compute_R_p_w_s_rho_p_AS
    mass_density_dry = c.mass_density_AS_dry
elif solute_type == "NaCl":
    compute_R_p_w_s_rho_p = compute_R_p_w_s_rho_p_NaCl
    mass_density_dry = c.mass_density_NaCl_dry

for seed_n,seed_SIP_gen in enumerate(seed_SIP_gen_list):
#seed_SIP_gen = 3711

    # for collisons
    seed_sim = seed_sim_list[seed_n]
    
    #%% LOAD GRID AND PARTICLES AT TIME t_grid
    
    grid_folder =\
        f"{solute_type}" \
        + f"/grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm[0]}_{no_spcm[1]}/" \
        + f"{seed_SIP_gen}/"
    
    if simulation_mode == "spin_up":
        save_folder = "spin_up_wo_col_wo_grav/"
    elif simulation_mode == "wo_collision":
        if spin_up_finished:
            save_folder = "w_spin_up_wo_col/"
        else:
            save_folder = "wo_spin_up_wo_col/"
    elif simulation_mode == "with_collision":
        if spin_up_finished:
            save_folder = f"w_spin_up_w_col/{seed_sim}/"
        else:
            save_folder = f"wo_spin_up_w_col/{seed_sim}/"
    
    # load grid and particles full from grid_path at time t
    if int(t_grid) == 0:        
        grid_path = simdata_path + grid_folder
    elif spin_up_finished:    
        grid_path = simdata_path + grid_folder + "spin_up_wo_col_wo_grav/"
    else:    
        grid_path = simdata_path + grid_folder + save_folder
    
    load_path = simdata_path + grid_folder + save_folder
    
#    if simulation_mode == 
    
    #reload = True
    #
    #if reload:
    grid, pos, cells, vel, m_w, m_s, xi, active_ids  = \
        load_grid_and_particles_full(t_grid, grid_path)
    
    #%%
    
    save_times = np.load(load_path + "grid_save_times.npy")

    R_p, w_s, rho_p = compute_R_p_w_s_rho_p(m_w, m_s,
                                            grid.temperature[tuple(cells)] )
    R_s = compute_radius_from_mass_vec(m_s, mass_density_dry)
    
    #mask1 = R_p < 0.5
    #mask3 = R_p > 25.0
    #
    ##mask2 = np.logical_and( np.invert(mask1), np.invert(mask3))
    ## only possible, because mask1 and mask3 cant exist both AND there are only
    ## 3 opportunities
    #mask2 = mask1 == mask3
    #
    #print(mask1.shape)
    #print(mask2.shape)
    #print(mask3.shape)
    #
    #masks = np.array((mask1, mask2, mask3))
    #
    #sum_ = 0
    #
    #for mask in masks:
    #    sum_ += mask.sum()
    #
    #no_idx = masks.sum(axis=1)
    #    
    #print(sum_)    
    #no_rows= 1
    ##fig, ax = plt.subplots(no_rows, figsize = (18,12))
    #for i, mask in enumerate(masks):    
    #    if i == 0:
    #        low_border = 0
    #    else:
    #        low_border += no_idx[i-1]
    #    ax.plot(R_p[mask], np.arange(low_border,low_border+no_idx[i]) , "o", markersize = 1)
    
    #%% TEST THE INDEXING FOR RADIUS BASED CLASSIFICATION
#    idx_01 = np.where(R_p < 0.5)
#    
#    bins = [0.0,0.5,25,np.inf]
#    bins2 = [0.5,25]
#    
#    h1 = np.histogram(R_p, bins)
#    D1 = np.digitize(R_p, bins2)
#    
#    mask_D1 = D1 == 0
#    mask_D2 = D1 == 1
#    mask_D3 = D1 == 2
#    
#    ### TESTED: mask_D == masks is true for all indices
#    mask_D = np.array((mask_D1, mask_D2, mask_D3))
#    
#    AA = np.arange(3).reshape((3,1))
#    
#    ### TESTED: mask_D_AA == mask_D is true for all indices
#    masks_D_AA = AA == D1
#    
#    no_idx_2 = masks_D_AA.sum(axis=1)
#    
#    no_rows= 1
#    #fig, ax = plt.subplots(no_rows, figsize = (18,12))
#    for i, mask in enumerate(masks_D_AA):    
#        if i == 0:
#            low_border = 0
#        else:
#            low_border += no_idx_2[i-1]
    #    ax.plot(R_p[mask],
    #            np.arange(low_border,low_border+no_idx[i]) , "o", markersize = 1)
    
    
    #%%
    
    
        
    no_SIPS = len(xi)
    
    bins_drop_class = [0.5,25]
    idx_R_p = np.digitize(R_p, bins_drop_class)
    idx_classification = np.arange(3).reshape((3,1))
    
    masks_R_p = idx_classification == idx_R_p
    
    
    mixing_ratio_water_aerosols = np.zeros_like(grid.mixing_ratio_water_liquid)
    mixing_ratio_water_cloud = np.zeros_like(grid.mixing_ratio_water_liquid)
    #mixing_ratio_water_cloud2 = np.zeros_like(grid.mixing_ratio_water_liquid)
    mixing_ratio_water_rain = np.zeros_like(grid.mixing_ratio_water_liquid)
    
    mixing_ratio_water = np.array((mixing_ratio_water_aerosols,
                                   mixing_ratio_water_cloud,
                                   mixing_ratio_water_rain))
    
    active_ids = np.ones(no_SIPS, dtype=np.bool)
    id_list = np.arange(len(xi))
    
    for mask_n in range(3):
        mask = np.logical_and(masks_R_p[mask_n], active_ids)
        update_mixing_ratio(mixing_ratio_water[mask_n], m_w, xi, cells,
                            grid.mass_dry_inv, 
                            id_list, mask)
        
    mixing_ratio_water_sum = mixing_ratio_water.sum(axis = 0)
    
    print(mixing_ratio_water_sum - grid.mixing_ratio_water_liquid)
    print()
    
    #mixing_ratio_water_sum[np.abs(mixing_ratio_water_sum - grid.mixing_ratio_water_liquid) > 1E-16]
    #np.abs(mixing_ratio_water_sum - grid.mixing_ratio_water_liquid) > 1E-16
    
    # REL DEV TESTED: in each cell |r_aero + r_c + r_r - r_l| / r_l < 1E-15
    print(np.where(np.abs(mixing_ratio_water_sum - grid.mixing_ratio_water_liquid)
                    /grid.mixing_ratio_water_liquid > 1E-15))
    
    #for ID in id_list[mask]:
    #    mixing_ratio_water_cloud[cells[0,ID],cells[1,ID]] += m_w[ID] * xi[ID]
    #mixing_ratio_water_cloud *= 1.0E-18 * grid.mass_dry_inv        
    
    #update_mixing_ratio(mixing_ratio_water_cloud, m_w, xi, cells,
    #                    grid.mass_dry_inv, 
    #                    id_list, mask)
    mixing_ratio_water *= 1E3
    
    #%%
    # units g/kg
    
    no_cols = 3
    fig, axes = plt.subplots(ncols=no_cols, figsize = (15,4))
    
    tick_ranges = grid.ranges
    no_ticks=[6,6]
    field_min = 1E-7
    import matplotlib as mpl
    import cmocean.cm as cmo
    #cmap = "rainbow"
    #cmap = "gist_rainbow_r"
    #cmap = "nipy_spectral"
    #cmap = "nipy_spectral_r"
    #cmap = "gist_ncar_r"
    cmap = "cubehelix_r"
    #cmap = "viridis_r"
    #cmap = "plasma_r"
    #cmap = "magma_r"
    cmap = cmo.rain
    alpha = 0.7
    
    colors1 = plt.cm.get_cmap('gist_ncar_r', 256)
    #top = plt.cm.get_cmap('gist_rainbow_r', 256)
    colors2 = plt.cm.get_cmap('rainbow', 256)
    #top = plt.cm.get_cmap('Greys', 128)
    #bottom = plt.cm.get_cmap('Blues', 128)
    
    newcolors = np.vstack((colors1(np.linspace(0, 0.16, 16)),
                           colors2(np.linspace(0, 1, 256))))
    #newcolors = np.vstack((top(np.linspace(0, 1, 128)),
    #                       bottom(np.linspace(0, 1, 128))))
    cmap_new = mpl.colors.ListedColormap(newcolors, name='my_rainbow')
    
    import matplotlib.ticker as mticker
    class FormatScalarFormatter(mticker.ScalarFormatter):
        def __init__(self, fformat="%1.1f", offset=True, mathText=True):
            self.fformat = fformat
            mticker.ScalarFormatter.__init__(self,useOffset=offset,
                                                            useMathText=mathText)
    #    def _set_format(self, vmin, vmax):
        def _set_format(self):
            self.format = self.fformat
            if self._useMathText:
                self.format = '$%s$' % mticker._mathdefault(self.format)
                
    def fmt(x, pos):
        a, b = '{:.2e}'.format(x).split('e')
#        a = float(a)
        b = int(b)
        return r'${} \times 10^{{{}}}$'.format(a, b)
    
#    def fmt_once(x, pos, oom_max):
#        a, b = '{:.2e}'.format(x).split('e')
#        b = int(b) - oom_max
#        return r'${:1.2f}$'.format(a*10**(b))    
#    def fmt_once(x, pos, oom_max):
#        a, b = '{:.2e}'.format(x).split('e')
#        b = int(b)
#        return r'${}$'.format(a*10**(b))    
    
    def compute_order_of_magnitude(x):
        a, b = '{:.2e}'.format(x).split('e')
        b = int(b)
        return b   

    cbar_precision = 2      
    def fmt_cbar(x, pos):
#        a, b = '{:.2e}'.format(x).split('e')
#        a = float(a)
#        b = int(b) - oom_max
        return r'${0:.{prec}f}$'.format(x, prec=cbar_precision)     
    
    field_names = ["r_\mathrm{aero}", "r_c", "r_r"]
    units = ["g/kg", "g/kg", "g/kg"]

#    cbar = []
#    CS = []
    TTFS = 16
    TKFS = 12
    LFS = 12
    for j in range(3):
        ax = axes[j]
        field = mixing_ratio_water[j]
#        field_min = field.min()
    #    field_min = field.min()*100
        field_max = field.max()
#        oom_max = compute_order_of_magnitude(field_max)
        oom_max = oom = int(math.log10(field_max))
        oom_factor = 10**(-oom)
        
        field_min = field.min() * oom_factor
        field_max *= oom_factor
        
        
#        oom_min = compute_order_of_magnitude(field_min)
#        CS.append(ax.pcolormesh(*grid.corners, field, cmap=cmap, alpha=alpha,
#                                vmin=field_min, vmax=field_max,
#                                edgecolor="face", zorder=1
#                                ))
#        CS[j].cmap.set_under("white")
        CS = ax.pcolormesh(*grid.corners, field*oom_factor,
                           cmap=cmap, alpha=alpha,
                            vmin=field_min, vmax=field_max,
                            edgecolor="face", zorder=1
                            )
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
        ax.set_title( r"${0}$ ({1})".format(field_names[j], units[j]),
                     fontsize = TTFS)
    #    cbar.append(fig.colorbar(CS, ax=ax,
    #                        format=FormatScalarFormatter(
    #                                fformat="%.2f",
    #                                offset=True,
    #                                mathText=True)))
    #                        extend = "min",
    #                        format=OOMFormatter(-8, mathText=True))
    #    cbar = fig.colorbar(CS, ax=ax,
    ##                        extend = "min",
    #                        format=OOMFormatter(-8, mathText=True))
        cbar = plt.colorbar(CS, ax=ax, extend = "min",
                            format=mticker.FuncFormatter(fmt_cbar))
#        cbar = plt.colorbar(CS[j], ax=ax, extend = "min",
#                            format=mticker.FuncFormatter(fmt_once))
        cbar.ax.text(-2.3*(field_max-field_min), field_max*1.01,
                     r'$\times\,10^{{{}}}$'.format(oom_max),
                     va='bottom', ha='left', fontsize = TKFS)
        cbar.ax.tick_params(labelsize=TKFS)
        
    #    cbar.ax.text(-0.25, 1, r'$\times$10$^{-3}$', va='bottom', ha='left')
    #    cbar.ax.text(0.0,0.0,"ha")
    fig.tight_layout()
