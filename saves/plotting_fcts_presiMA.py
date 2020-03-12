#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 19:52:25 2019

@author: jdesk
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import hex2color, LinearSegmentedColormap
import matplotlib.ticker as mticker

import os
import math
import numpy as np
# import os
# from datetime import datetime
# import timeit

from microphysics import compute_R_p_w_s_rho_p_AS
from microphysics import compute_R_p_w_s_rho_p_NaCl
from microphysics import compute_radius_from_mass_vec
from microphysics import compute_mass_from_radius_vec
from file_handling import load_grid_and_particles_full,\
                          load_grid_scalar_fields\

import constants as c

from plotting import cm2inch
#from plotting import generate_rcParams_dict
#from plotting import pgf_dict, pdf_dict
#plt.rcParams.update(pgf_dict)
#plt.rcParams.update(pdf_dict)

from file_handling import load_grid_and_particles_full,\
                          load_grid_scalar_fields\

from analysis import sample_masses, sample_radii
from analysis import sample_masses_per_m_dry , sample_radii_per_m_dry
from analysis import plot_size_spectra_R_Arabas, generate_size_spectra_R_Arabas

                          
                          
colors1 = plt.cm.get_cmap('gist_ncar_r', 256)
#top = plt.cm.get_cmap('gist_rainbow_r', 256)
colors2 = plt.cm.get_cmap('rainbow', 256)
#top = plt.cm.get_cmap('Greys', 128)
#bottom = plt.cm.get_cmap('Blues', 128)

newcolors = np.vstack((colors1(np.linspace(0, 0.16, 24)),
                       colors2(np.linspace(0, 1, 256))))
#newcolors = np.vstack((top(np.linspace(0, 1, 128)),
#                       bottom(np.linspace(0, 1, 128))))
cmap_new = mpl.colors.ListedColormap(newcolors, name='my_rainbow')

### CREATE COLORMAP LIKE ARABAS 2015
hex_colors = ['#FFFFFF', '#993399', '#00CCFF', '#66CC00',
              '#FFFF00', '#FC8727', '#FD0000']
rgb_colors = [hex2color(cl) + tuple([1.0]) for cl in hex_colors]
no_colors = len(rgb_colors)

cdict_lcpp_colors = np.zeros( (3, no_colors, 3) )

for i in range(3):
    cdict_lcpp_colors[i,:,0] = np.linspace(0.0,1.0,no_colors)
    for j in range(no_colors):
        cdict_lcpp_colors[i,j,1] = rgb_colors[j][i]
        cdict_lcpp_colors[i,j,2] = rgb_colors[j][i]

cdict_lcpp = {"red": cdict_lcpp_colors[0],
              "green": cdict_lcpp_colors[1],
              "blue": cdict_lcpp_colors[2]}

cmap_lcpp = LinearSegmentedColormap('testCmap', segmentdata=cdict_lcpp, N=256)

#%% FUNCTION DEF: PLOT SCALAR FIELDS
def plot_scalar_field_frames_extend_avg_PRESI(grid, fields_with_time,
                                        save_times,
                                        field_names,
                                        units,
                                        scales,
                                        solute_type,
                                        simulation_mode, # for time in label
                                        fig_path,
                                        figsize,
                                        SIM_N=None,
                                        no_ticks=[6,6],
                                        alpha = 1.0,
                                        TTFS = 12, LFS = 10, TKFS = 10,
                                        cbar_precision = 2,
                                        show_target_cells = False,
                                        target_cell_list = None,
                                        no_cells_x = 0,
                                        no_cells_z = 0
                                        ):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    for i,fm in enumerate(field_names):
        print(i,fm)
    print(save_times)
    
    tick_ranges = grid.ranges
#    tick_ranges_label = [[0,1.2],[0,1.2]]

    no_rows = len(field_names)
    no_cols = len(save_times)
#    no_cols = len(field_names)
    
    if SIM_N == 7:
        print ("target_cell_list")
        print (target_cell_list)
        target_cell_list = np.delete(target_cell_list, np.s_[2::3], 1 )
        target_cell_list = np.delete(target_cell_list, np.s_[8:10:], 1 )
        print (target_cell_list)        
    
#    fig, axes = plt.subplots(nrows=no_rows, ncols=no_cols,
#                       figsize = (4.5*no_cols, 4*no_rows),
#                       sharex=True, sharey=True)    
    fig, axes = plt.subplots(nrows=no_rows, ncols=no_cols,
                       figsize = figsize,
                       sharex=True, sharey=True)
    
    for field_n in range(no_rows):
        for time_n in range(no_cols):
#            ax = axes[time_n,field_n]
            ax = axes[field_n,time_n]
            field = fields_with_time[time_n, field_n] * scales[field_n]
            ax_title = field_names[field_n]
            unit = units[field_n]
            if ax_title in ["T","p",r"\Theta"]:
                cmap = "coolwarm"
                alpha = 1.0
            else :
#                cmap = "rainbow"
                cmap = cmap_lcpp
#                alpha = 0.8
                
            field_max = field.max()
            field_min = field.min()
            
            xticks_major = None
            xticks_minor = None
            
            title_add = None
            title_add2 = None
            
            norm_ = mpl.colors.Normalize 
            if ax_title in ["r_r", "n_r"]: #and field_max > 1E-2:
                norm_ = mpl.colors.LogNorm
                field_min = 0.01
                cmap = cmap_lcpp                
                if ax_title == "r_r":
                    field_max = 1.
                    xticks_major = [0.01,0.1,1.]
                    xticks_minor = np.concatenate((
                            np.linspace(2E-2,1E-1,9),
                            np.linspace(2E-1,1,9),
                            ))
                    title_add = "R_p > \SI{25}{\micro m}"
                elif ax_title == "n_r":
                    field_max = 10.
                    xticks_major = [0.01,0.1,1.,10.]
                    xticks_minor = np.concatenate((
                            np.linspace(2E-2,1E-1,9),
                            np.linspace(2E-1,1,9),
                            np.linspace(2,10,9),
                            ))
            else: norm_ = mpl.colors.Normalize   
            
            if ax_title == r"\Theta":
#                if SIM_N == 7:
#                    field_min = 289.0
#                    field_max = 292.5
#                    xticks_major = [289,290,291,292]
#                else:
                field_min = 289.2
                field_max = 292.5
                xticks_major = [290,291,292]
            if ax_title == "r_v":
                field_min = 6.5
                field_max = 7.6
                xticks_minor = [6.75,7.25]
            if ax_title == "r_l":
                field_min = 0.0
                field_max = 1.3
                xticks_minor = [0.25,0.75,1.25]
                
            if ax_title == "r_c":
                field_min = 0.0
                field_max = 1.3
#                xticks_major = np.linspace(0,1.2,7)
#                xticks_major = np.linspace(0,1.25,6)
                xticks_major = [0,0.5,1]
                xticks_minor = [0.25,0.75,1.25]     
                title_add = "R_p > \SI{0.5}{\micro m}"
                title_add2 = "R_p \leq \;\SI{25}{\micro m}"
            if ax_title == "n_c":
                field_min = 0.0
                field_max = 150.
                xticks_major = [0,50,100,150]
                xticks_minor = [25,75,125]                
            if ax_title == "n_\mathrm{aero}":
                field_min = 0.0
                field_max = 150.
                xticks_major = [0,50,100,150]
#                xticks_minor = [25,75,125]
                title_add = "R_p < \SI{0.5}{\micro m}"
            if ax_title in [r"R_\mathrm{avg}", r"R_{2/1}", r"R_\mathrm{eff}"]:
                xticks_major = [1,5,10,15,20]
#                field_min = 0.
                field_min = 1
#                field_min = 1.5
                field_max = 20.
#                cmap = cmap_new
                # Arabas 2015
                cmap = cmap_lcpp
                unit = r"\si{\micro\meter}"
                
            oom_max = oom = int(math.log10(field_max))
            
            my_format = False
            oom_factor = 1.0
            
            if oom_max > 2 or oom_max < 0:
                my_format = True
                oom_factor = 10**(-oom)
                
                field_min *= oom_factor
                field_max *= oom_factor            
            
            if oom_max ==2: str_format = "%.0f"
#            if oom_max ==2: str_format = "%.1f"
            
            else: str_format = "%.2g"
#            else: str_format = "%.2f"
            
            if field_min/field_max < 1E-4:
#                cmap = cmap_new
                # Arabas 2015                
                cmap = cmap_lcpp
#                alpha = 0.8
            
            # REMOVE FIX APLHA HERE
#            alpha = 1.0
#        CS = ax.pcolormesh(*grid.corners, grid_r_l,
#                           cmap=cmap, alpha=alpha,
#                            edgecolor="face", zorder=1,
#                            vmin=field_min, vmax=field_max,
#                            antialiased=True, linewidth=0.0
##                            norm = norm_(vmin=field_min, vmax=field_max)
#                            )            
            CS = ax.pcolormesh(*grid.corners, field*oom_factor,
                               cmap=cmap, alpha=alpha,
                                edgecolor="face", zorder=1,
                                norm = norm_(vmin=field_min, vmax=field_max),
                                rasterized=True,
                                antialiased=True, linewidth=0.0
                                )
            if ax_title == r"\Theta":
                cmap_x = mpl.cm.get_cmap('coolwarm')
                print("cmap_x(0.0)")
                print(cmap_x(0.0))
                CS.cmap.set_under(cmap_x(0.0))
#                pass
            else:
#            if ax_title != r"\Theta":
                CS.cmap.set_under("white")
            
            ax.set_xticks( np.linspace( tick_ranges[0,0],
                                             tick_ranges[0,1],
                                             no_ticks[0] ) )
            ax.set_yticks( np.linspace( tick_ranges[1,0],
                                             tick_ranges[1,1],
                                             no_ticks[1] ) )

#            ax.set_xticks( np.linspace( tick_ranges[0,0],
#                                             tick_ranges[0,1],
#                                             no_ticks[0] ) )
#            ax.set_yticks( np.linspace( tick_ranges[1,0],
#                                             tick_ranges[1,1],
#                                             no_ticks[1] ) )
#            ax.tick_params(axis='both', which='major', labelsize=TKFS)
            ax.tick_params(axis='both', which='major', labelsize=TKFS,
                           length = 2.5, width=0.6)
#            ax.tick_params(axis='both', which='minor', labelsize=TKFS,
#                           length = 3)            
            ax.grid(color='gray', linestyle='dashed', zorder = 2)
            ax.set_aspect('equal')
            if field_n == no_rows-1:
#                tlabels = ax.get_xticklabels()
#                print(tlabels)
#                tlabels[-1] = ""
#                print(tlabels)
#                ax.set_xticklabels(tlabels)    
#                xticks1 = ax.xaxis.get_major_ticks()
#                xticks1[-1].label1.set_visible(False)
                ax.set_xlabel(r'$x$ (km)', fontsize = LFS)
            if time_n == 0:            
                ax.set_ylabel(r'$z$ (km)', fontsize = LFS)
                if field_n >= 1:
                    yticks1 = ax.yaxis.get_major_ticks()
                    yticks1[-1].label1.set_visible(False)
#            if time_n == 0:
#                ax.set_title(
#    r"\begin{{center}}${0}$ ({1})\\ t = 0\end{{center}}".format(ax_title, unit),
#                             fontsize = TTFS)
#                ax.set_title(
#    r"\begin{{center}}${0}$ ({1})\\ t = 0\end{{center}}".format(ax_title, unit),
#                             fontsize = TTFS)
#            else:  
            if field_n == 0:
                ax.set_title( r"$t$ = {0} min".format(int(save_times[time_n]/60)),
                             fontsize = TTFS, pad=2)
#            ax.set_title( r"${0}$ ({1}), t = {2} min".format(ax_title, unit,
#                         int(save_times[time_n]/60)),
#                         fontsize = TTFS)
            if time_n == no_cols-1:
#                if no_cols == 4:
#            if time_n == no_rows - 1:
                axins = inset_axes(ax,
                                   width="10%",  # width = 5% of parent_bbox width
                                   height="100%",  # height
                                   loc='lower right',
                                   bbox_to_anchor=(0.14, 0.0, 1, 1),
#                                   bbox_to_anchor=(0.3, 0.0, 1, 1),
#                                   , 1, 1),
                                   bbox_transform=ax.transAxes,
                                   borderpad=0,
                                   )      
#                else:
#                    axins = inset_axes(ax,
#                                       width="90%",  # width = 5% of parent_bbox width
#                                       height="8%",  # height
#                                       loc='lower center',
#                                       bbox_to_anchor=(0.0, 1.4, 1, 1),
#    #                                   , 1, 1),
#                                       bbox_transform=ax.transAxes,
#                                       borderpad=0,
#                                       )      
#                divider = make_axes_locatable(ax)
#                cax = divider.append_axes("top", size="6%", pad=0.3)
                
                cbar = plt.colorbar(CS, cax=axins,
#                                    fraction=0.046, pad=-0.1,
                                    format=mticker.FormatStrFormatter(str_format),
                                    orientation="vertical"
                                    )
                
#                axins.yaxis.set_ticks_position("left")
                
#                axins.xaxis.set_ticks_position("top")
                axins.tick_params(axis="y",direction="inout",which="both")
#                axins.tick_params(axis="x",direction="inout")
                axins.tick_params(axis='y', which='major', labelsize=TKFS,
                               length = 5, width=0.8)                
                axins.tick_params(axis='y', which='minor', labelsize=TKFS,
                               length = 3, width=0.5,bottom=True)                
                
                
                if xticks_major is not None:
                    axins.yaxis.set_ticks(xticks_major)
                if xticks_minor is not None:
                    axins.yaxis.set_ticks(xticks_minor, minor=True)
                if ax_title == "n_c":
                    xticks2 = axins.xaxis.get_major_ticks()
                    xticks2[-1].label1.set_visible(False)                
#                axins.set_ylabel(r"${0}$ ({1})".format(ax_title, unit))
                
                if title_add is None:
                    cbar.set_label(r"${0}$ ({1})".format(ax_title, unit),rotation=0)
                elif title_add2 is None:
                    cbar.set_label(r"${0}$ ({1})".format(ax_title, unit)
                                   +"\n"
                                   +r"(${0}$)".format(title_add),
                                   rotation=0,
                                   labelpad=23,
                                   y = 0.65)
                else:
                    cbar.set_label(r"${0}$ ({1})".format(ax_title, unit)
                                   +"\n"
                                   +r"(${0}$".format(title_add)
                                   +"\n"
                                   +r"\;\;${0}$)".format(title_add2),
                                   rotation=0,
                                   labelpad=23,
                                   y = 0.7)
                    
#                    cbar.axis.set_label_coords(0.05, 0.75)
            # my_format dos not work with log scale here!!

                if my_format:
                    cbar.ax.text(field_min - (field_max-field_min),
                                 field_max + (field_max-field_min)*0.01,
                                 r'$\times\,10^{{{}}}$'.format(oom_max),
                                 va='bottom', ha='left', fontsize = TKFS)
                cbar.ax.tick_params(labelsize=TKFS)

            if show_target_cells:
                ### ad the target cells
                no_neigh_x = no_cells_x // 2
                no_neigh_z = no_cells_z // 2
                dx = grid.steps[0]
                dz = grid.steps[1]
                
                no_tg_cells = len(target_cell_list[0])
                LW_rect = .5
                for tg_cell_n in range(no_tg_cells):
                    x = (target_cell_list[0, tg_cell_n] - no_neigh_x - 0.1) * dx
                    z = (target_cell_list[1, tg_cell_n] - no_neigh_z - 0.1) * dz
                    
            #        dx *= no_cells_x
            #        dz *= no_cells_z
                    
                    rect = plt.Rectangle((x, z), dx*no_cells_x,dz*no_cells_z,
                                         fill=False,
                                         linewidth = LW_rect,
        #                                 linestyle = "dashed",
                                         edgecolor='k',
                                         zorder = 99)        
                    ax.add_patch(rect)

#    if no_cols == 4:
#    pad_ax_h = 0.1
#    pad_ax_v = 0.05
    pad_ax_h = 0.08
    pad_ax_v = -0.04
#    else:        
#        pad_ax_h = -0.5 
#        pad_ax_v = 0.08
    #    pad_ax_v = 0.005
    fig.subplots_adjust(hspace=pad_ax_h, wspace=pad_ax_v)
#    fig.subplots_adjust(wspace=pad_ax_v)
             
#    fig.tight_layout()
    if fig_path is not None:
#        if 
#        DPI =
        fig.savefig(fig_path,
    #                    bbox_inches = 0,
                    bbox_inches = 'tight',
                    pad_inches = 0.03,
#                    dpi=300,
                    dpi=600
                    )   
        
        
        
        
#%% FUNCTION DEF: PLOT ABSOLUTE DEVS OF TWO SCALAR FIELDS
def plot_scalar_field_frames_abs_dev_PRESI(grid,
                                        fields_with_time1,
                                        fields_with_time_std1,
                                        fields_with_time2,
                                        fields_with_time_std2,
                                        save_times,
                                        field_names,
                                        units,
                                        scales,
                                        solute_type,
                                        simulation_mode, # for time in label
                                        compare_type,
                                        fig_path,
                                        fig_path_abs_err,
                                        figsize,
                                        no_ticks=[6,6],
                                        alpha = 1.0,
                                        TTFS = 12, LFS = 10, TKFS = 10,
                                        cbar_precision = 2,
                                        show_target_cells = False,
                                        target_cell_list = None,
                                        no_cells_x = 0,
                                        no_cells_z = 0
                                        ):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    title_add = None
    title_add2 = None
    
    for i,fm in enumerate(field_names):
        print(i,fm)
    print(save_times)
    
    tick_ranges = grid.ranges
#    tick_ranges_label = [[0,1.2],[0,1.2]]

    no_cols = len(save_times)
    no_rows = len(field_names)
    
    abs_dev_with_time = fields_with_time2 - fields_with_time1
    
#    fig, axes = plt.subplots(nrows=no_rows, ncols=no_cols,
#                       figsize = (4.5*no_cols, 4*no_rows),
#                       sharex=True, sharey=True)    
    fig, axes = plt.subplots(nrows=no_rows, ncols=no_cols,
                       figsize = figsize,
                       sharex=True, sharey=True)
    
    vline_pos = (0.33, 1.17, 1.33)
    
    for time_n in range(no_cols):
        for field_n in range(no_rows):
#            ax = axes[time_n, field_n]
            ax = axes[field_n, time_n]
            
            if compare_type == "Nsip" and time_n == 2:
                for vline_pos_ in vline_pos:
                    ax.axvline(vline_pos_, alpha=0.5, c="k", zorder= 3, linewidth = 1.3)
            
            field = abs_dev_with_time[time_n, field_n] * scales[field_n]
            ax_title = field_names[field_n]

            print(time_n, ax_title, field.min(), field.max())

            unit = units[field_n]
            if ax_title in ["T","p",r"\Theta"]:
                cmap = "coolwarm"
                alpha = 1.0
            else :
#                cmap = "rainbow"
                cmap = cmap_lcpp
#                alpha = 0.8
                
            field_max = field.max()
            field_min = field.min()
            
            xticks_major = None
            xticks_minor = None
            
            norm_ = mpl.colors.Normalize 

            if compare_type in ["Ncell", "solute", "Kernel"]:
                if ax_title in ["r_r", "n_r"]: #and field_max > 1E-2:
    #                norm_ = mpl.colors.SymLogNorm
                    norm_ = mpl.colors.Normalize
    #                norm_ = mpl.colors.LogNorm
                    cmap = cmap_lcpp                
                    if ax_title == "r_r":
                        field_max = 0.1
                        field_min = -field_max
    #                    linthresh=(-0.01,0.01)
                        linthresh=0.01
    #                    xticks_major = [-0.1, 0, 0.1]
                        xticks_major = np.linspace(field_min,field_max,5)
#                        xticks_major = [field_min, -0.04, 0., 0.04, field_max]
    #                    xticks_major = [0, 0.01, 0.1]
                    elif ax_title == "n_r":
                        pass
                        field_max = 0.9
                        field_min = -field_max
#                        xticks_major = [0.01,0.1,1.,10.]
#                        xticks_minor = np.concatenate((
#                                np.linspace(2E-2,1E-1,9),
#                                np.linspace(2E-1,1,9),
#                                np.linspace(2,10,9),
#                                ))
                else: norm_ = mpl.colors.Normalize   
                
                if ax_title == r"\Theta":
                    field_min = 289.2
                    field_max = 292.5
                    xticks_major = [290,291,292]
                if ax_title == "r_v":
                    field_min = 6.5
                    field_max = 7.6
                    xticks_minor = [6.75,7.25]
                if ax_title == "r_l":
                    field_min = 0.0
                    field_max = 1.3
                    xticks_minor = [0.25,0.75,1.25]
                    
                if ax_title == "r_c":
                    field_max = 0.3
                    field_min = -field_max                
    #                field_min = -0.2
    #                field_max = 0.2
#                    xticks_major = [field_min, -0.05, 0., 0.05, field_max]
                    xticks_major = np.linspace(field_min,field_max,5)
    #                xticks_major = np.linspace(0,1.2,7)
                if ax_title == "n_c":
#                    pass
                    field_max = 40.
                    field_min = -field_max
                if ax_title == "n_\mathrm{aero}":
                    field_max = 40.
                    field_min = -field_max
                    xticks_major = np.linspace(field_min,field_max,5)
#                    xticks_major = [field_min, -4, 0., 4, field_max]
    #                xticks_minor = [25,75,125]
                if ax_title in [r"R_\mathrm{avg}", r"R_{2/1}", r"R_\mathrm{eff}"]:
    #                xticks_major = [1,5,10,15,20]
    #                field_min = 0.
    #                field_min = 1.5
                    field_max = 8.
#                    field_max = 5.
                    field_min = -field_max
                    xticks_major = np.linspace(field_min,field_max,5)
    #                cmap = cmap_new
                    # Arabas 2015
                    cmap = cmap_lcpp
                    unit = r"\si{\micro\meter}"
                    
            elif compare_type == "Nsip":
#            elif compare_type == "dt_col":
                if ax_title in ["r_r", "n_r"]: #and field_max > 1E-2:
    #                norm_ = mpl.colors.SymLogNorm
                    norm_ = mpl.colors.Normalize
    #                norm_ = mpl.colors.LogNorm
                    cmap = cmap_lcpp                
                    if ax_title == "r_r":
                        field_max = 0.08
                        field_min = -field_max
    #                    linthresh=(-0.01,0.01)
                        linthresh=0.01
    #                    xticks_major = [-0.1, 0, 0.1]
                        xticks_major = [field_min, -0.04, 0., 0.04, field_max]
    #                    xticks_major = [0, 0.01, 0.1]
                    elif ax_title == "n_r":
                        field_max = 10.
                        xticks_major = [0.01,0.1,1.,10.]
                        xticks_minor = np.concatenate((
                                np.linspace(2E-2,1E-1,9),
                                np.linspace(2E-1,1,9),
                                np.linspace(2,10,9),
                                ))
                else: norm_ = mpl.colors.Normalize   
                
                if ax_title == r"\Theta":
                    field_min = 289.2
                    field_max = 292.5
                    xticks_major = [290,291,292]
                if ax_title == "r_v":
                    field_min = 6.5
                    field_max = 7.6
                    xticks_minor = [6.75,7.25]
                if ax_title == "r_l":
                    field_min = 0.0
                    field_max = 1.3
                    xticks_minor = [0.25,0.75,1.25]
                    
                if ax_title == "r_c":
                    field_max = 0.12
                    field_min = -field_max                
    #                field_min = -0.2
    #                field_max = 0.2
#                    xticks_major = np.linspace(field_min,field_max,5)
                    xticks_major = [-0.1, -0.05, 0., 0.05, 0.1]
    #                xticks_major = np.linspace(0,1.2,7)
                if ax_title == "n_c":
                    field_min = 0.0
                    field_max = 150.
                if ax_title == "n_\mathrm{aero}":
                    field_max = 20.
                    field_min = -field_max
                    xticks_major = np.linspace(field_min,field_max,5)
#                    xticks_major = [field_min, -4, 0., 4, field_max]
    #                xticks_minor = [25,75,125]
                if ax_title in [r"R_\mathrm{avg}", r"R_{2/1}", r"R_\mathrm{eff}"]:
    #                xticks_major = [1,5,10,15,20]
    #                field_min = 0.
    #                field_min = 1.5
#                    field_max = 8.
                    field_max = 5.
                    field_min = -field_max
                    xticks_major = np.linspace(field_min,field_max,5)
    #                cmap = cmap_new
                    # Arabas 2015
                    cmap = cmap_lcpp
                    unit = r"\si{\micro\meter}"
            else:
#            elif compare_type == "dt_col":
                if ax_title in ["r_r", "n_r"]: #and field_max > 1E-2:
    #                norm_ = mpl.colors.SymLogNorm
                    norm_ = mpl.colors.Normalize
    #                norm_ = mpl.colors.LogNorm
                    cmap = cmap_lcpp                
                    if ax_title == "r_r":
                        field_max = 0.08
                        field_min = -field_max
    #                    linthresh=(-0.01,0.01)
                        linthresh=0.01
    #                    xticks_major = [-0.1, 0, 0.1]
                        xticks_major = [field_min, -0.04, 0., 0.04, field_max]
    #                    xticks_major = [0, 0.01, 0.1]
                    elif ax_title == "n_r":
                        field_max = 10.
                        xticks_major = [0.01,0.1,1.,10.]
                        xticks_minor = np.concatenate((
                                np.linspace(2E-2,1E-1,9),
                                np.linspace(2E-1,1,9),
                                np.linspace(2,10,9),
                                ))
                else: norm_ = mpl.colors.Normalize   
                
                if ax_title == r"\Theta":
                    field_min = 289.2
                    field_max = 292.5
                    xticks_major = [290,291,292]
                if ax_title == "r_v":
                    field_min = 6.5
                    field_max = 7.6
                    xticks_minor = [6.75,7.25]
                if ax_title == "r_l":
                    field_min = 0.0
                    field_max = 1.3
                    xticks_minor = [0.25,0.75,1.25]
                    
                if ax_title == "r_c":
                    field_max = 0.1
                    field_min = -field_max                
    #                field_min = -0.2
    #                field_max = 0.2
                    xticks_major = [field_min, -0.05, 0., 0.05, field_max]
    #                xticks_major = np.linspace(0,1.2,7)
                if ax_title == "n_c":
                    field_min = 0.0
                    field_max = 150.
                if ax_title == "n_\mathrm{aero}":
                    field_max = 8.
                    field_min = -field_max
                    xticks_major = [field_min, -4, 0., 4, field_max]
    #                xticks_minor = [25,75,125]
                if ax_title in [r"R_\mathrm{avg}", r"R_{2/1}", r"R_\mathrm{eff}"]:
    #                xticks_major = [1,5,10,15,20]
    #                field_min = 0.
    #                field_min = 1.5
                    field_max = 3.
                    field_min = -field_max
                    xticks_major = np.linspace(field_min,field_max,7)
    #                cmap = cmap_new
                    # Arabas 2015
                    cmap = cmap_lcpp
                    unit = r"\si{\micro\meter}"
                    
            oom_max = oom = int(math.log10(field_max))
            
            my_format = False
            oom_factor = 1.0
            
#            if oom_max > 2 or oom_max < 0:
#                my_format = True
#                oom_factor = 10**(-oom)
#                
#                field_min *= oom_factor
#                field_max *= oom_factor            
            
            if oom_max ==2: str_format = "%.0f"
#            if oom_max ==2: str_format = "%.1f"
            
            else: str_format = "%.2g"
#            else: str_format = "%.2f"
            
            cmap = "bwr"
            
            # REMOVE FIX APLHA HERE
#            alpha = 1.0
#        CS = ax.pcolormesh(*grid.corners, grid_r_l,
#                           cmap=cmap, alpha=alpha,
#                            edgecolor="face", zorder=1,
#                            vmin=field_min, vmax=field_max,
#                            antialiased=True, linewidth=0.0
##                            norm = norm_(vmin=field_min, vmax=field_max)
#                            )          
            if False:
#            if ax_title in ["r_r", "n_r"]:
                norm = norm_(linthresh = linthresh, vmin=field_min, vmax=field_max)
            else:
                norm = norm_(vmin=field_min, vmax=field_max)
            CS = ax.pcolormesh(*grid.corners, field*oom_factor,
                               cmap=cmap, alpha=alpha,
                                edgecolor="face", zorder=1,
                                norm = norm,
                                rasterized=True,
                                antialiased=True, linewidth=0.0
                                )
            CS.cmap.set_under("blue")
            CS.cmap.set_over("red")
            
            ax.set_xticks( np.linspace( tick_ranges[0,0],
                                             tick_ranges[0,1],
                                             no_ticks[0] ) )
            ax.set_yticks( np.linspace( tick_ranges[1,0],
                                             tick_ranges[1,1],
                                             no_ticks[1] ) )

#            ax.set_xticks( np.linspace( tick_ranges[0,0],
#                                             tick_ranges[0,1],
#                                             no_ticks[0] ) )
#            ax.set_yticks( np.linspace( tick_ranges[1,0],
#                                             tick_ranges[1,1],
#                                             no_ticks[1] ) )
#            ax.tick_params(axis='both', which='major', labelsize=TKFS)
            ax.tick_params(axis='both', which='major', labelsize=TKFS,
                           length = 3, width=.6)
#            ax.tick_params(axis='both', which='minor', labelsize=TKFS,
#                           length = 3)            
            ax.grid(color='gray', linestyle='dashed', zorder = 2)
            ax.set_aspect('equal')
            if field_n == no_rows-1:
#                tlabels = ax.get_xticklabels()
#                print(tlabels)
#                tlabels[-1] = ""
#                print(tlabels)
#                ax.set_xticklabels(tlabels)    
                xticks1 = ax.xaxis.get_major_ticks()
                xticks1[-1].label1.set_visible(False)
                ax.set_xlabel(r'$x$ (km)', fontsize = LFS)
            if time_n == 0:            
                ax.set_ylabel(r'$z$ (km)', fontsize = LFS)
#            if time_n == 0:
#                ax.set_title(
#    r"\begin{{center}}${0}$ ({1})\\ t = 0\end{{center}}".format(ax_title, unit),
#                             fontsize = TTFS)
#                ax.set_title(
#    r"\begin{{center}}${0}$ ({1})\\ t = 0\end{{center}}".format(ax_title, unit),
#                             fontsize = TTFS)
#            else:
            if field_n == 0:                
                ax.set_title( r"$t$ = {0} min".format(int(save_times[time_n]/60)),
                             fontsize = TTFS)
#            ax.set_title( r"${0}$ ({1}), t = {2} min".format(ax_title, unit,
#                         int(save_times[time_n]/60)),
#                         fontsize = TTFS)
            if time_n == no_cols-1:
                
#            if time_n == no_rows - 1:
                
                axins = inset_axes(ax,
                                   width="10%",  # width = 5% of parent_bbox width
                                   height="100%",  # height
                                   loc='lower right',
                                   bbox_to_anchor=(0.14, 0.0, 1, 1),
#                                   bbox_to_anchor=(0.3, 0.0, 1, 1),
#                                   , 1, 1),
                                   bbox_transform=ax.transAxes,
                                   borderpad=0,
                                   )      
#                else:
#                    axins = inset_axes(ax,
#                                       width="90%",  # width = 5% of parent_bbox width
#                                       height="8%",  # height
#                                       loc='lower center',
#                                       bbox_to_anchor=(0.0, 1.4, 1, 1),
#    #                                   , 1, 1),
#                                       bbox_transform=ax.transAxes,
#                                       borderpad=0,
#                                       )      
#                divider = make_axes_locatable(ax)
#                cax = divider.append_axes("top", size="6%", pad=0.3)
                
                cbar = plt.colorbar(CS, cax=axins,
#                                    fraction=0.046, pad=-0.1,
                                    format=mticker.FormatStrFormatter(str_format),
                                    orientation="vertical"
                                    )
                
#                axins.yaxis.set_ticks_position("left")
                
#                axins.xaxis.set_ticks_position("top")
                axins.tick_params(axis="y",direction="inout",which="both")
#                axins.tick_params(axis="x",direction="inout")
                axins.tick_params(axis='y', which='major', labelsize=TKFS,
                               length = 4, width=0.8)                
                axins.tick_params(axis='y', which='minor', labelsize=TKFS,
                               length = 3, width=0.5,bottom=True)                
                
                
                if xticks_major is not None:
                    axins.yaxis.set_ticks(xticks_major)
                if xticks_minor is not None:
                    axins.yaxis.set_ticks(xticks_minor, minor=True)
                if ax_title == "n_c":
                    xticks2 = axins.xaxis.get_major_ticks()
                    xticks2[-1].label1.set_visible(False)                
#                axins.set_ylabel(r"${0}$ ({1})".format(ax_title, unit))
                
                if title_add is None:
                    cbar.set_label(r"$\Delta {0}$ ({1})".format(ax_title, unit),
                                   rotation=90)
                elif title_add2 is None:
                    cbar.set_label(r"${0}$ ({1})".format(ax_title, unit)
                                   +"\n"
                                   +r"(${0}$)".format(title_add),
                                   rotation=0,
                                   labelpad=23,
                                   y = 0.65)
                else:
                    cbar.set_label(r"${0}$ ({1})".format(ax_title, unit)
                                   +"\n"
                                   +r"(${0}$)".format(title_add)
                                   +"\n"
                                   +r"(${0}$)".format(title_add2),
                                   rotation=0,
                                   labelpad=23,
                                   y = 0.7)
                    
#                    cbar.axis.set_label_coords(0.05, 0.75)
            # my_format dos not work with log scale here!!

                if my_format:
                    cbar.ax.text(field_min - (field_max-field_min),
                                 field_max + (field_max-field_min)*0.01,
                                 r'$\times\,10^{{{}}}$'.format(oom_max),
                                 va='bottom', ha='left', fontsize = TKFS)
                cbar.ax.tick_params(labelsize=TKFS)                
                
            if show_target_cells:
                ### ad the target cells
                no_neigh_x = no_cells_x // 2
                no_neigh_z = no_cells_z // 2
                dx = grid.steps[0]
                dz = grid.steps[1]
                
                no_tg_cells = len(target_cell_list[0])
                LW_rect = .5
                for tg_cell_n in range(no_tg_cells):
                    x = (target_cell_list[0, tg_cell_n] - no_neigh_x - 0.1) * dx
                    z = (target_cell_list[1, tg_cell_n] - no_neigh_z - 0.1) * dz
                    
            #        dx *= no_cells_x
            #        dz *= no_cells_z
                    
                    rect = plt.Rectangle((x, z), dx*no_cells_x,dz*no_cells_z,
                                         fill=False,
                                         linewidth = LW_rect,
        #                                 linestyle = "dashed",
                                         edgecolor='k',
                                         zorder = 99)        
                    ax.add_patch(rect)


    pad_ax_h = -0.3
    pad_ax_v = 0.08
#    pad_ax_v = 0.005
    fig.subplots_adjust(hspace=pad_ax_h, wspace=pad_ax_v)
#    fig.subplots_adjust(wspace=pad_ax_v)
             
#    fig.tight_layout()
    if fig_path is not None:
        fig.savefig(fig_path,
    #                    bbox_inches = 0,
                    bbox_inches = 'tight',
                    pad_inches = 0.05,
                    dpi=600
                    )     
    plt.close("all")

    
#%%

    #######################################################################
    ### PLOT ABS ERROR OF THE DIFFERENCE OF TWO FIELDS:
#    # Var = Var_1 + Var_2 (assuming no correlations)
    # 
#    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
#    for i,fm in enumerate(field_names):
#        print(i,fm)
#    print(save_times)
    
#            fields_with_time[time_n, field_n] *= scales[field_n]
#            rel_std[time_n, field_n] = \
#                np.where(fields_with_time[time_n, field_n] <= 0.,
#                         0.,
#                         std[time_n, field_n] / fields_with_time[time_n, field_n])
#    print("plotting ABS ERROR of field1 - field2")
#    
#    std1 = np.nan_to_num( fields_with_time_std1 )
#    std2 = np.nan_to_num( fields_with_time_std2 )
#    std = np.sqrt( std1**2 + std2**2 )
##    rel_std = fields_with_time_std
##    rel_std = np.zeros_like(std)
#    
##    rel_std = np.where(fields_with_time == 0.,
##                       0.,
##                       fields_with_time_std / fields_with_time)
#    
#    tick_ranges = grid.ranges
##    tick_ranges_label = [[0,1.2],[0,1.2]]
#    
#    no_rows = len(save_times)
#    no_cols = len(field_names)
#
##    fig, axes = plt.subplots(nrows=no_rows, ncols=no_cols,
##                       figsize = (4.5*no_cols, 4*no_rows),
##                       sharex=True, sharey=True)    
#    
#    for time_n in range(no_rows):
#        for field_n in range(no_cols):
#            std[time_n, field_n] *= scales[field_n]
    
#    field_max_all = np.amax(std, axis=(0,2,3))
#    field_min_all = np.amin(std, axis=(0,2,3))
##    field_max_all_std = np.amax(rel_std, axis=(0,2,3))
##    field_min_all_std = np.amin(rel_std, axis=(0,2,3))
#
#    pad_ax_h = 0.1     
#    pad_ax_v = 0.03
    ### PLOT ABS ERROR  
#    if plot_abs:
#    fig2, axes = plt.subplots(nrows=no_rows, ncols=no_cols,
#                       figsize = figsize,
#                       sharex=True, sharey=True)
#            
#    for time_n in range(no_rows):
#        for field_n in range(no_cols):
#            ax = axes[time_n,field_n]
##            field = rel_std[time_n, field_n] * scales[field_n]
#            field = std[time_n, field_n]
#            ax_title = field_names[field_n]
#            unit = units[field_n]
#            
##            cmap = "coolwarm"
##            cmap = "Greys"
##            cmap = "Greens"
##            cmap = "Reds"
#            cmap = cmap_lcpp
#            
#            field_min = 0.
#            field_max = field_max_all[field_n]
##            field_min = field_min_all[field_n]
#
#            xticks_major = None
#            xticks_minor = None
#            
#            if compare_type == "Nsip":
#                if ax_title == "n_\mathrm{aero}":
##                    field_max = 5.
##                    xticks_minor = np.linspace(1.25,5,3)
#                    xticks_minor = np.array((1.25,3.75))
#                if ax_title == "r_c":
#                    field_max = 0.06
#                    xticks_minor = np.linspace(0.01,0.05,3)
#                if ax_title == "r_r":
##                    field_max = 0.04
#                    xticks_minor = np.linspace(0.01,0.03,2)
##                        xticks_minor = np.linspace(1,5,3)
#                #, r"R_{2/1}", r"R_\mathrm{eff}"]:                    
#                if ax_title in [r"R_\mathrm{eff}"]:
#                    xticks_major = np.linspace(0.,1.5,4)
#            
#            if compare_type == "dt_col":
#                if ax_title == "n_\mathrm{aero}":
##                    field_max = 5.
##                    xticks_minor = np.linspace(1.25,5,3)
#                    xticks_minor = np.array((1.25, 3.75, 6.25))
#                if ax_title == "r_c":
##                    field_max = 0.0
##                    xticks_minor = np.linspace(0.01,0.05,3)
#                    xticks_major = np.array((0.,0.02,0.04,0.06))
#                if ax_title == "r_r":
##                    field_max = 0.04
#                    xticks_minor = np.linspace(0.01,0.05,3)
##                        xticks_minor = np.linspace(1,5,3)
#                #, r"R_{2/1}", r"R_\mathrm{eff}"]:                    
#                if ax_title in [r"R_\mathrm{eff}"]:
#                    xticks_major = np.linspace(0.,1.5,4)
#            
##            if SIM_N == 2:
##                if ax_title == "r_r":
##                    field_max = 0.03
###                        xticks_minor = np.linspace(0.01,0.03,2)
###                        xticks_minor = np.linspace(1,5,3)
##                if ax_title == "r_c":
###                        field_max = 0.05
##                    xticks_minor = np.linspace(0.01,0.03,2)
##                if ax_title == "n_\mathrm{aero}":
###                        pass
###                        field_max = 5.
##                    xticks_minor = np.linspace(1,3,2)
##            if SIM_N == 10:
##                if ax_title == "r_r":
###                        field_max = 0.04
##                    xticks_minor = np.linspace(0.01,0.03,2)
###                        xticks_minor = np.linspace(1,5,3)
##                if ax_title == "r_c":
##                    field_max = 0.05
##                    xticks_minor = np.linspace(0.01,0.05,3)
##                if ax_title == "n_\mathrm{aero}":
##                    field_max = 5.
##                    xticks_minor = np.linspace(1,5,3)
#            
##            field_max = field.max()
##            field_min = field.min()
#            
#            norm_ = mpl.colors.Normalize 
#                
#            str_format = "%.2g"
#            
#            my_format = False
#            oom_factor = 1.0
#
#            oom_max = oom = 1
#
#            CS = ax.pcolormesh(*grid.corners,
#                               field*oom_factor,
#                               cmap=cmap, alpha=alpha,
#                                edgecolor="face", zorder=1,
##                                norm = norm_(vmin=-minmax, vmax=minmax),
#                                norm = norm_(vmin=field_min, vmax=field_max),
##                                norm = norm_(vmin=field_min, vmax=field_max),
#                                rasterized=True,
#                                antialiased=True, linewidth=0.0
#                                )
#            CS.cmap.set_under("white")
#            
#            ax.set_xticks( np.linspace( tick_ranges[0,0],
#                                             tick_ranges[0,1],
#                                             no_ticks[0] ) )
#            ax.set_yticks( np.linspace( tick_ranges[1,0],
#                                             tick_ranges[1,1],
#                                             no_ticks[1] ) )
#
#            ax.tick_params(axis='both', which='major', labelsize=TKFS,
#                           length = 3, width=1)
#            ax.grid(color='gray', linestyle='dashed', zorder = 2)
#            ax.set_aspect('equal')
#            if time_n == no_rows-1:
#                xticks1 = ax.xaxis.get_major_ticks()
#                xticks1[-1].label1.set_visible(False)
#                ax.set_xlabel(r'$x$ (km)', fontsize = LFS)
#            if field_n == 0:            
#                ax.set_ylabel(r'$z$ (km)', fontsize = LFS)
#            ax.set_title( r"$t$ = {0} min".format(int(save_times[time_n]/60)),
#                         fontsize = TTFS)
#            if time_n == 0:
#                axins = inset_axes(ax,
#                                   width="90%",  # width = 5% of parent_bbox width
#                                   height="8%",  # height
#                                   loc='lower center',
#                                   bbox_to_anchor=(0.0, 1.35, 1, 1),
##                                   , 1, 1),
#                                   bbox_transform=ax.transAxes,
#                                   borderpad=0,
#                                   )      
#                cbar = plt.colorbar(CS, cax=axins,
##                                    fraction=0.046, pad=-0.1,
#                                    format=mticker.FormatStrFormatter(str_format),
#                                    orientation="horizontal"
#                                    )
#                axins.xaxis.set_ticks_position("bottom")
#                axins.tick_params(axis="x",direction="inout",which="both")
#                axins.tick_params(axis='x', which='major', labelsize=TKFS,
#                               length = 7, width=1)                
#                axins.tick_params(axis='x', which='minor', labelsize=TKFS,
#                               length = 5, width=0.5,bottom=True)                
#                
#                if xticks_major is not None:
#                    axins.xaxis.set_ticks(xticks_major)
#                if xticks_minor is not None:
#                    axins.xaxis.set_ticks(xticks_minor, minor=True)
#                if ax_title in [r"R_\mathrm{avg}", r"R_{2/1}", r"R_\mathrm{eff}"]:
#                    unit = r"\si{\micro\meter}"                                            
#                axins.set_title(r"${0}$ abs. error ({1})".format(ax_title, unit))
#            # my_format dos not work with log scale here!!
#                
#                if my_format:
#                    cbar.ax.text(1.0,1.0,
#                                 r'$\times\,10^{{{}}}$'.format(oom_max),
#                                 va='bottom', ha='right', fontsize = TKFS,
#                                 transform=ax.transAxes)
##                    cbar.ax.text(field_min - (field_max-field_min),
##                                 field_max + (field_max-field_min)*0.01,
##                                 r'$\times\,10^{{{}}}$'.format(oom_max),
##                                 va='bottom', ha='left', fontsize = TKFS,
##                                 transform=ax.transAxes)
#                cbar.ax.tick_params(labelsize=TKFS)
#
#            if show_target_cells:
#                ### ad the target cells
#                no_neigh_x = no_cells_x // 2
#                no_neigh_z = no_cells_z // 2
#                dx = grid.steps[0]
#                dz = grid.steps[1]
#                
#                no_tg_cells = len(target_cell_list[0])
#                LW_rect = .5
#                for tg_cell_n in range(no_tg_cells):
#                    x = (target_cell_list[0, tg_cell_n] - no_neigh_x - 0.1) * dx
#                    z = (target_cell_list[1, tg_cell_n] - no_neigh_z - 0.1) * dz
#                    
#            #        dx *= no_cells_x
#            #        dz *= no_cells_z
#                    
#                    rect = plt.Rectangle((x, z), dx*no_cells_x,dz*no_cells_z,
#                                         fill=False,
#                                         linewidth = LW_rect,
#        #                                 linestyle = "dashed",
#                                         edgecolor='k',
#                                         zorder = 99)        
#                    ax.add_patch(rect)
#    
#    
#    
#    #    pad_ax_v = 0.005
#    fig2.subplots_adjust(hspace=pad_ax_h, wspace=pad_ax_v)
##    fig.subplots_adjust(wspace=pad_ax_v)
#             
##    fig.tight_layout()
#    if fig_path_abs_err is not None:
#        fig2.savefig(fig_path_abs_err,
#    #                    bbox_inches = 0,
#                    bbox_inches = 'tight',
#                    pad_inches = 0.05,
#                    dpi=600
#                    )               
    
#%%    