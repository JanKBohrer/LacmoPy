#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 14:42:55 2019

@author: jdesk
"""

### PLOTTING

import numpy as np
import matplotlib.pyplot as plt

def simple_plot(x_, y_arr_):
    fig = plt.figure()
    ax = plt.gca()
    for y_ in y_arr_:
        ax.plot (x_, y_)
    ax.grid()

# INWORK: add title and ax labels
def plot_scalar_field_2D( grid_centers_x_, grid_centers_y_, field_,
                         tick_ranges_, no_ticks_=[5,5],
                         no_contour_colors_ = 10, no_contour_lines_ = 5,
                         colorbar_fraction_=0.046, colorbar_pad_ = 0.02):
    fig, ax = plt.subplots(figsize=(8,8))

    contours = plt.contour(grid_centers_x_, grid_centers_y_,
                           field_, no_contour_lines_, colors = 'black')
    ax.clabel(contours, inline=True, fontsize=8)
    CS = ax.contourf( grid_centers_x_, grid_centers_y_,
                     field_,
                     levels = no_contour_colors_,
                     vmax = field_.max(),
                     vmin = field_.min(),
                    cmap = plt.cm.coolwarm)
    ax.set_xticks( np.linspace( tick_ranges_[0,0], tick_ranges_[0,1],
                                no_ticks_[0] ) )
    ax.set_yticks( np.linspace( tick_ranges_[1,0], tick_ranges_[1,1],
                                no_ticks_[1] ) )
    plt.colorbar(CS, fraction=colorbar_fraction_ , pad=colorbar_pad_)

from evaluation import sample_masses, sample_radii
    
def plot_particle_size_spectra(m_w, m_s, xi, cells, grid,
                          target_cell, no_cells_x, no_cells_z,
                          no_rows=1, no_cols=1, TTFS=12, LFS=10, TKFS=10,
                          fig_path = None):
    R, Rs, multi = sample_radii(m_w, m_s, xi, cells, grid.temperature,
                                    target_cell, no_cells_x, no_cells_z)

    log_R = np.log10(R)
    log_Rs = np.log10(Rs)
    multi = np.array(multi)
    if no_cells_x % 2 == 0: no_cells_x += 1
    if no_cells_z % 2 == 0: no_cells_z += 1
    
    V_an = no_cells_x * no_cells_z * grid.volume_cell * 1.0E6
    # V_an = (no_neighbors_x * 2 + 1) * (no_neighbors_z * 2 + 1) * grid.volume_cell * 1.0E6
    
    no_bins = 40
    no_bins_s = 30
    bins = np.empty(no_bins)
    
    R_min = 1E-2
    Rs_max = 0.3
    # R_max = 12.0
    R_max = 120.0
    
    R_min_log = np.log10(R_min)
    Rs_max_log = np.log10(Rs_max)
    R_max_log = np.log10(R_max)
    
    # print(log_R.shape)
    # print(log_Rs.shape)
    # print(multi.shape)
    
    h1, bins1 = np.histogram( log_R, bins=no_bins,  weights=multi/V_an )
    h2, bins2 = np.histogram( log_Rs, bins=no_bins_s,  weights=multi/V_an )
    # h1, bins1 = np.histogram( log_R, bins=no_bins, range=(R_min_log,R_max_log), weights=multi/V_an )
    # h2, bins2 = np.histogram( log_Rs, bins=no_bins_s, range=(R_min_log, Rs_max_log), weights=multi/V_an )
    bins1 = 10 ** bins1
    bins2 = 10 ** bins2
    d_bins1 = np.diff(bins1)
    d_bins2 = np.diff(bins2)
    
    #########################
    
    # # title size (pt)
    # TTFS = 22
    # # labelsize (pt)
    # LFS = 20
    # # ticksize (pt)
    # TKFS = 18
    
    # figsize_x = cm2inch(10.8)
    # figsize_y = cm2inch(10.3)
    # figsize_y = 8
    
    # no_rows = 1
    # no_cols = 1
    
    def cm2inch(*tupl):
        inch = 2.54
        if isinstance(tupl[0], tuple):
            return tuple(i/inch for i in tupl[0])
        else:
            return tuple(i/inch for i in tupl)
    
    fig, axes = plt.subplots(nrows = no_rows, ncols = no_cols,
                           figsize = cm2inch(10.8,9.3),
                           # figsize = (figsize_x * no_cols, figsize_y * no_rows),
    #                        sharey=True,
    #                        sharex=True,
                             )
    ax = axes
    # ax.hist(R, bins = bins1, weights = multi/d_bins1)
    ax.bar(bins1[:-1], h1, width = d_bins1, align = 'edge',
    # ax.bar(bins1[:-1], h1/d_bins1, width = d_bins1, align = 'edge',
    #        fill = False,
    #        color = None,
            alpha = 0.05,
           linewidth = 0,
    #        color = (0,0,1,0.05),
    #        edgecolor = (0,0,0,0.0),
           
          )
    ax.bar(bins2[:-1], h2, width = d_bins2, align = 'edge', 
    # ax.bar(bins2[:-1], h2/d_bins2, width = d_bins2, align = 'edge',
    #        fill = False,
    #        color = None,
            alpha = 0.1,
           linewidth = 0,
    #        color = (1,0,0,0.05),
    #        edgecolor = (1,0,0,1.0),
          )
    LW = 2
    ax.plot(np.repeat(bins1,2)[:], np.hstack( [[0.001], np.repeat(h1,2)[:], [0.001] ] ),
            linewidth = LW, zorder = 6, label = "wet")
    ax.plot(np.repeat(bins2,2)[:], np.hstack( [[0.001], np.repeat(h2,2)[:], [0.001] ] ), linewidth = LW,
           label = "dry")
    
    
    ax.tick_params(axis='both', which='major', labelsize=TKFS, length = 5)
    ax.tick_params(axis='both', which='minor', labelsize=TKFS, length = 3)
    
    height = int(grid.compute_location(*target_cell,0.0,0.0)[1])
    ax.set_xlabel(r"particle radius (mu)", fontsize = LFS)
    ax.set_ylabel(r"concentration (${\mathrm{cm}^{3}}$)", fontsize = LFS)
    # ax.set_xlabel(r"particle radius ($\si{\micro m}$)", fontsize = LFS)
    # ax.set_ylabel(r"concentration ($\si{\# / cm^{3}}$)", fontsize = LFS)
    ax.set_title( f'height = {height} m', fontsize = TTFS )
    
    # X = np.linspace(1E-2,1.0,1000)
    # Y = np.log(X)
    # Z = gaussian(Y, np.log(0.075), np.log(1.6))
    
    # ax.plot(X,Z*11,'--',c="k")
    
    # ax.set_xticks(bins1)
    ax.set_xscale("log")
    # ax.set_xticks(bins1)
    ax.set_yscale("log")
    ax.set_ylim( [0.01,50] )
    ax.grid(linestyle="dashed")
    
    handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles[::-1], labels[::-1], title='Line', loc='upper left')
    ax.legend(handles[::-1], labels[::-1], ncol = 2, prop={'size': TKFS},
              loc='upper center', bbox_to_anchor=(0.5, 1.05), frameon = False)
    fig.tight_layout()
    plt.show()
    if fig_path is not None:
        fig.savefig(fig_path + "spectrum_" + str(height) +".pdf")

    
    