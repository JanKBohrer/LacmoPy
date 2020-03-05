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

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import hex2color, LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#%% BASIC FUNCTIONS

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)
    
#%% CUSTOM COLOR MAPS

colors1 = plt.cm.get_cmap('gist_ncar_r', 256)
colors2 = plt.cm.get_cmap('rainbow', 256)

newcolors = np.vstack((colors1(np.linspace(0, 0.16, 24)),
                       colors2(np.linspace(0, 1, 256))))
cmap_my_rainbow = mpl.colors.ListedColormap(newcolors, name='my_rainbow')

### colormap 'lcpp' like Arabas 2015
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

cdict_lcpp = {'red': cdict_lcpp_colors[0],
              'green': cdict_lcpp_colors[1],
              'blue': cdict_lcpp_colors[2]}

cmap_lcpp = LinearSegmentedColormap('myRainbow', segmentdata=cdict_lcpp, N=256)

#%% rcParams DICTIONARIES
    
pdf_dict = {
#    'backend' : 'pgf',    
    'text.usetex': True,
#    'pgf.rcfonts': False,   # Do not set up fonts from rc parameters.
#    'pgf.texsystem': 'lualatex',
#    'pgf.texsystem': 'pdflatex',
    'text.latex.preamble': [
        r'usepackage[ttscale=.9]{libertine}',
        r'usepackage[libertine]{newtxmath}',
        r'usepackage[T1]{fontenc}',
        r'usepackage[]{siunitx}',        
#        r'usepackage[no-math]{fontspec}',
        ],
    'font.family': 'serif'
}
pgf_dict = {
#    'backend' : 'pgf',    
    'text.usetex': True,
#    'pgf.rcfonts': False,   # Do not set up fonts from rc parameters.
#    'pgf.texsystem': 'lualatex',
    'pgf.texsystem': 'pdflatex',
    'pgf.preamble': [
        r'usepackage[ttscale=.9]{libertine}',
        r'usepackage[libertine]{newtxmath}',
        r'usepackage[T1]{fontenc}',
        r'usepackage[]{siunitx}',
#        r'usepackage[no-math]{fontspec}',
        ],
    'font.family': 'serif'
}
    
pgf_dict2 = {
#    'backend' : 'pgf',    
    'text.usetex': True,
#    'pgf.rcfonts': False,   # Do not set up fonts from rc parameters.
#    'pgf.texsystem': 'lualatex',
    'pgf.texsystem': 'pdflatex',
    'pgf.preamble': [
        r'usepackage[ttscale=.9]{libertine}',
        r'usepackage[libertine]{newtxmath}',
        r'usepackage[T1]{fontenc}',
        r'usepackage[]{siunitx}',
        r'usepackage[]{sfmath}',
#        r'usepackage[no-math]{fontspec}',
        ],
    'font.family': 'sans'
}

def generate_rcParams_dict(LW, MS, TTFS, LFS, TKFS, DPI):
    dict_ = {'lines.linewidth' : LW,
             'lines.markersize' : MS,
             'axes.titlesize' : TTFS,
             'axes.labelsize' : LFS,
             'legend.fontsize' : LFS-2,
             'xtick.labelsize' : TKFS,
             'ytick.labelsize' : TKFS,
             #{'center', 'top', 'bottom', 'baseline', 'center_baseline'}
             #center_baseline seems to be def, center is OK
             'xtick.alignment' : 'center',
             'ytick.alignment' : 'center',
             'savefig.dpi' : DPI
             }
    
    return dict_

#%% SIMPLE SCALAR FIELD TEMPLATE

def simple_plot(x_, y_arr_):
    fig = plt.figure()
    ax = plt.gca()
    for y_ in y_arr_:
        ax.plot (x_, y_)
    ax.grid()

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
    
#%% FUNCTION DEF SIZE SPECTRA

def plot_size_spectra_vs_R(f_R_p_list, f_R_s_list,
                           bins_R_p_list, bins_R_s_list,
                           grid_r_l_list,
                           R_min_list, R_max_list,
                           save_times_out,
                           solute_type,
                           grid,                               
                           target_cell_list,
                           no_cells_x, no_cells_z,
                           no_bins_R_p, no_bins_R_s,
                           no_rows, no_cols,
                           SIM_N = None,
                           TTFS=12, LFS=10, TKFS=10, LW = 4.0, MS = 3.0,
                           figsize_spectra = None,
                           figsize_trace_traj = None,
                           fig_path = None,
                           show_target_cells = False,
                           fig_path_tg_cells = None,
                           fig_path_R_eff = None,
                           trajectory = None,
                           show_textbox = True,
                           xshift_textbox = 0.5,
                           yshift_textbox = -0.35
                           ):
    scale_x = 1E-3
    
#    annotations = ['A', 'B', 'C', 'D', 'E', 'F',
#                   'G', 'H', 'I', 'J', 'K', 'L',
#                   'M', 'N', 'O', 'P', 'Q', 'R']
    annotations = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    
    no_seeds = len(f_R_p_list[0])
    print('no_seeds =', no_seeds)
    grid_steps = grid.steps
    no_tg_cells = len(target_cell_list[0])    
    
    f_R_p_avg = np.average(f_R_p_list, axis=1)
    f_R_s_avg = np.average(f_R_s_list, axis=1)
    
    if f_R_p_list.shape[1] != 1:
        f_R_p_std = np.std(f_R_p_list, axis=1, ddof=1) / np.sqrt(no_seeds)
        f_R_s_std = np.std(f_R_s_list, axis=1, ddof=1) / np.sqrt(no_seeds)

    if figsize_spectra is not None:
        figsize = figsize_spectra
    else:        
        figsize = (no_cols*5, no_rows*4)

    fig, axes = plt.subplots(nrows = no_rows, ncols = no_cols,
                             figsize = figsize, sharex = True, sharey=True )
    
    plot_n = -1
    for row_n in range(no_rows):
        for col_n in range(no_cols):
            plot_n += 1
            if no_cols == 1:
                if no_rows == 1:
                    ax = axes
                else:                    
                    ax = axes[row_n]
            else:
                ax = axes[row_n, col_n]        
            
            target_cell = target_cell_list[:,plot_n]
            
            f_R_p = f_R_p_avg[plot_n]
            f_R_s = f_R_s_avg[plot_n]
            
            if f_R_p_list.shape[1] != 1:
                f_R_p_err = f_R_p_std[plot_n]
                f_R_s_err = f_R_s_std[plot_n]
            
            bins_R_p = bins_R_p_list[plot_n]
            bins_R_s = bins_R_s_list[plot_n]
            
            f_R_p_min = f_R_p.min()
            f_R_s_min = f_R_s.min()
            
            LW_spectra = 1.
            
            ax.plot(np.repeat(bins_R_p,2),
                    np.hstack( [[f_R_p_min*1E-1],
                                np.repeat(f_R_p,2),
                                [f_R_p_min*1E-1] ] ),
                    linewidth = LW_spectra, label = 'wet')
            ax.plot(np.repeat(bins_R_s,2),
                    np.hstack( [[f_R_s_min*1E-1],
                                np.repeat(f_R_s,2),
                                [f_R_s_min*1E-1] ] ),
                    linewidth = LW_spectra, label = 'dry')  
            
            # added error-regions here
            if f_R_p_list.shape[1] != 1:
                ax.fill_between(np.repeat(bins_R_p,2)[1:-1],
                                np.repeat(f_R_p,2) - np.repeat(f_R_p_err,2),
                                np.repeat(f_R_p,2) + np.repeat(f_R_p_err,2),
                                alpha=0.5,
                                facecolor='lightblue',
                                edgecolor='blue', lw=0.5,
                                zorder=0)
                ax.fill_between(np.repeat(bins_R_s,2)[1:-1],
                                np.repeat(f_R_s,2) - np.repeat(f_R_s_err,2),
                                np.repeat(f_R_s,2) + np.repeat(f_R_s_err,2),
                                alpha=0.4,
                                facecolor='orange',
                                edgecolor='darkorange', lw=0.5,
                                zorder=0)
            AN_pos = (2.5E-3, 2E3)
            ax.annotate(f'\textbf{{{annotations[plot_n]}}}',
                    AN_pos
                    )
            LW_vline = 1
            LS_vline = '--'
            ax.axvline(0.5, c ='k', linewidth=LW_vline,ls=LS_vline)
            ax.axvline(25., c ='k', linewidth=LW_vline, ls = LS_vline)            
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_yticks(np.logspace(-2,4,7))
            
            ax.set_xticks(np.logspace(-2,2,5))
            ax.set_xticks(
                    np.concatenate((
                            np.linspace(2E-3,9E-3,8),
                            np.linspace(1E-2,9E-2,9),
                            np.linspace(1E-1,9E-1,9),
                            np.linspace(1,9,9),
                            np.linspace(10,90,9),
                            np.linspace(1E2,3E2,3),
                            )),
                    minor=True
                    )
                    
            ax.set_xlim( [2E-3, 1E2] )  
            
            ax.set_ylim( [5E-3, 1E4] )    
            
            ax.tick_params(axis='both', which='major', labelsize=TKFS,
                           length = 6, width=1)
            ax.tick_params(axis='both', which='minor', labelsize=TKFS,
                           length = 3)
            
            xx = ( (target_cell[0] + 0.5 )*grid_steps[0])
            height = ( (target_cell[1] + 0.5 )*grid_steps[1])
            if row_n==no_rows-1:
                ax.set_xlabel(r'$R_p$ (si{micrometer})',
                              fontsize = LFS)
            if col_n==0:
                ax.set_ylabel('$f_R$ $(\si{\micro\meter^{-1}\,mg^{-1}})$',
                              fontsize = LFS)
                
            if trajectory is None:            
                ax.set_title( f'$(x,z)=$ ({xx:.2}, {height:.2}) km, ', 
                              fontsize = TTFS )            
            else:
                ax.set_title( f'$(x,z)=$ ({xx:.2}, {height:.2}), ' 
                              + f't = {save_times_out[plot_n]//60-120} min',
                              fontsize = TTFS )            
            ax.grid()
            ax.legend(
                       loc = 'upper right',
                       ncol = 2,
                       handlelength=1, handletextpad=0.3,
                      columnspacing=0.8, borderpad=0.25)            
    pad_ax_h = 0.35          
    if trajectory is None:
        pad_ax_v = 0.08
        fig.subplots_adjust(hspace=pad_ax_h, wspace=pad_ax_v)
    else:        
        fig.subplots_adjust(hspace=pad_ax_h)
    if fig_path is not None:
        fig.savefig(fig_path[:-4] + '_ext.pdf',
                    bbox_inches = 'tight',
                    pad_inches = 0.05
                    )        
    # calc effective radius = moment3/moment2 from analysis of f_r
    R_eff_list = np.zeros((no_tg_cells, 3), dtype = np.float64)
    fig3, axes = plt.subplots(nrows = 1, ncols = 1,
                 figsize = (6, 5) )
    for tg_cell_n in range(no_tg_cells):
        bins_log = np.log(bins_R_p_list[tg_cell_n])
        bins_width = bins_R_p_list[tg_cell_n,1:] - bins_R_p_list[tg_cell_n,:-1]
        bins_center_log = 0.5 * (bins_log[1:] + bins_log[:-1])
        bins_center_log_lin = np.exp(bins_center_log)
        
        mom0 = np.sum( f_R_p_list[tg_cell_n] * bins_width)
        mom1 = np.sum( f_R_p_list[tg_cell_n] * bins_width
                       * bins_center_log_lin)
        mom2 = np.sum( f_R_p_list[tg_cell_n] * bins_width
                       * bins_center_log_lin * bins_center_log_lin)
        mom3 = np.sum( f_R_p_list[tg_cell_n] * bins_width
                       * bins_center_log_lin**3)

        R_eff_list[tg_cell_n,0] = mom1/mom0
        R_eff_list[tg_cell_n,1] = mom2/mom1
        R_eff_list[tg_cell_n,2] = mom3/mom2
    
    for mom_n in range(3):        
        axes.plot( np.arange(1,no_tg_cells+1), R_eff_list[:,mom_n], 'o',
                  label = f'mom {mom_n}')
        if mom_n == 2:
            for tg_cell_n in range(no_tg_cells):
                axes.annotate(f'({target_cell_list[0,tg_cell_n]} '
                              + f' {target_cell_list[1,tg_cell_n]})',
                              (tg_cell_n+1, 3. + R_eff_list[tg_cell_n,mom_n]),
                              fontsize=8)
    axes.legend()        
    if fig_path_R_eff is not None:
        fig3.savefig(fig_path_R_eff)            

    ### plot target cells in extra plot
    if show_target_cells:
        grid_r_l = grid_r_l_list[1]*1E3
        cmap = cmap_lcpp
        alpha = 1.0
        no_ticks = [6,6]
        str_format = '%.2f'
        
        tick_ranges = grid.ranges
        
        if figsize_trace_traj is not None:
            figsize = figsize_trace_traj
        else:
            figsize = (9, 8)            
        
        fig2, axes = plt.subplots(nrows = 1, ncols = 1,
                         figsize = figsize )
        ax = axes
        
        field_min = 0.0
        field_max = grid_r_l.max()
        CS = ax.pcolormesh(*grid.corners, grid_r_l,
                           cmap=cmap, alpha=alpha,
                            edgecolor='face', zorder=1,
                            vmin=field_min, vmax=field_max,
                            antialiased=True, linewidth=0.0
                            )
        CS.cmap.set_under('white')
        
        ax.set_xticks( np.linspace( tick_ranges[0,0],
                                         tick_ranges[0,1],
                                         no_ticks[0] ) )
        ax.set_yticks( np.linspace( tick_ranges[1,0],
                                         tick_ranges[1,1],
                                         no_ticks[1] ) )
        cbar = plt.colorbar(CS, ax=ax, fraction =.045,
                            format=mticker.FormatStrFormatter(str_format))    
        
        ### add vel. field
        ARROW_WIDTH= 0.004
        ARROW_SCALE = 20.0
        no_arrows_u=16
        no_arrows_v=16
        if no_arrows_u < grid.no_cells[0]:
            arrow_every_x = grid.no_cells[0] // (no_arrows_u - 1)
        else:
            arrow_every_x = 1

        if no_arrows_v < grid.no_cells[1]:
            arrow_every_y = grid.no_cells[1] // (no_arrows_v - 1)
        else:
            arrow_every_y = 1
        centered_u_field = ( grid.velocity[0][0:-1,0:-1]\
                             + grid.velocity[0][1:,0:-1] ) * 0.5
        centered_w_field = ( grid.velocity[1][0:-1,0:-1]\
                             + grid.velocity[1][0:-1,1:] ) * 0.5
        ax.quiver(
            grid.centers[0][arrow_every_y//2::arrow_every_y,
                        arrow_every_x//2::arrow_every_x],
            grid.centers[1][arrow_every_y//2::arrow_every_y,
                        arrow_every_x//2::arrow_every_x],
            centered_u_field[::arrow_every_y,::arrow_every_x],
            centered_w_field[::arrow_every_y,::arrow_every_x],
                  pivot = 'mid',
                  width = ARROW_WIDTH, scale = ARROW_SCALE, zorder=3 )                            
        
        ### ad the target cells
        no_neigh_x = no_cells_x // 2
        no_neigh_z = no_cells_z // 2
        dx = grid_steps[0]
        dz = grid_steps[1]
        
        LW_rect = 0.8
        
        textbox = []
        for tg_cell_n in range(no_tg_cells):
            x = (target_cell_list[0, tg_cell_n] - no_neigh_x - 0.1) * dx
            z = (target_cell_list[1, tg_cell_n] - no_neigh_z - 0.1) * dz
            
            print('x,z tg cells')
            print(x,z)
            
            rect = plt.Rectangle((x, z), dx*no_cells_x,dz*no_cells_z,
                                 fill=False,
                                 linewidth = LW_rect,
                                 edgecolor='k',
                                 zorder = 99)        
            ax.add_patch(rect)
            if trajectory is None:
                x_ann_shift = 80E-3
                z_ann_shift = 30E-3
                # yields col number
                if (tg_cell_n % no_cols) == 2:
                    x_ann_shift = -80E-3
                    
                # yields row number
                if (tg_cell_n // no_cols) == 1:
                    z_ann_shift = -40E-3
                if (tg_cell_n // no_cols) == 2:
                    z_ann_shift = -20E-3
                if (tg_cell_n // no_cols) == 3:
                    z_ann_shift = 40E-3
                if (tg_cell_n // no_cols) == 4:
                    z_ann_shift = 30E-3
                    
                ANS = 8
                ax.annotate(f'\textbf{{{annotations[tg_cell_n]}}}',
                            (x-x_ann_shift,z-z_ann_shift),
                            fontsize=ANS, zorder=100,
                            )
            else:
                x_ann_shift = 60E-3
                z_ann_shift = 60E-3
                ax.annotate(f'\textbf{{{annotations[tg_cell_n]}}}',
                            (x-x_ann_shift,z-z_ann_shift),
                            fontsize=8, zorder=100,
                            )
            textbox.append(f'\textbf{{{annotations[tg_cell_n]}}}: '
                           + f'{save_times_out[tg_cell_n]//60 - 120}')
        if show_textbox:
            textbox = r'$t$ (min): quad' + ', '.join(textbox)
            props = dict(boxstyle='round', facecolor='white', alpha=1)
            ax.text(xshift_textbox, yshift_textbox, textbox,
                    transform=ax.transAxes,
                    horizontalalignment='center',
                    verticalalignment='center', fontsize=8,
                    bbox=props)            
            
        if trajectory is not None:
            ax.plot(scale_x * trajectory[:,0], scale_x * trajectory[:,1],'o',
                    markersize = MS, c='k')
        
        ax.tick_params(axis='both', which='major', labelsize=TKFS)
        ax.grid(color='gray', linestyle='dashed', zorder = 2)
        
        ax.set_title(r'$r_l$ (g/kg), $t$ = {} min'.format(
                         -120 + save_times_out[1]//60))
        ax.set_xlabel(r'$x$ (km)', fontsize = LFS)
        ax.set_ylabel(r'$z$ (km)', fontsize = LFS)  
        ax.set_xlim((0,1.5))
        ax.set_ylim((0,1.5))
        ax.set_aspect('equal')
        if fig_path_tg_cells is not None:
            fig2.savefig(fig_path_tg_cells,
                        bbox_inches = 'tight',
                        pad_inches = 0.05
                        )   
            
#%% FUNCTION DEF: PLOT SCALAR FIELDS
def plot_scalar_field_frames_avg(grid, fields_with_time,
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
                                        show_target_cell_labels = False,
                                        no_cells_x = 0,
                                        no_cells_z = 0
                                        ):
    for i,fm in enumerate(field_names):
        print(i,fm)
    print(save_times)
    
    tick_ranges = grid.ranges

    no_rows = len(save_times)
    no_cols = len(field_names)
    
    fig, axes = plt.subplots(nrows=no_rows, ncols=no_cols,
                       figsize = figsize,
                       sharex=True, sharey=True)
    
    for time_n in range(no_rows):
        for field_n in range(no_cols):
            ax = axes[time_n,field_n]
            field = fields_with_time[time_n, field_n] * scales[field_n]
            ax_title = field_names[field_n]
            unit = units[field_n]
            if ax_title in ['T','p',r'Theta']:
                cmap = 'coolwarm'
                alpha = 1.0
            else :
                cmap = cmap_lcpp
                
            field_max = field.max()
            field_min = field.min()
            
            xticks_major = None
            xticks_minor = None

            norm_ = mpl.colors.Normalize 
            if ax_title in ['r_r', 'n_r']: #and field_max > 1E-2:
                norm_ = mpl.colors.LogNorm
                field_min = 0.01
                cmap = cmap_lcpp                
                if ax_title == 'r_r':
                    field_max = 1.
                    xticks_major = [0.01,0.1,1.]
                    xticks_minor = np.concatenate((
                            np.linspace(2E-2,1E-1,9),
                            np.linspace(2E-1,1,9),
                            ))
                elif ax_title == 'n_r':
                    field_max = 10.
                    xticks_major = [0.01,0.1,1.,10.]
                    xticks_minor = np.concatenate((
                            np.linspace(2E-2,1E-1,9),
                            np.linspace(2E-1,1,9),
                            np.linspace(2,10,9),
                            ))
            else: norm_ = mpl.colors.Normalize   
            
            if ax_title == r'Theta':
                field_min = 289.2
                field_max = 292.5
                xticks_major = [290,291,292]
            if ax_title == 'r_v':
                field_min = 6.5
                field_max = 7.6
                xticks_minor = [6.75,7.25]
            if ax_title == 'r_l':
                field_min = 0.0
                field_max = 1.3
                xticks_minor = [0.25,0.75,1.25]
                
            if ax_title == 'r_c':
                field_min = 0.0
                field_max = 1.3
                xticks_major = np.linspace(0,1.2,7)
            if ax_title == 'n_c':
                field_min = 0.0
                field_max = 150.
                xticks_major = [0,50,100,150]
                xticks_minor = [25,75,125]                
            if ax_title == 'n_mathrm{aero}':
                field_min = 0.0
                field_max = 150.
                xticks_major = [0,50,100,150]
            if ax_title in [r'R_mathrm{avg}', r'R_{2/1}', r'R_mathrm{eff}']:
                xticks_major = [1,5,10,15,20]
                field_min = 1
                field_max = 20.
                # Arabas 2015
                cmap = cmap_lcpp
                unit = r'si{micrometer}'
                
            oom_max = int(math.log10(field_max))
            
            my_format = False
            oom_factor = 1.0
            
            if oom_max > 2 or oom_max < 0:
                my_format = True
                oom_factor = 10**(-oom_max)
                
                field_min *= oom_factor
                field_max *= oom_factor            
            
            if oom_max ==2: str_format = '%.0f'
            
            else: str_format = '%.2g'
            
            if field_min/field_max < 1E-4:
                # Arabas 2015                
                cmap = cmap_lcpp
            
            # may remove fix alpha here
            # alpha = 1.0
            
            CS = ax.pcolormesh(*grid.corners, field*oom_factor,
                               cmap=cmap, alpha=alpha,
                                edgecolor='face', zorder=1,
                                norm = norm_(vmin=field_min, vmax=field_max),
                                rasterized=True,
                                antialiased=True, linewidth=0.0
                                )
            if ax_title == r'Theta':
                cmap_x = mpl.cm.get_cmap('coolwarm')
                print('cmap_x(0.0)')
                print(cmap_x(0.0))
                CS.cmap.set_under(cmap_x(0.0))
            else:
                CS.cmap.set_under('white')
            
            
            ax.set_xticks( np.linspace( tick_ranges[0,0],
                                             tick_ranges[0,1],
                                             no_ticks[0] ) )
            ax.set_yticks( np.linspace( tick_ranges[1,0],
                                             tick_ranges[1,1],
                                             no_ticks[1] ) )
            ax.tick_params(axis='both', which='major', labelsize=TKFS,
                           length = 3, width=1)
            ax.grid(color='gray', linestyle='dashed', zorder = 2)
            ax.set_aspect('equal')
            if time_n == no_rows-1:
                xticks1 = ax.xaxis.get_major_ticks()
                xticks1[-1].label1.set_visible(False)
                ax.set_xlabel(r'$x$ (km)', fontsize = LFS)
            if field_n == 0:            
                ax.set_ylabel(r'$z$ (km)', fontsize = LFS)
            ax.set_title( r'$t$ = {0} min'.format(int(save_times[time_n]/60)),
                         fontsize = TTFS)
            if time_n == 0:
                if no_cols == 4:
                    axins = inset_axes(
                                ax,
                                width='90%',  # width = 5% of parent_bbox width
                                height='8%',  # height
                                loc='lower center',
                                bbox_to_anchor=(0.0, 1.35, 1, 1),
                                bbox_transform=ax.transAxes,
                                borderpad=0,
                                )      
                else:
                    axins = inset_axes(
                                ax,
                                width='90%',  # width = 5% of parent_bbox width
                                height='8%',  # height
                                loc='lower center',
                                bbox_to_anchor=(0.0, 1.4, 1, 1),
                                bbox_transform=ax.transAxes,
                                borderpad=0,
                                )      

                cbar = plt.colorbar(CS, cax = axins,
                                    format =
                                        mticker.FormatStrFormatter(str_format),
                                    orientation='horizontal'
                                    )
                
                axins.xaxis.set_ticks_position('bottom')
                axins.tick_params(axis='x',direction='inout',which='both')
                axins.tick_params(axis='x', which='major', labelsize=TKFS,
                               length = 7, width=1)                
                axins.tick_params(axis='x', which='minor', labelsize=TKFS,
                               length = 5, width=0.5,bottom=True)                
                
                if xticks_major is not None:
                    axins.xaxis.set_ticks(xticks_major)
                if xticks_minor is not None:
                    axins.xaxis.set_ticks(xticks_minor, minor=True)
                if ax_title == 'n_c':
                    xticks2 = axins.xaxis.get_major_ticks()
                    xticks2[-1].label1.set_visible(False)                
                axins.set_title(r'${0}$ ({1})'.format(ax_title, unit))
            
                # 'my_format' dos not work with log scale here!!
                if my_format:
                    cbar.ax.text(field_min - (field_max-field_min),
                                 field_max + (field_max-field_min)*0.01,
                                 r'$\times,10^{{{}}}$'.format(oom_max),
                                 va='bottom', ha='left', fontsize = TKFS)
                cbar.ax.tick_params(labelsize=TKFS)
            
            if time_n == 2:
                if show_target_cells:
                    annotations = ['a', 'b', 'c', 'd', 'e', 'f',
                                   'g', 'h', 'i', 'j', 'k', 'l',
                                   'm', 'n', 'o', 'p', 'q', 'r']                
                    ### add the target cells
                    no_neigh_x = no_cells_x // 2
                    no_neigh_z = no_cells_z // 2
                    dx = grid.steps[0]
                    dz = grid.steps[1]
                    
                    no_tg_cells = len(target_cell_list[0])
                    LW_rect = .5
                    for tg_cell_n in range(no_tg_cells):
                        x = (target_cell_list[0, tg_cell_n]
                             - no_neigh_x - 0.1) * dx
                        z = (target_cell_list[1, tg_cell_n]
                             - no_neigh_z - 0.1) * dz
                        
                        rect = plt.Rectangle((x, z), dx*no_cells_x,
                                             dz*no_cells_z,
                                             fill=False,
                                             linewidth = LW_rect,
                                             edgecolor='k',
                                             zorder = 99)        
                        ax.add_patch(rect)
                        
                        if show_target_cell_labels:
                            
                            no_tg_cell_cols = 2
                            
                            x_ann_shift = 80E-3
                            z_ann_shift = 30E-3
                            # yields col number
                            if (tg_cell_n % no_tg_cell_cols) == 2:
                                x_ann_shift = -80E-3
                                
                            # yields row number
                            if (tg_cell_n // no_tg_cell_cols) == 1:
                                z_ann_shift = -40E-3
                            if (tg_cell_n // no_tg_cell_cols) == 2:
                                z_ann_shift = -20E-3
                            if (tg_cell_n // no_tg_cell_cols) == 3:
                                z_ann_shift = 40E-3
                            if (tg_cell_n // no_tg_cell_cols) == 4:
                                z_ann_shift = 30E-3
                            
                            ANS = 8
                            ax.annotate(
                                f'\textbf{{{annotations[tg_cell_n]}}}',
                                (x-x_ann_shift,z-z_ann_shift),
                                fontsize=ANS, zorder=100,
                                )       
                            
    if no_cols == 4:
        pad_ax_h = 0.1     
        pad_ax_v = 0.05
    else:        
        pad_ax_h = -0.5 
        pad_ax_v = 0.08
    fig.subplots_adjust(hspace=pad_ax_h, wspace=pad_ax_v)
             
    if fig_path is not None:
        fig.savefig(fig_path,
                    bbox_inches = 'tight',
                    pad_inches = 0.05,
                    dpi=600
                    )   
        
#%% FUNCTION DEF: PLOT ABSOLUTE DEVS OF TWO SCALAR FIELDS
def plot_scalar_field_frames_abs_dev_MA(grid,
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
    
    
    for i,fm in enumerate(field_names):
        print(i,fm)
    print(save_times)
    
    tick_ranges = grid.ranges

    no_rows = len(save_times)
    no_cols = len(field_names)
    
    abs_dev_with_time = fields_with_time2 - fields_with_time1
    
    fig, axes = plt.subplots(nrows=no_rows, ncols=no_cols,
                       figsize = figsize,
                       sharex=True, sharey=True)
    
    vline_pos = (0.33, 1.17, 1.33)
    
    for time_n in range(no_rows):
        for field_n in range(no_cols):
            ax = axes[time_n, field_n]
            
            if compare_type == 'Nsip' and time_n == 2:
                for vline_pos_ in vline_pos:
                    ax.axvline(vline_pos_, alpha=0.5, c='k',
                               zorder= 3, linewidth = 1.3)
            
            field = abs_dev_with_time[time_n, field_n] * scales[field_n]
            ax_title = field_names[field_n]

            print(time_n, ax_title, field.min(), field.max())

            unit = units[field_n]
            if ax_title in ['T','p',r'Theta']:
                cmap = 'coolwarm'
                alpha = 1.0
            else :
                cmap = cmap_lcpp
                
            field_max = field.max()
            field_min = field.min()
            
            xticks_major = None
            xticks_minor = None
            
            norm_ = mpl.colors.Normalize 

            if compare_type in ['Ncell', 'solute', 'Kernel']:
                if ax_title in ['r_r', 'n_r']: #and field_max > 1E-2:
                    norm_ = mpl.colors.Normalize
                    cmap = cmap_lcpp                
                    if ax_title == 'r_r':
                        field_max = 0.1
                        field_min = -field_max
                        linthresh=0.01
                        xticks_major = np.linspace(field_min,field_max,5)
                    elif ax_title == 'n_r':
                        pass
                        field_max = 0.9
                        field_min = -field_max
                else: norm_ = mpl.colors.Normalize   
                
                if ax_title == r'Theta':
                    field_min = 289.2
                    field_max = 292.5
                    xticks_major = [290,291,292]
                if ax_title == 'r_v':
                    field_min = 6.5
                    field_max = 7.6
                    xticks_minor = [6.75,7.25]
                if ax_title == 'r_l':
                    field_min = 0.0
                    field_max = 1.3
                    xticks_minor = [0.25,0.75,1.25]
                    
                if ax_title == 'r_c':
                    field_max = 0.3
                    field_min = -field_max                
                    xticks_major = np.linspace(field_min,field_max,5)
                if ax_title == 'n_c':
                    field_max = 40.
                    field_min = -field_max
                if ax_title == 'n_mathrm{aero}':
                    field_max = 40.
                    field_min = -field_max
                    xticks_major = np.linspace(field_min,field_max,5)
                if ax_title in [r'R_mathrm{avg}',
                                r'R_{2/1}', r'R_mathrm{eff}']:
                    field_max = 8.
                    field_min = -field_max
                    xticks_major = np.linspace(field_min,field_max,5)
                    # Arabas 2015
                    cmap = cmap_lcpp
                    unit = r'si{micrometer}'
                    
            elif compare_type == 'Nsip':
                if ax_title in ['r_r', 'n_r']: #and field_max > 1E-2:
                    norm_ = mpl.colors.Normalize
                    cmap = cmap_lcpp                
                    if ax_title == 'r_r':
                        field_max = 0.08
                        field_min = -field_max
                        linthresh=0.01
                        xticks_major = [field_min, -0.04, 0., 0.04, field_max]
                    elif ax_title == 'n_r':
                        field_max = 10.
                        xticks_major = [0.01,0.1,1.,10.]
                        xticks_minor = np.concatenate((
                                np.linspace(2E-2,1E-1,9),
                                np.linspace(2E-1,1,9),
                                np.linspace(2,10,9),
                                ))
                else: norm_ = mpl.colors.Normalize   
                
                if ax_title == r'Theta':
                    field_min = 289.2
                    field_max = 292.5
                    xticks_major = [290,291,292]
                if ax_title == 'r_v':
                    field_min = 6.5
                    field_max = 7.6
                    xticks_minor = [6.75,7.25]
                if ax_title == 'r_l':
                    field_min = 0.0
                    field_max = 1.3
                    xticks_minor = [0.25,0.75,1.25]
                    
                if ax_title == 'r_c':
                    field_max = 0.12
                    field_min = -field_max                
                    xticks_major = [-0.1, -0.05, 0., 0.05, 0.1]
                if ax_title == 'n_c':
                    field_min = 0.0
                    field_max = 150.
                if ax_title == 'n_mathrm{aero}':
                    field_max = 20.
                    field_min = -field_max
                    xticks_major = np.linspace(field_min,field_max,5)
                if ax_title in [r'R_mathrm{avg}',
                                r'R_{2/1}', r'R_mathrm{eff}']:
                    field_max = 5.
                    field_min = -field_max
                    xticks_major = np.linspace(field_min,field_max,5)
                    # Arabas 2015
                    cmap = cmap_lcpp
                    unit = r'si{micrometer}'
            else:
                if ax_title in ['r_r', 'n_r']: #and field_max > 1E-2:
                    norm_ = mpl.colors.Normalize
                    cmap = cmap_lcpp                
                    if ax_title == 'r_r':
                        field_max = 0.08
                        field_min = -field_max
                        linthresh=0.01
                        xticks_major = [field_min, -0.04, 0., 0.04, field_max]
                    elif ax_title == 'n_r':
                        field_max = 10.
                        xticks_major = [0.01,0.1,1.,10.]
                        xticks_minor = np.concatenate((
                                np.linspace(2E-2,1E-1,9),
                                np.linspace(2E-1,1,9),
                                np.linspace(2,10,9),
                                ))
                else: norm_ = mpl.colors.Normalize   
                
                if ax_title == r'Theta':
                    field_min = 289.2
                    field_max = 292.5
                    xticks_major = [290,291,292]
                if ax_title == 'r_v':
                    field_min = 6.5
                    field_max = 7.6
                    xticks_minor = [6.75,7.25]
                if ax_title == 'r_l':
                    field_min = 0.0
                    field_max = 1.3
                    xticks_minor = [0.25,0.75,1.25]
                    
                if ax_title == 'r_c':
                    field_max = 0.1
                    field_min = -field_max                
                    xticks_major = [field_min, -0.05, 0., 0.05, field_max]
                if ax_title == 'n_c':
                    field_min = 0.0
                    field_max = 150.
                if ax_title == 'n_mathrm{aero}':
                    field_max = 8.
                    field_min = -field_max
                    xticks_major = [field_min, -4, 0., 4, field_max]
                if ax_title in [r'R_mathrm{avg}',
                                r'R_{2/1}', r'R_mathrm{eff}']:
                    field_max = 3.
                    field_min = -field_max
                    xticks_major = np.linspace(field_min,field_max,7)
                    # Arabas 2015
                    cmap = cmap_lcpp
                    unit = r'si{micrometer}'
                    
            oom_max = int(math.log10(field_max))
            
            my_format = False
            oom_factor = 1.0
            
            if oom_max ==2: str_format = '%.0f'
            else: str_format = '%.2g'
            
            cmap = 'bwr'
            
            if False:
                norm = norm_(linthresh = linthresh,
                             vmin=field_min, vmax=field_max)
            else:
                norm = norm_(vmin=field_min, vmax=field_max)
            CS = ax.pcolormesh(*grid.corners, field*oom_factor,
                               cmap=cmap, alpha=alpha,
                                edgecolor='face', zorder=1,
                                norm = norm,
                                rasterized=True,
                                antialiased=True, linewidth=0.0
                                )
            CS.cmap.set_under('blue')
            CS.cmap.set_over('red')
            
            ax.set_xticks( np.linspace( tick_ranges[0,0],
                                             tick_ranges[0,1],
                                             no_ticks[0] ) )
            ax.set_yticks( np.linspace( tick_ranges[1,0],
                                             tick_ranges[1,1],
                                             no_ticks[1] ) )

            ax.tick_params(axis='both', which='major', labelsize=TKFS,
                           length = 3, width=1)
            ax.grid(color='gray', linestyle='dashed', zorder = 2)
            ax.set_aspect('equal')
            if time_n == no_rows-1:
                xticks1 = ax.xaxis.get_major_ticks()
                xticks1[-1].label1.set_visible(False)
                ax.set_xlabel(r'$x$ (km)', fontsize = LFS)
            if field_n == 0:            
                ax.set_ylabel(r'$z$ (km)', fontsize = LFS)
            ax.set_title( r'$t$ = {0} min'.format(int(save_times[time_n]/60)),
                         fontsize = TTFS)
            if time_n == 0:
                axins = inset_axes(ax,
                                   width='90%', # width=5% of parent_bbox width
                                   height='8%',  # height
                                   loc='lower center',
                                   bbox_to_anchor=(0.0, 1.35, 1, 1),
                                   bbox_transform=ax.transAxes,
                                   borderpad=0,
                                   )      
                cbar = plt.colorbar(CS, cax=axins,
                                    format=mticker.FormatStrFormatter(
                                            str_format),
                                    orientation='horizontal'
                                    )
                
                axins.xaxis.set_ticks_position('bottom')
                axins.tick_params(axis='x',direction='inout',which='both')
                axins.tick_params(axis='x', which='major', labelsize=TKFS,
                               length = 7, width=1)                
                axins.tick_params(axis='x', which='minor', labelsize=TKFS,
                               length = 5, width=0.5,bottom=True)                
                
                if xticks_major is not None:
                    axins.xaxis.set_ticks(xticks_major)
                if xticks_minor is not None:
                    axins.xaxis.set_ticks(xticks_minor, minor=True)
                axins.set_title(r'$Delta {0}$ ({1})'.format(ax_title, unit))

                if my_format:
                    cbar.ax.text(field_min - (field_max-field_min),
                                 field_max + (field_max-field_min)*0.01,
                                 r'$\times,10^{{{}}}$'.format(oom_max),
                                 va='bottom', ha='left', fontsize = TKFS)
                cbar.ax.tick_params(labelsize=TKFS)

            if show_target_cells:
                ### add the target cells
                no_neigh_x = no_cells_x // 2
                no_neigh_z = no_cells_z // 2
                dx = grid.steps[0]
                dz = grid.steps[1]
                
                no_tg_cells = len(target_cell_list[0])
                LW_rect = .5
                for tg_cell_n in range(no_tg_cells):
                    x = (target_cell_list[0, tg_cell_n]
                         - no_neigh_x - 0.1) * dx
                    z = (target_cell_list[1, tg_cell_n]
                         - no_neigh_z - 0.1) * dz
                    
                    rect = plt.Rectangle((x, z), dx*no_cells_x,dz*no_cells_z,
                                         fill=False,
                                         linewidth = LW_rect,
                                         edgecolor='k',
                                         zorder = 99)        
                    ax.add_patch(rect)


    pad_ax_h = 0.1     
    pad_ax_v = 0.05
    fig.subplots_adjust(hspace=pad_ax_h, wspace=pad_ax_v)
             
    if fig_path is not None:
        fig.savefig(fig_path,
                    bbox_inches = 'tight',
                    pad_inches = 0.05,
                    dpi=600
                    )     
    plt.close('all')
    
    ### PLOT ABS ERROR OF THE DIFFERENCE OF TWO FIELDS:
    # Var = Var_1 + Var_2 (assuming no correlations)

    print('plotting ABS ERROR of field1 - field2')
    
    std1 = np.nan_to_num( fields_with_time_std1 )
    std2 = np.nan_to_num( fields_with_time_std2 )
    std = np.sqrt( std1**2 + std2**2 )

    tick_ranges = grid.ranges
    
    no_rows = len(save_times)
    no_cols = len(field_names)
    
    for time_n in range(no_rows):
        for field_n in range(no_cols):
            std[time_n, field_n] *= scales[field_n]

    field_max_all = np.amax(std, axis=(0,2,3))
#    field_min_all = np.amin(std, axis=(0,2,3))

    pad_ax_h = 0.1     
    pad_ax_v = 0.03

    ### PLOT ABS ERROR  
    fig2, axes = plt.subplots(nrows=no_rows, ncols=no_cols,
                       figsize = figsize,
                       sharex=True, sharey=True)
            
    for time_n in range(no_rows):
        for field_n in range(no_cols):
            ax = axes[time_n,field_n]
            field = std[time_n, field_n]
            ax_title = field_names[field_n]
            unit = units[field_n]
            cmap = cmap_lcpp
            
            field_min = 0.
            field_max = field_max_all[field_n]

            xticks_major = None
            xticks_minor = None
            
            if compare_type == 'Nsip':
                if ax_title == 'n_mathrm{aero}':
                    xticks_minor = np.array((1.25,3.75))
                if ax_title == 'r_c':
                    field_max = 0.06
                    xticks_minor = np.linspace(0.01,0.05,3)
                if ax_title == 'r_r':
                    xticks_minor = np.linspace(0.01,0.03,2)
                if ax_title in [r'R_mathrm{eff}']:
                    xticks_major = np.linspace(0.,1.5,4)
            
            if compare_type == 'dt_col':
                if ax_title == 'n_mathrm{aero}':
                    xticks_minor = np.array((1.25, 3.75, 6.25))
                if ax_title == 'r_c':
                    xticks_major = np.array((0.,0.02,0.04,0.06))
                if ax_title == 'r_r':
                    xticks_minor = np.linspace(0.01,0.05,3)
                if ax_title in [r'R_mathrm{eff}']:
                    xticks_major = np.linspace(0.,1.5,4)
            
            norm_ = mpl.colors.Normalize 
                
            str_format = '%.2g'
            
            my_format = False
            oom_factor = 1.0
#            oom_max = oom = 1

            CS = ax.pcolormesh(*grid.corners,
                               field*oom_factor,
                               cmap=cmap, alpha=alpha,
                                edgecolor='face', zorder=1,
                                norm = norm_(vmin=field_min, vmax=field_max),
                                rasterized=True,
                                antialiased=True, linewidth=0.0
                                )
            CS.cmap.set_under('white')
            
            ax.set_xticks( np.linspace( tick_ranges[0,0],
                                             tick_ranges[0,1],
                                             no_ticks[0] ) )
            ax.set_yticks( np.linspace( tick_ranges[1,0],
                                             tick_ranges[1,1],
                                             no_ticks[1] ) )

            ax.tick_params(axis='both', which='major', labelsize=TKFS,
                           length = 3, width=1)
            ax.grid(color='gray', linestyle='dashed', zorder = 2)
            ax.set_aspect('equal')
            if time_n == no_rows-1:
                xticks1 = ax.xaxis.get_major_ticks()
                xticks1[-1].label1.set_visible(False)
                ax.set_xlabel(r'$x$ (km)', fontsize = LFS)
            if field_n == 0:            
                ax.set_ylabel(r'$z$ (km)', fontsize = LFS)
            ax.set_title( r'$t$ = {0} min'.format(int(save_times[time_n]/60)),
                         fontsize = TTFS)
            if time_n == 0:
                axins = inset_axes(ax,
                                   width='90%', # width=5% of parent_bbox width
                                   height='8%', # height
                                   loc='lower center',
                                   bbox_to_anchor=(0.0, 1.35, 1, 1),
                                   bbox_transform=ax.transAxes,
                                   borderpad=0,
                                   )      
                cbar = plt.colorbar(
                           CS, cax=axins,
                           format=mticker.FormatStrFormatter(str_format),
                           orientation='horizontal'
                           )
                axins.xaxis.set_ticks_position('bottom')
                axins.tick_params(axis='x',direction='inout',which='both')
                axins.tick_params(axis='x', which='major', labelsize=TKFS,
                               length = 7, width=1)                
                axins.tick_params(axis='x', which='minor', labelsize=TKFS,
                               length = 5, width=0.5,bottom=True)                
                
                if xticks_major is not None:
                    axins.xaxis.set_ticks(xticks_major)
                if xticks_minor is not None:
                    axins.xaxis.set_ticks(xticks_minor, minor=True)
                if ax_title in [r'R_mathrm{avg}', r'R_{2/1}',
                                r'R_mathrm{eff}']:
                    unit = r'si{micrometer}'                                            
                axins.set_title(r'${0}$ abs. error ({1})'.format(ax_title,
                                                                 unit))
            
                # 'my_format' dos not work with log scale here!!
                if my_format:
                    cbar.ax.text(1.0,1.0,
                                 r'$\times,10^{{{}}}$'.format(oom_max),
                                 va='bottom', ha='right', fontsize = TKFS,
                                 transform=ax.transAxes)
                cbar.ax.tick_params(labelsize=TKFS)

            if show_target_cells:
                ### add the target cells
                no_neigh_x = no_cells_x // 2
                no_neigh_z = no_cells_z // 2
                dx = grid.steps[0]
                dz = grid.steps[1]
                
                no_tg_cells = len(target_cell_list[0])
                LW_rect = .5
                for tg_cell_n in range(no_tg_cells):
                    x = (target_cell_list[0, tg_cell_n]
                         - no_neigh_x - 0.1) * dx
                    z = (target_cell_list[1, tg_cell_n]
                         - no_neigh_z - 0.1) * dz
                    
                    rect = plt.Rectangle((x, z), dx*no_cells_x,dz*no_cells_z,
                                         fill=False,
                                         linewidth = LW_rect,
                                         edgecolor='k',
                                         zorder = 99)        
                    ax.add_patch(rect)

    fig2.subplots_adjust(hspace=pad_ax_h, wspace=pad_ax_v)

    if fig_path_abs_err is not None:
        fig2.savefig(fig_path_abs_err,
                    bbox_inches = 'tight',
                    pad_inches = 0.05,
                    dpi=600
                    )        
        
#%% PARTICLE TRACKING        
# traj = [ pos0, pos1, pos2, .. ]
# where pos0 = [pos_x_0, pos_z_0] is pos at time0
# where pos_x_0 = [x0, x1, x2, ...]
# selection = [n1, n2, ...] --> take only these indic. from the list of traj !!!
# title font size (pt)
# TTFS = 10
# # labelsize (pt)
# LFS = 10
# # ticksize (pt)
# TKFS = 10        
def plot_particle_trajectories(traj, grid, selection=None,
                               no_ticks=[6,6], figsize=(8,8),
                               MS=1.0, arrow_every=5,
                               ARROW_SCALE=12,ARROW_WIDTH=0.005, 
                               TTFS = 10, LFS=10,TKFS=10,fig_name=None,
                               t_start=0, t_end=3600):
    centered_u_field = ( grid.velocity[0][0:-1,0:-1]
                         + grid.velocity[0][1:,0:-1] ) * 0.5
    centered_w_field = ( grid.velocity[1][0:-1,0:-1]
                         + grid.velocity[1][0:-1,1:] ) * 0.5
    
    pos_x = grid.centers[0][arrow_every//2::arrow_every,
                            arrow_every//2::arrow_every]
    pos_z = grid.centers[1][arrow_every//2::arrow_every,
                            arrow_every//2::arrow_every]
    
    vel_x = centered_u_field[arrow_every//2::arrow_every,
                             arrow_every//2::arrow_every]
    vel_z = centered_w_field[arrow_every//2::arrow_every,
                             arrow_every//2::arrow_every]
    
    tick_ranges = grid.ranges
    fig, ax = plt.subplots(figsize=figsize)
    if traj[0,0].size == 1:
        ax.plot(traj[:,0], traj[:,1] ,'o', markersize = MS)
    else:        
        if selection == None: selection=range(len(traj[0,0]))
        for ID in selection:
            x = traj[:,0,ID]
            z = traj[:,1,ID]
            ax.plot(x,z,'o', markersize = MS)
    ax.quiver(pos_x, pos_z, vel_x, vel_z,
              pivot = 'mid',
              width = ARROW_WIDTH, scale = ARROW_SCALE, zorder=99 )
    ax.set_xticks( np.linspace( tick_ranges[0,0], tick_ranges[0,1],
                                no_ticks[0] ) )
    ax.set_yticks( np.linspace( tick_ranges[1,0], tick_ranges[1,1],
                                no_ticks[1] ) )
    ax.tick_params(axis='both', which='major', labelsize=TKFS)
    ax.set_xlabel('horizontal position (m)', fontsize = LFS)
    ax.set_ylabel('vertical position (m)', fontsize = LFS)
    ax.set_title(
    'Air velocity field and arbitrary particle trajectories\nfrom $t = $'\
    + str(t_start) + ' s to ' + str(t_end) + ' s',
        fontsize = TTFS, y = 1.04)
    ax.grid(color='gray', linestyle='dashed', zorder = 0)    
    fig.tight_layout()
    if fig_name is not None:
        fig.savefig(fig_name)
        
        
#%% PARTICLE POSITIONS AND VELOCITIES

def plot_pos_vel_pt(pos, vel, grid,
                    figsize=(8,8), no_ticks = [6,6],
                    MS = 1.0, ARRSCALE=2, fig_name=None):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(grid.corners[0], grid.corners[1], 'x', color='red', markersize=MS)
    ax.plot(pos[0],pos[1], 'o', color='k', markersize=2*MS)
    ax.quiver(*pos, *vel, scale=ARRSCALE, pivot='mid')
    x_min = grid.ranges[0,0]
    x_max = grid.ranges[0,1]
    y_min = grid.ranges[1,0]
    y_max = grid.ranges[1,1]
    ax.set_xticks( np.linspace(x_min, x_max, no_ticks[0]) )
    ax.set_yticks( np.linspace(y_min, y_max, no_ticks[1]) )
    ax.set_xticks(grid.corners[0][:,0], minor = True)
    ax.set_yticks(grid.corners[1][0,:], minor = True)
    ax.grid()
    fig.tight_layout()
    if fig_name is not None:
        fig.savefig(fig_name)
        
# pos = [pos0, pos1, pos2, ..] where pos0[0] = [x0, x1, x2, x3, ..] etc
def plot_pos_vel_pt_with_time(pos_data, vel_data, grid, save_times,
                    figsize=(8,8), no_ticks = [6,6],
                    MS = 1.0, ARRSCALE=2, fig_name=None):
    no_rows = len(pos_data)
    fig, axes = plt.subplots(nrows=no_rows, figsize=figsize)
    for i,ax in enumerate(axes):
        pos = pos_data[i]
        vel = vel_data[i]
        ax.plot(grid.corners[0], grid.corners[1], 'x', color='red',
                markersize=MS)
        ax.plot(pos[0],pos[1], 'o', color='k', markersize=2*MS)
        ax.quiver(*pos, *vel, scale=ARRSCALE, pivot='mid')
        x_min = grid.ranges[0,0]
        x_max = grid.ranges[0,1]
        y_min = grid.ranges[1,0]
        y_max = grid.ranges[1,1]
        ax.set_xticks( np.linspace(x_min, x_max, no_ticks[0]) )
        ax.set_yticks( np.linspace(y_min, y_max, no_ticks[1]) )
        ax.set_xticks(grid.corners[0][:,0], minor = True)
        ax.set_yticks(grid.corners[1][0,:], minor = True)
        ax.grid()
        ax.set_title('t = ' + str(save_times[i]) + ' s')
        ax.set_xlabel('x (m)')
        ax.set_xlabel('z (m)')
    fig.tight_layout()
    if fig_name is not None:
        fig.savefig(fig_name)
        
#%% ENSEMBLE DATA
        
def plot_ensemble_data(kappa, mass_density, eta, r_critmin,
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
        ensemble_dir):
    if dist == 'expo':
        conc_per_mass_np = conc_per_mass_expo_np
    elif dist == 'lognormal'   :     
        conc_per_mass_np = conc_per_mass_lognormal_np
    
    sample_mode = 'given_bins'
### 1. plot xi_avg vs r    
    no_rows = 1
    fig, axes = plt.subplots(nrows=no_rows, figsize=(10,6*no_rows))
    ax=axes
    ax.plot(bins_rad_centers[3], f_m_num_avg*bins_mass_width)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    if dist == 'expo':
        ax.set_yticks(np.logspace(-4,8,13))
        ax.set_ylim((1.0E-4,1.0E8))
    ax.grid()
    
    ax.set_xlabel(r'radius ($mathrm{mu m}$)')
    ax.set_ylabel(r'mean multiplicity per SIP')
    
    fig.tight_layout()
    
    fig_name = f'xi_vs_R_{sample_mode}_no_sims_{no_sims}'
    if sample_mode == 'given_bins': fig_name += '.png'
    elif sample_mode == 'auto_bins': fig_name += f'_no_bins_{no_bins}.png'
        
    fig.savefig(ensemble_dir + fig_name)

### 2. my lin approx plot
    
    m_ = np.logspace(np.log10(bins_mass[0]), np.log10(bins_mass[-1]), 1000)
    R_ = compute_radius_from_mass_vec(m_*1.0E18, mass_density)
#        R_ = compute_radius_from_mass_vec(m_*1.0E18, c.mass_density_water_liquid_NTP)
#        f_m_ana = conc_per_mass_np(m_, DNC0, DNC0/LWC0)
    f_m_ana = conc_per_mass_np(m_, *dist_par)
#    g_m_ana = m_ * f_m_ana
#    g_ln_r_ana = 3 * m_ * g_m_ana * 1000.0
    
    no_rows = 1
    
    MS= 15.0
    
    fig, axes = plt.subplots(nrows=no_rows, figsize=(10,10*no_rows))
    ax = axes
    ax.plot(m_, f_m_ana_)
    ax.plot(bins_mass_center_lin_my, f_m_num_avg_my_ext, 'x', c='green',
            markersize=MS, zorder=99)
    ax.plot(bins_mass_centers_my[4], f_m_num_avg_my, 'x', c = 'red',
            markersize=MS,
            zorder=99)
    # for n in range(len(bins_mass_centers_my[0])):
    # lin approx
    for n in range(len(bins_mass_center_lin_my)-1):
        m_ = np.linspace(bins_mass_center_lin_my[n],
                          bins_mass_center_lin_my[n+1], 100)
        f_ = lin_par[0,n] + lin_par[1,n] * m_
        # ax.plot(m_,f_)
        ax.plot(m_,f_, '-.', c = 'orange')
    # for n in range(len(bins_mass_center_lin_my)-2):
    #     m_ = np.linspace(bins_mass_center_lin_my[n],
    #                       bins_mass_center_lin_my[n+2], 1000)
    #     f_ = aa[0,n] + aa[1,n] * m_ + aa[2,n] * m_*m_
    #     ax.plot(m_,f_)
    #     # ax.plot(m_,f_, c = 'k')
    ax.vlines(bins_mass_my,f_m_num_avg_my_ext.min()*0.5,
              f_m_num_avg_my_ext.max()*2,
              linestyle='dashed', zorder=0)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('mass (kg)')
    ax.set_ylabel(r'$f_m$ $mathrm{(kg^{-1} , m^{-3})}$')
    
    fig.tight_layout()
    
    fig_name = f'fm_my_lin_approx_{sample_mode}_no_sims_{no_sims}'
    if sample_mode == 'given_bins': fig_name += '.png'
    elif sample_mode == 'auto_bins': fig_name += f'_no_bins_{no_bins}.png'
        
    fig.savefig(ensemble_dir + fig_name)
    
### 3. SAMPLED DATA: fm gm glnR moments
    if dist == 'expo':
        bins_mass_center_exact = bins_mass_centers[3]
        bins_rad_center_exact = bins_rad_centers[3]
    elif dist == 'lognormal':
        bins_mass_center_exact = bins_mass_centers[0]
        bins_rad_center_exact = bins_rad_centers[0]
    
    m_ = np.logspace(np.log10(bins_mass[0]), np.log10(bins_mass[-1]), 1000)
    R_ = compute_radius_from_mass_vec(m_*1.0E18, mass_density)
    f_m_ana = conc_per_mass_np(m_, *dist_par)
#    g_m_ana = m_ * f_m_ana
#    g_ln_r_ana = 3 * m_ * g_m_ana * 1000.0
    
    no_rows = 5
    fig, axes = plt.subplots(nrows=no_rows, figsize=(10,5*no_rows))
    ax = axes[0]
    ax.plot(bins_mass_center_exact, f_m_num_sampled, 'x')
    ax.plot(m_, f_m_ana_)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('mass (kg)')
    ax.set_ylabel(r'$f_m$ $mathrm{(kg^{-1} , m^{-3})}$')
    
    ax = axes[1]
    ax.plot(bins_mass_center_exact, g_m_num_sampled, 'x')
    ax.plot(m_, g_m_ana_)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('mass (kg)')
    ax.set_ylabel(r'$g_m$ $mathrm{(m^{-3})}$')
    
    
    ax = axes[2]
    ax.plot(bins_rad_center_exact, g_ln_r_num_sampled*1000.0, 'x')
    ax.plot(R_, g_ln_r_ana_)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('radius $mathrm{(mu m)}$')
    ax.set_ylabel(r'$g_{ln(r)}$ $mathrm{(g ; m^{-3})}$')
    # ax.xaxis.set_ticks(np.logspace(np.log10(0.6), np.log10(30),18))
    # ax.xaxis.set_ticks([0.6,1.0,2.0,5.0,10.0,20.0,30.0])
    if dist == 'expo':
        ax.set_xticks([0.6,1.0,2.0,5.0,10.0,20.0,30.0])
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        # ax.get_xaxis().get_major_formatter().labelOnlyBase = False
        ax.yaxis.set_ticks(np.logspace(-11,0,12))
    ax.grid(which='both')
    
    # fm with my binning method
    ax = axes[3]
    ax.plot(bins_mass_centers_my[4], f_m_num_avg_my, 'x')
    ax.plot(m_, f_m_ana_)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('mass (kg)')
    ax.set_ylabel(r'$f_m$ $mathrm{(kg^{-1} , m^{-3})}$  [my lin fit]')
    ax.grid()
    
    ax = axes[4]
    for n in range(4):
        ax.plot(n*np.ones_like(moments_sampled[n]),
                moments_sampled[n]/moments_an[n], 'o')
    ax.errorbar(np.arange(4), moments_sampled_avg_norm, moments_sampled_std_norm,
                fmt = 'x' , c = 'k', markersize = 20.0, linewidth =5.0,
                capsize=10, elinewidth=5, markeredgewidth=2,
                zorder=99)
    ax.plot(np.arange(4), np.ones_like(np.arange(4)))
    ax.xaxis.set_ticks([0,1,2,3])
    ax.set_xlabel('$k$')
    # ax.set_ylabel(r'($k$-th moment of $f_m$)/(analytic value)')
    ax.set_ylabel(r'$lambda_k / lambda_{k,analytic}$')
    
    for ax in axes[:2]:
        ax.grid()
    
    fig.tight_layout()
    
    fig_name = f'fm_gm_glnR_moments_{sample_mode}_no_sims_{no_sims}'
    if sample_mode == 'given_bins': fig_name += '.png'
    elif sample_mode == 'auto_bins': fig_name += f'_no_bins_{no_bins}.png'
    fig.savefig(ensemble_dir + fig_name)

### 4. SAMPLED DATA: DEVIATIONS OF fm
    no_rows = 4
    fig, axes = plt.subplots(nrows=no_rows, figsize=(10,5*no_rows), sharex=True)
    
    ax_titles = ['lin', 'log', 'COM', 'exact']
    
    for n in range(no_rows):
        ax = axes[n]
        f_m_ana = conc_per_mass_np(bins_mass_centers[n], *dist_par)
        ax.plot(bins_mass_centers[n], (f_m_num_sampled-f_m_ana)/f_m_ana, 'x')
        # ax.plot(bins_mass_width, (f_m_num_sampled-f_m_ana)/f_m_ana, 'x')
        ax.set_xscale('log')
        ax.set_ylabel(r'$(f_{m,num}-f_{m}(\tilde{m}))/f_{m}(\tilde{m})$ ')
        ax.set_title(ax_titles[n])
    # axes[3].set_xlabel('bin width $Delta hat{m}$ (kg)')
    axes[3].set_xlabel('mass (kg)')
    
    for ax in axes:
        ax.grid()
    
    fig.tight_layout()
    fig_name = f'Deviations_fm_sampled_data_{sample_mode}_no_sims_{no_sims}'
    if sample_mode == 'given_bins': fig_name += '.png'
    elif sample_mode == 'auto_bins': fig_name += f'_no_bins_{no_bins}.png'
    fig.savefig(ensemble_dir + fig_name)
    


### PLOTTING STATISTICAL ANALYSIS OVER no_sim runs

### 5. ERRORBARS: fm gm g_ln_r moments GIVEN BINS
    
    m_ = np.logspace(np.log10(bins_mass[0]), np.log10(bins_mass[-1]), 1000)
    R_ = compute_radius_from_mass_vec(m_*1.0E18, mass_density)
    f_m_ana = conc_per_mass_np(m_, *dist_par)
#    g_m_ana = m_ * f_m_ana
#    g_ln_r_ana = 3 * m_ * g_m_ana * 1000.0
    
    no_rows = 5
    fig, axes = plt.subplots(nrows=no_rows, figsize=(10,5*no_rows))
    ax = axes[0]
    ax.errorbar(bins_mass_center_exact,
                f_m_num_avg,
                f_m_num_std,
                fmt = 'x' ,
                # c = 'k',
                # c = 'lightblue',
                markersize = 5.0,
                linewidth = 2.0,
                capsize=3, elinewidth=2, markeredgewidth=1,
                zorder=99)
    ax.plot(m_, f_m_ana_)
    ax.set_xscale('log')
    ax.set_yscale('log')
    if dist == 'expo':
        ax.set_yticks(np.logspace(6,21,16))
        ax.set_ylim((f_m_ana_[-1],1.0E21))
    # ax.set_ylim((1.0E6,1.0E21))
    ax.set_xlabel('mass (kg) [exact centers]')
    ax.set_ylabel(r'$f_m$ $mathrm{(kg^{-1} , m^{-3})}$')
    ax = axes[1]
    ax.errorbar(bins_mass_center_exact,
                # bins_mass_width,
                g_m_num_avg,
                g_m_num_std,
                fmt = 'x' ,
                # c = 'k',
                # c = 'lightblue',
                markersize = 5.0,
                linewidth = 2.0,
                capsize=3, elinewidth=2, markeredgewidth=1,
                zorder=99)
    ax.plot(m_, g_m_ana_)
    ax.set_xscale('log')
    ax.set_yscale('log')
    if dist == 'expo':
        ax.set_yticks(np.logspace(-4,8,13))
        # ax.set_ylim((1.0E-4,3.0E8))
        ax.set_ylim((g_m_ana_[-1],3.0E8))
        ax.set_xlabel('mass (kg) [exact centers]')
        ax.set_ylabel(r'$g_m$ $mathrm{(m^{-3})}$')
    ax = axes[2]
    ax.errorbar(bins_rad_center_exact,
                # bins_mass_width,
                g_ln_r_num_avg*1000,
                g_ln_r_num_std*1000,
                fmt = 'x' ,
                # c = 'k',
                # c = 'lightblue',
                markersize = 5.0,
                linewidth = 2.0,
                capsize=3, elinewidth=2, markeredgewidth=1,
                zorder=99)
    ax.plot(R_, g_ln_r_ana_)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('radius $mathrm{(mu m)}$ [exact centers]')
    ax.set_ylabel(r'$g_{ln(r)}$ $mathrm{(g ; m^{-3})}$')
    # ax.xaxis.set_ticks(np.logspace(np.log10(0.6), np.log10(30),18))
    # ax.xaxis.set_ticks([0.6,1.0,2.0,5.0,10.0,20.0,30.0])
    if dist == 'expo':
        ax.set_xticks([0.6,1.0,2.0,5.0,10.0,20.0,30.0])
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.set_yticks(np.logspace(-11,0,12))
        ax.set_ylim((1.0E-11,5.0))
    ax.grid(which='both')
    
    # my binning method
    ax = axes[3]
    ax.errorbar(bins_mass_centers_my[4],
                # bins_mass_width,
                f_m_num_avg_my,
                f_m_num_std_my,
                fmt = 'x' ,
                # c = 'k',
                # c = 'lightblue',
                markersize = 5.0,
                linewidth = 2.0,
                capsize=3, elinewidth=2, markeredgewidth=1,
                zorder=99)
    ax.plot(m_, f_m_ana_)
    ax.set_xscale('log')
    ax.set_yscale('log')
    if dist == 'expo':
        ax.set_yticks(np.logspace(6,21,16))
        # ax.set_ylim((1.0E6,1.0E21))
        ax.set_ylim((f_m_ana_[-1],1.0E21))
    ax.set_xlabel('mass (kg) [my lin fit centers]')
    ax.set_ylabel(r'$f_m$ $mathrm{(kg^{-1} , m^{-3})}$  [my lin fit]')
    ax.grid()
    
    ax = axes[4]
    for n in range(4):
        ax.plot(n*np.ones_like(moments_sampled[n]),
                moments_sampled[n]/moments_an[n], 'o')
    ax.errorbar(np.arange(4), moments_sampled_avg_norm, moments_sampled_std_norm,
                fmt = 'x' , c = 'k', markersize = 20.0, linewidth =5.0,
                capsize=10, elinewidth=5, markeredgewidth=2,
                zorder=99)
    ax.plot(np.arange(4), np.ones_like(np.arange(4)))
    ax.xaxis.set_ticks([0,1,2,3])
    ax.set_xlabel('$k$')
    # ax.set_ylabel(r'($k$-th moment of $f_m$)/(analytic value)')
    ax.set_ylabel(r'$lambda_k / lambda_{k,mathrm{analytic}}$')
    
    for ax in axes[:2]:
        ax.grid()
    
    fig.tight_layout()
    fig_name = f'fm_gm_glnR_moments_errorbars_{sample_mode}_no_sims_{no_sims}'
    if sample_mode == 'given_bins': fig_name += '.png'
    elif sample_mode == 'auto_bins': fig_name += f'_no_bins_{no_bins}.png'
    fig.savefig(ensemble_dir + fig_name)
    
### 5b. ERRORBARS: fm gm g_ln_r moments AUTO BINS
    sample_mode = 'auto_bins'
    if dist == 'expo':
        bins_mass_center_exact = bins_mass_centers_auto[3]
        bins_rad_center_exact = bins_rad_centers_auto[3]
    elif dist == 'lognormal':
        bins_mass_center_exact = bins_mass_centers_auto[0]
        bins_rad_center_exact = bins_rad_centers_auto[0]

    m_ = np.logspace(np.log10(bins_mass[0]), np.log10(bins_mass[-1]), 1000)
    R_ = compute_radius_from_mass_vec(m_*1.0E18, mass_density)
    f_m_ana = conc_per_mass_np(m_, *dist_par)
#    g_m_ana = m_ * f_m_ana
#    g_ln_r_ana = 3 * m_ * g_m_ana * 1000.0
    
    no_rows = 5
    fig, axes = plt.subplots(nrows=no_rows, figsize=(10,5*no_rows))
    ax = axes[0]
    ax.errorbar(bins_mass_center_exact,
                f_m_num_avg_auto,
                f_m_num_std_auto,
                fmt = 'x' ,
                # c = 'k',
                # c = 'lightblue',
                markersize = 5.0,
                linewidth = 2.0,
                capsize=3, elinewidth=2, markeredgewidth=1,
                zorder=99)
    ax.plot(m_, f_m_ana_)
    ax.set_xscale('log')
    ax.set_yscale('log')
    if dist == 'expo':
        ax.set_yticks(np.logspace(6,21,16))
        ax.set_ylim((f_m_ana_[-1],1.0E21))
    # ax.set_ylim((1.0E6,1.0E21))
    ax.set_xlabel('mass (kg) [exact centers]')
    ax.set_ylabel(r'$f_m$ $mathrm{(kg^{-1} , m^{-3})}$')
    ax = axes[1]
    ax.errorbar(bins_mass_center_exact,
                # bins_mass_width,
                g_m_num_avg_auto,
                g_m_num_std_auto,
                fmt = 'x' ,
                # c = 'k',
                # c = 'lightblue',
                markersize = 5.0,
                linewidth = 2.0,
                capsize=3, elinewidth=2, markeredgewidth=1,
                zorder=99)
    ax.plot(m_, g_m_ana_)
    ax.set_xscale('log')
    ax.set_yscale('log')
    if dist == 'expo':
        ax.set_yticks(np.logspace(-4,8,13))
        # ax.set_ylim((1.0E-4,3.0E8))
        ax.set_ylim((g_m_ana_[-1],3.0E8))
        ax.set_xlabel('mass (kg) [exact centers]')
        ax.set_ylabel(r'$g_m$ $mathrm{(m^{-3})}$')
    ax = axes[2]
    ax.errorbar(bins_rad_center_exact,
                # bins_mass_width,
                g_ln_r_num_avg_auto*1000,
                g_ln_r_num_std_auto*1000,
                fmt = 'x' ,
                # c = 'k',
                # c = 'lightblue',
                markersize = 5.0,
                linewidth = 2.0,
                capsize=3, elinewidth=2, markeredgewidth=1,
                zorder=99)
    ax.plot(R_, g_ln_r_ana_)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('radius $mathrm{(mu m)}$ [exact centers]')
    ax.set_ylabel(r'$g_{ln(r)}$ $mathrm{(g ; m^{-3})}$')
    # ax.xaxis.set_ticks(np.logspace(np.log10(0.6), np.log10(30),18))
    # ax.xaxis.set_ticks([0.6,1.0,2.0,5.0,10.0,20.0,30.0])
    if dist == 'expo':
        ax.set_xticks([0.6,1.0,2.0,5.0,10.0,20.0,30.0])
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.set_yticks(np.logspace(-11,0,12))
        ax.set_ylim((1.0E-11,5.0))
    ax.grid(which='both')
    
    # my binning method
    ax = axes[3]
    ax.errorbar(bins_mass_centers_my[4],
                # bins_mass_width,
                f_m_num_avg_my,
                f_m_num_std_my,
                fmt = 'x' ,
                # c = 'k',
                # c = 'lightblue',
                markersize = 5.0,
                linewidth = 2.0,
                capsize=3, elinewidth=2, markeredgewidth=1,
                zorder=99)
    ax.plot(m_, f_m_ana_)
    ax.set_xscale('log')
    ax.set_yscale('log')
    if dist == 'expo':
        ax.set_yticks(np.logspace(6,21,16))
        # ax.set_ylim((1.0E6,1.0E21))
        ax.set_ylim((f_m_ana_[-1],1.0E21))
    ax.set_xlabel('mass (kg) [my lin fit centers]')
    ax.set_ylabel(r'$f_m$ $mathrm{(kg^{-1} , m^{-3})}$  [my lin fit]')
    ax.grid()
    
    ax = axes[4]
    for n in range(4):
        ax.plot(n*np.ones_like(moments_sampled[n]),
                moments_sampled[n]/moments_an[n], 'o')
    ax.errorbar(np.arange(4), moments_sampled_avg_norm, moments_sampled_std_norm,
                fmt = 'x' , c = 'k', markersize = 20.0, linewidth =5.0,
                capsize=10, elinewidth=5, markeredgewidth=2,
                zorder=99)
    ax.plot(np.arange(4), np.ones_like(np.arange(4)))
    ax.xaxis.set_ticks([0,1,2,3])
    ax.set_xlabel('$k$')
    # ax.set_ylabel(r'($k$-th moment of $f_m$)/(analytic value)')
    ax.set_ylabel(r'$lambda_k / lambda_{k,mathrm{analytic}}$')
    
    for ax in axes[:2]:
        ax.grid()
    
    fig.tight_layout()
    fig_name = f'fm_gm_glnR_moments_errorbars_{sample_mode}_no_sims_{no_sims}'
    if sample_mode == 'given_bins': fig_name += '.png'
    elif sample_mode == 'auto_bins': fig_name += f'_no_bins_{no_bins}.png'
    fig.savefig(ensemble_dir + fig_name)

### 6. ERRORBARS: DEVIATIONS of fm SEPA PLOTS GIVEN BINS
    sample_mode = 'given_bins'
    no_rows = 4
    fig, axes = plt.subplots(nrows=no_rows, figsize=(8,4*no_rows), sharex=True)
    
    ax_titles = ['lin', 'log', 'COM', 'exact']
    
    for n in range(no_rows):
        ax = axes[n]
        f_m_ana = conc_per_mass_np(bins_mass_centers[n], *dist_par)
        # ax.plot(bins_mass_centers[n], (f_m_num_sampled-f_m_ana)/f_m_ana, 'x')
        ax.errorbar(bins_mass_centers[n],
                    # bins_mass_width,
                    (f_m_num_avg-f_m_ana)/f_m_ana,
                    (f_m_num_std)/f_m_ana,
                    fmt = 'x' ,
                    c = 'k',
                    # c = 'lightblue',
                    markersize = 5.0,
                    linewidth =2.0,
                    capsize=3, elinewidth=2, markeredgewidth=1,
                    zorder=99)
        ax.set_xscale('log')
        ax.set_ylabel(r'$(f_{m,num}-f_{m}(\tilde{m}))/f_{m}(\tilde{m})$ ')
        ax.set_title(ax_titles[n])
    # axes[3].set_xlabel('bin width $Delta hat{m}$ (kg)')
    axes[3].set_xlabel('mass (kg)')
    
    for ax in axes:
        ax.grid()
    
    fig.tight_layout()
    fig_name = f'Deviations_fm_errorbars_sepa_plots_{sample_mode}_no_sims_{no_sims}'
    if sample_mode == 'given_bins': fig_name += '.png'
    elif sample_mode == 'auto_bins': fig_name += f'_no_bins_{no_bins}.png'
    fig.savefig(ensemble_dir + fig_name)

### 6b. ERRORBARS: DEVIATIONS of fm SEPA PLOTS AUTO BINS
    sample_mode = 'auto_bins'
    no_rows = 4
    fig, axes = plt.subplots(nrows=no_rows, figsize=(8,4*no_rows), sharex=True)
    
    ax_titles = ['lin', 'log', 'COM', 'exact']
    
    for n in range(no_rows):
        ax = axes[n]
        f_m_ana = conc_per_mass_np(bins_mass_centers_auto[n], *dist_par)
        # ax.plot(bins_mass_centers[n], (f_m_num_sampled-f_m_ana)/f_m_ana, 'x')
        ax.errorbar(bins_mass_centers_auto[n],
                    # bins_mass_width,
                    (f_m_num_avg_auto-f_m_ana)/f_m_ana,
                    (f_m_num_std_auto)/f_m_ana,
                    fmt = 'x' ,
                    c = 'k',
                    # c = 'lightblue',
                    markersize = 5.0,
                    linewidth =2.0,
                    capsize=3, elinewidth=2, markeredgewidth=1,
                    zorder=99)
        ax.set_xscale('log')
        ax.set_ylabel(r'$(f_{m,num}-f_{m}(\tilde{m}))/f_{m}(\tilde{m})$ ')
        ax.set_title(ax_titles[n])
    # axes[3].set_xlabel('bin width $Delta hat{m}$ (kg)')
    axes[3].set_xlabel('mass (kg)')
    
    for ax in axes:
        ax.grid()
    
    fig.tight_layout()
    fig_name = f'Deviations_fm_errorbars_sepa_plots_{sample_mode}_no_sims_{no_sims}'
    if sample_mode == 'given_bins': fig_name += '.png'
    elif sample_mode == 'auto_bins': fig_name += f'_no_bins_{no_bins}.png'
    fig.savefig(ensemble_dir + fig_name)

### 7. ERRORBARS: DEVIATIONS ALL IN ONE
    sample_mode = 'given_bins'
    no_rows = 2
    fig, axes = plt.subplots(nrows=no_rows, figsize=(10,5*no_rows), sharex=True)
    
    last_ind = 0
    # frac = 1.0
    frac = f_m_counts[0] / no_sims
    count_frac_limit = 0.1
    while frac > count_frac_limit and last_ind < len(f_m_counts)-2:
        last_ind += 1
        frac = f_m_counts[last_ind] / no_sims
    
    # exclude_ind_last = 3
    # last_ind = len(bins_mass_width)-exclude_ind_last
    
    # last_ind = len(bins_mass_centers[0])-16
    
    ax_titles = ['lin', 'log', 'COM', 'exact']
    
    # ax = axes
    ax = axes[0]
    for n in range(3):
        # ax = axes[n]
        f_m_ana = conc_per_mass_np(bins_mass_centers[n], *dist_par)
        ax.errorbar(bins_mass_centers[n][:last_ind],
                    100*(f_m_num_avg[:last_ind]-f_m_ana[:last_ind])\
                    /f_m_ana[:last_ind],
                    100*(f_m_num_std[:last_ind])/f_m_ana[:last_ind],
                    fmt = 'x' ,
                    # c = 'k',
                    # c = 'lightblue',
                    markersize = 10.0,
                    linewidth =2.0,
                    capsize=3, elinewidth=2, markeredgewidth=1,
                    label=ax_titles[n],
                    zorder=99)
    ax.legend()
    ax.set_ylabel(r'$(f_{m,num}-f_{m}(\tilde{m}))/f_{m}(\tilde{m})$ (%)')
    ax.set_xscale('log')
    # ax.set_yscale('symlog')
    # TT1 = np.array([-5,-4,-3,-2,-1,-0.5,-0.2,-0.1])
    # TT2 = np.array([-0.6,-0.2,-0.1])
    # TT1 = np.concatenate((np.append(TT2,0.0), -TT1) )
    # ax.yaxis.set_ticks(TT1)
    ax.grid()
    
    ax = axes[1]
    f_m_ana = conc_per_mass_np(bins_mass_centers[3], *dist_par)
    ax.errorbar(bins_mass_centers[3][:last_ind],
                # bins_mass_width[:last_ind],
                100*(f_m_num_avg[:last_ind]-f_m_ana[:last_ind])/f_m_ana[:last_ind],
                100*(f_m_num_std[:last_ind])/f_m_ana[:last_ind],
                fmt = 'x' ,
                # c = 'k',
                # c = 'lightblue',
                markersize = 10.0,
                linewidth =2.0,
                capsize=3, elinewidth=2, markeredgewidth=1,
                label=ax_titles[3],
                zorder=99)
    ax.set_ylabel(r'$(f_{m,num}-f_{m}(\tilde{m}))/f_{m}(\tilde{m})$ (%)')
    # ax.set_xlabel(r'mass $\tilde{m}$ (kg)')
    ax.set_xlabel(r'mass $m$ (kg)')
    ax.legend()
    # ax.set_xscale('log')
    # ax.set_yscale('symlog')
    # TT1 = np.array([-0.08,-0.04,-0.02,-0.01])
    # TT1 = np.array([-0.08,-0.04,-0.02,-0.01,-0.005])
    # TT2 = np.array([-0.6,-0.2,-0.1])
    # TT1 = np.concatenate((np.append(TT1,0.0), -TT1) )
    # ax.yaxis.set_ticks(100*TT1)
    # ax.set_ylim([-10.0,10.0])
    ax.set_xscale('log')
    ax.grid()
    
    fig.suptitle(
        f'kappa={kappa}, eta={eta}, r_critmin={r_critmin}, no_sims={no_sims}',
        y = 0.98)
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    fig_name = f'Deviations_fm_errorbars_{sample_mode}_no_sims_{no_sims}'
    if sample_mode == 'given_bins': fig_name += '.png'
    elif sample_mode == 'auto_bins': fig_name += f'_no_bins_{no_bins}.png'
    fig.savefig(ensemble_dir + fig_name)

### 8. MYHISTO BINNING DEVIATIONS of fm SEPA PLOTS
    no_rows = 7
    fig, axes = plt.subplots(nrows=no_rows, figsize=(8,4*no_rows), sharex=True)
    
    ax_titles = ['lin', 'log', 'COM', 'exact', 'linfit', 'qfit', 'h_over_g']
    
    for n in range(no_rows):
        ax = axes[n]
        f_m_ana = conc_per_mass_np(bins_mass_centers_my[n], *dist_par)
        # ax.plot(bins_mass_centers[n], (f_m_num_sampled-f_m_ana)/f_m_ana, 'x')
        ax.errorbar(bins_mass_centers_my[n],
                    # bins_mass_width,
                    (f_m_num_avg_my-f_m_ana)/f_m_ana,
                    f_m_num_std_my/f_m_ana,
                    fmt = 'x' ,
                    c = 'k',
                    # c = 'lightblue',
                    markersize = 5.0,
                    linewidth =2.0,
                    capsize=3, elinewidth=2, markeredgewidth=1,
                    zorder=99)
        ax.set_xscale('log')
        ax.set_ylabel(r'$(f_{m,num}-f_{m}(\tilde{m}))/f_{m}(\tilde{m})$ ')
        ax.set_title(ax_titles[n])
    # axes[3].set_xlabel('bin width $Delta hat{m}$ (kg)')
    axes[-1].set_xlabel('mass (kg)')
    
    for ax in axes:
        ax.grid()
    
    fig.tight_layout()
    fig_name = 'Deviations_fm_errorbars_myH_sepa_plots_no_sims_' \
               + f'{no_sims}_no_bins_{no_bins}.png'
    fig.savefig(ensemble_dir + fig_name)

### 9. MYHISTO BINNING DEVIATIONS PLOT ALL IN ONE
    no_rows = 2
    fig, axes = plt.subplots(nrows=no_rows, figsize=(10,5*no_rows), sharex=True)
    
    # last_ind = 0
    # # frac = 1.0
    # frac = f_m_counts[0] / no_sims
    # count_frac_limit = 0.1
    # while frac > count_frac_limit:
    #     last_ind += 1
    #     frac = f_m_counts[last_ind] / no_sims
    
    # exclude_ind_last = 3
    # last_ind = len(bins_mass_width)-exclude_ind_last
    
    last_ind = len(bins_mass_centers_my[0])
    
    ax_titles = ['lin', 'log', 'COM', 'exact']
    
    # ax = axes
    ax = axes[0]
    for n in range(3):
        # ax = axes[n]
        f_m_ana = conc_per_mass_np(bins_mass_centers_my[n], *dist_par)
        # ax.plot(bins_mass_centers[n], (f_m_num_sampled-f_m_ana)/f_m_ana, 'x')
        # ax.errorbar(bins_mass_width,
        #             100*((f_m_num_avg-f_m_ana)/f_m_ana)[0:3],
        #             100*(f_m_num_std/f_m_ana)[0:3],
        #             # 100*(f_m_num_avg[0:-3]-f_m_ana[0:-3])/f_m_ana[0:-3],
        #             # 100*f_m_num_std[0:-3]/f_m_ana[0:-3],
        #             fmt = 'x' ,
        #             # c = 'k',
        #             # c = 'lightblue',
        #             markersize = 10.0,
        #             linewidth =2.0,
        #             capsize=3, elinewidth=2, markeredgewidth=1,
        #             zorder=99)
        ax.errorbar(bins_mass_centers_my[n][:last_ind],
                    # bins_mass_width[:last_ind],
                    100*(f_m_num_avg_my[:last_ind]-f_m_ana[:last_ind])\
                    /f_m_ana[:last_ind],
                    100*(f_m_num_std_my[:last_ind])/f_m_ana[:last_ind],
                    fmt = 'x' ,
                    # c = 'k',
                    # c = 'lightblue',
                    markersize = 10.0,
                    linewidth =2.0,
                    capsize=3, elinewidth=2, markeredgewidth=1,
                    label=ax_titles[n],
                    zorder=99)
    ax.legend()
    ax.set_ylabel(r'$(f_{m,num}-f_{m}(\tilde{m}))/f_{m}(\tilde{m})$ (%)')
    ax.set_xscale('log')
    # ax.set_yscale('symlog')
    # TT1 = np.array([-5,-4,-3,-2,-1,-0.5,-0.2,-0.1])
    # TT2 = np.array([-0.6,-0.2,-0.1])
    # TT1 = np.concatenate((np.append(TT2,0.0), -TT1) )
    # ax.yaxis.set_ticks(TT1)
    ax.grid()
    
    ax = axes[1]
    f_m_ana = conc_per_mass_np(bins_mass_centers_my[3], *dist_par)
    ax.errorbar(bins_mass_centers_my[3][:last_ind],
                # bins_mass_width[:last_ind],
                100*(f_m_num_avg_my[:last_ind]-f_m_ana[:last_ind])/f_m_ana[:last_ind],
                100*(f_m_num_std_my[:last_ind])/f_m_ana[:last_ind],
                fmt = 'x' ,
                # c = 'k',
                # c = 'lightblue',
                markersize = 10.0,
                linewidth =2.0,
                capsize=3, elinewidth=2, markeredgewidth=1,
                label=ax_titles[3],
                zorder=99)
    ax.set_ylabel(r'$(f_{m,num}-f_{m}(\tilde{m}))/f_{m}(\tilde{m})$ (%)')
    # ax.set_xlabel(r'mass $\tilde{m}$ (kg)')
    ax.set_xlabel(r'mass $m$ (kg)')
    ax.legend()
    # ax.set_xscale('log')
    # ax.set_yscale('symlog')
    # TT1 = np.array([-0.08,-0.04,-0.02,-0.01])
    # TT1 = np.array([-0.08,-0.04,-0.02,-0.01,-0.005])
    # TT2 = np.array([-0.6,-0.2,-0.1])
    # TT1 = np.concatenate((np.append(TT1,0.0), -TT1) )
    # ax.yaxis.set_ticks(100*TT1)
    # ax.set_ylim([-10.0,10.0])
    ax.set_xscale('log')
    
    ax.grid()
    
    fig.suptitle(
        f'kappa={kappa}, eta={eta}, r_critmin={r_critmin}, no_sims={no_sims}',
        y = 0.98)
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    fig_name = 'Deviations_fm_errorbars_myH_no_sims_' \
               + f'{no_sims}_no_bins_{no_bins}.png'
    fig.savefig(ensemble_dir + fig_name)

    plt.close('all')        



