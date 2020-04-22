#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TROPOS LAGRANGIAN CLOUD MODEL
Super-Droplet method in two-dimensional kinematic framework
(Test Case 1 ICMW 2012, Muhlbauer et al. 2013)
Author: Jan Bohrer (bohrer@tropos.de)
Further contact: Oswald Knoth (knoth@tropos.de)

PLOTTING FUNCTIONS FOR THE GMD PUBLICATION

plotting functions for the collision box model can be found in
"collision.box_model.py"

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

# matplotlib rcParams dict. for pdf backend -> set mpl.use('pdf')    
pdf_dict = {
    'text.usetex': True,
#    'pgf.rcfonts': False,   # Do not set up fonts from rc parameters.
    'text.latex.preamble': [
        r'\usepackage[ttscale=.9]{libertine}',
        r'\usepackage[libertine]{newtxmath}',
        r'\usepackage[T1]{fontenc}',
        r'\usepackage[]{siunitx}',        
#        r'\usepackage[no-math]{fontspec}',
        ],
    'font.family': 'serif'
}

# matplotlib rcParams dict. for pgf backend -> set mpl.use('pgf')
pgf_dict = {
    'text.usetex': True,
#    'pgf.rcfonts': False,   # Do not set up fonts from rc parameters.
    'pgf.texsystem': 'pdflatex',
    'pgf.preamble': [
        r'\usepackage[T1]{fontenc}',
        r'\usepackage[]{siunitx}',
        ],
    'font.family': 'serif'
}

# matplotlib rcParams dict. for pgf backend -> set mpl.use('pgf')
pgf_dict_libertine = {
    'text.usetex': True,
#    'pgf.rcfonts': False,   # Do not set up fonts from rc parameters.
    'pgf.texsystem': 'pdflatex',
    'pgf.preamble': [
        r'\usepackage[ttscale=.9]{libertine}',
        r'\usepackage[libertine]{newtxmath}',
        r'\usepackage[T1]{fontenc}',
        r'\usepackage[]{siunitx}',
        ],
    'font.family': 'serif'
}

# matplotlib rcParams dict. for pgf backend -> set mpl.use('pgf')    
pgf_dict_sans = {
    'text.usetex': True,
#    'pgf.rcfonts': False,   # Do not set up fonts from rc parameters.
    'pgf.texsystem': 'pdflatex',
    'pgf.preamble': [
        r'\usepackage[ttscale=.9]{libertine}',
        r'\usepackage[libertine]{newtxmath}',
        r'\usepackage[T1]{fontenc}',
        r'\usepackage[]{siunitx}',
        r'\usepackage[]{sfmath}',
        ],
    'font.family': 'sans'
}

def generate_rcParams_dict(LW, MS, TTFS, LFS, TKFS, DPI):
    """Generation of a simple matplotlib rcParams dictionary
    
    Can be used by matplotlib.rcParams.update(dict_).

    Parameters
    ----------
    LW : float
        Line width
    MS : float
        Marker size
    TTFS : TYPE
        Title fontsize
    LFS : TYPE
        Label fontsize
    TKFS : TYPE
        Tick-label fontsize
    DPI : TYPE
        DPI when saving

    Returns
    -------
    dict_ : dict
        

    """
    
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

def simple_plot(x, y_arr):
    """Plots a number of y-arrays versus the same x-array

    Parameters
    ----------
    x : ndarray
        1D array of x-values
    y_arr : list of ndarray
        List of 1D arrays [y0, y1, y2, ...]. Each y_i is plotted vs. x in
        the same plot

    Returns
    -------
    None.

    """
    
    fig = plt.figure()
    ax = plt.gca()
    for y in y_arr:
        ax.plot (x, y)
    ax.grid()

def plot_scalar_field_2D(grid_centers_x_, grid_centers_y_, field_,
                         tick_ranges_, no_ticks_=[5,5],
                         no_contour_colors_ = 10, no_contour_lines_ = 5,
                         colorbar_fraction_=0.046, colorbar_pad_ = 0.02):
    """Generates a contour plot with countour lines for a given field(x,y)

    Parameters
    ----------
    grid_centers_x_ : ndarray
        2D array with the x-positions corresponding to 'field_'
    grid_centers_y_ : ndarray
        2D array with the y-positions corresponding to 'field_'
    field_ : ndarray
        2D array with the field values at the positions given by 
        'grid_centers_x_' and 'grid_centers_y_'
    tick_ranges_ : ndarray
        Axes ticks are displayed between
        x_min = tick_ranges_[0,0], x_max = tick_ranges_[0,1]
        y_min = tick_ranges_[1,0], y_max = tick_ranges_[1,1]
    no_ticks_ : ndarray, optional
        Number of ticks on x-axis = no_ticks_[0].
        Number of ticks on y-axis = no_ticks_[1].
        The default is [5,5].
    no_contour_colors_ : int, optional
        Number of levels for the contour plot. The default is 10.
    no_contour_lines_ : int, optional
        Number of contour lines. The default is 5.
    colorbar_fraction_ : TYPE, optional
        Fraction for the pyplot colorbar. The default is 0.046.
    colorbar_pad_ : TYPE, optional
        Padding for the pyplot colorbar. The default is 0.02.

    Returns
    -------
    None.

    """
    
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
    
#%% PLOT SIZE SPECTRA

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
                           TTFS=12, LFS=10, TKFS=10, LW = 4.0, MS = 3.0,
                           figsize_spectra = None,
                           figsize_tg_cells = None,
                           fig_path = None,
                           show_target_cells = False,
                           fig_path_tg_cells = None,
                           fig_path_R_eff = None,
                           trajectory = None,
                           show_textbox = True,
                           xshift_textbox = 0.5,
                           yshift_textbox = -0.35):
    """Generates plots for wet and dry size spectra at target cells
    
    Plots are saved to hard disk for dry and wet size spectra as well as
    the effective radius in each target cell. Optionally, one can save a
    plot with the visualization of the target cell and one provided
    particle trajectory.
    Data for this plotting function can be generated with 
    'evaluation.generate_size_spectra_R'
    
    Parameters
    ----------
    f_R_p_list: ndarray, dtype=float
        f_R_p_list[n,k] = 1D array with the discretized radius distribution
        f_R_p(R) for 'target cell' index 'n' and random seed index 'k'
        (f_R_p: here, number of particles per dry air mass and particle
         radius in 1/(mg*mu))
    f_R_s_list: ndarray, dtype=float
        f_R_s_list[n,k] = 1D array with the discret. dry radius distribution
        f_R_s(R) for 'target cell' index 'n' and random seed index 'k'
        (f_R_s: here, number of particles per dry air mass and dry particle
         radius in 1/(mg*mu))
    bins_R_p_list: ndarray, dtype=float
        bins_R_p_list[n] = 1D array with the particle radius bin borders for
        'target cell' index 'n'
    bins_R_s_list: ndarray, dtype=float
        bins_R_s_list[n] = 1D array with the particle dry radius bin borders
        for 'target cell' index 'n'
    grid_r_l_list: ndarray, dtype=float
        grid_r_l_list[n] = 2D array holding the discret. liquid water mixing
        ratio at the simulation time, when the spectrum of target cell 'n'
        is evaluated
    R_min_list: list of float
        R_min_list[n] = Minimum particle radius in the evaluation volume
        around target cell 'n' over all independent simulation seeds
    R_max_list: list of float
        R_max_list[n] = Maximum particle radius in the evaluation volume
        around target cell 'n' over all independent simulation seeds
    save_times_out: ndarray, dtype=float
        save_times_out[n] = simulation time, at which the spectrum at target
        cell 'n' is evaluated (seconds)
    solute_type: str
        Particle solute material.
        Either 'AS' (ammonium sulfate) or 'NaCl' (sodium chloride)
    grid: :obj:`Grid`
        Grid-class object, holding the spatial grid parameters.
    target_cell_list: ndarray, dtype=int
        Collects all target cells, for which size spectrums were evaluated.
        target_cell_list[n] =
        1D array holding two indices of the cell, which is the center of
        the evaluated volume of the grid
        target_cell_list[n][0] = horizontal index of the grid cell
        target_cell_list[n][1] = vertical index of the grid cell
    no_cells_x: int
        A volume of no_cells_x * no_cells_z grid cells centered at
        'target_cell' is included in the evaluation.
        Provide as uneven integer.
    no_cells_z: int
        A volume of no_cells_x * no_cells_z grid cells centered at
        'target_cell' is included in the evaluation.
        Provide as uneven integer.
    no_bins_R_p: int
        Number of bins for the particle radius histograms
    no_bins_R_s: int
        Number of bins for the particle dry radius histograms
    no_rows: int
        Number of rows for the subplot-grid
    no_cols: int
        Number of columns for the subplot-grid
    TTFS : TYPE
        Title fontsize
    LFS : TYPE
        Label fontsize
    TKFS : TYPE
        Tick-label fontsize
    LW : float
        Line width
    MS : float
        Marker size
    figsize_spectra: tuple
        Figsize of the spectra plot (x, y). unit = inch
    figsize_tg_cells: tuple
        Figsize of the plot for the target cell visualization (x, y).
        unit = inch
    fig_path: str
        Total path to the spectra plot, including the name.
        E.g. '/path/to/plot/name.pdf'
    show_target_cells: bool
        Activate to generate an additional plot, which visualizes the target
        cells.
    fig_path_tg_cells:
        Total path to the target-cell plot, including the name.
        E.g. '/path/to/plot/name.pdf'
    fig_path_R_eff:
        Total path to the plot for the effective radius, including the name.
        E.g. '/path/to/plot/name.pdf'
    trajectory: ndarray
        Particle trajectory
        trajectory[it,0] = x-position at time 'it'
        trajectory[it,1] = y-position at time 'it'
    show_textbox: bool
        Show a 'custom' legend as textbox
    xshift_textbox: float
        Position adjustment for the textbox
    yshift_textbox: float
        Position adjustment for the textbox
    
    """
    
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
            ax.annotate(f'\\textbf{{{annotations[plot_n]}}}',
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
                ax.set_xlabel(r'$R_\mathrm{p}$ (\si{\micro\meter})',
                              fontsize = LFS)
            if col_n==0:
                ax.set_ylabel(
                    '$f_\mathrm{R}$ $(\si{\micro\meter^{-1}\,mg^{-1}})$',
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
        
        if figsize_tg_cells is not None:
            figsize = figsize_tg_cells
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
                ax.annotate(f'\\textbf{{{annotations[tg_cell_n]}}}',
                            (x-x_ann_shift,z-z_ann_shift),
                            fontsize=ANS, zorder=100,
                            )
            else:
                x_ann_shift = 60E-3
                z_ann_shift = 60E-3
                ax.annotate(f'\\textbf{{{annotations[tg_cell_n]}}}',
                            (x-x_ann_shift,z-z_ann_shift),
                            fontsize=8, zorder=100,
                            )
            textbox.append(f'\\textbf{{{annotations[tg_cell_n]}}}: '
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

#%% PLOT SCALAR FIELDS
def plot_scalar_field_frames_avg(grid, fields_with_time,
                                 save_times,
                                 field_names,
                                 units,
                                 scales,
                                 solute_type,
                                 fig_path,
                                 figsize,
                                 no_ticks=[6,6],
                                 alpha = 1.0,
                                 TTFS = 12, LFS = 10, TKFS = 10,
                                 show_target_cells = False,
                                 target_cell_list = None,
                                 show_target_cell_labels = False,
                                 no_cells_x = 0,
                                 no_cells_z = 0
                                 ):
    """
    
    Data for this plotting function can be generated with
    'evaluation.generate_size_spectra_R'
    
    Parameters
    ----------
    grid: :obj:`Grid`
        Grid-class object, holding (among others) the spatial grid parameters.
    fields_with_time: ndarray, dtype=float
        fields_with_time[it,n] = 2D array with the average over
        independent simulation runs of analyzed field 'n' at
        the time corresponding to index 'it'
    save_times: ndarray, dtype=float
        1D array of simulation times, where the fields where analyzed
    field_names: list of str
        List of strings with the names of the analyzed fields used for
        plotting.
    units: list of str
        List of strings with the names of the units of the analyzed
        fields used for plotting.
    scales: list of float
        List of scaling factors for the analyzed fields. The scaling
        factors are used for plotting to obtain appropriate units.
    solute_type: str
        Particle solute material.
        Either 'AS' (ammonium sulfate) or 'NaCl' (sodium chloride)
    fig_path: str
        Total path to the plot, including the name.
        E.g. '/path/to/plot/name.pdf'
    figsize: tuple
        Figsize of the plot (x, y). unit = inch
    no_ticks: list of int
        no_ticks[0]: Number of ticks on the x-axis
        no_ticks[1]: Number of ticks on the y-axis
    alpha: float
        'alpha' value of the colormesh plot
    TTFS : TYPE
        Title fontsize
    LFS : TYPE
        Label fontsize
    TKFS : TYPE
        Tick-label fontsize
    LW : float
        Line width
    show_target_cells: bool
        Activate to visualize the target cells as boxes in the grid plots.
    target_cell_list: ndarray, dtype=int
        Collects all target cells, for which size spectrums were evaluated.
        target_cell_list[n] =
        1D array holding two indices of the cell, which is the center of
        the evaluated volume of the grid
        target_cell_list[n][0] = horizontal index of the grid cell
        target_cell_list[n][1] = vertical index of the grid cell
    show_target_cell_labels: bool
        Activate to show the annotation of the target cells 'a', 'b', 'c', ..
    no_cells_x: int
        A volume of no_cells_x * no_cells_z grid cells centered at
        'target_cell' is included in the evaluation.
        Provide as uneven integer.
    no_cells_z: int
        A volume of no_cells_x * no_cells_z grid cells centered at
        'target_cell' is included in the evaluation.
        Provide as uneven integer.
    
    """

    print("Chosen atmospheric fields:")
    for i,fm in enumerate(field_names):
        print(i,fm)

    print("Chosen save times")
    print("(if full sim., the times do not include spin up)")
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
            if ax_title in ['T', 'p', r'\Theta']:
                cmap = 'coolwarm'
                alpha = 1.0
            else :
                cmap = cmap_lcpp
                
            field_max = field.max()
            field_min = field.min()
            
            xticks_major = None
            xticks_minor = None

            norm_ = mpl.colors.Normalize
            #and field_max > 1E-2:
            if ax_title in ['r_\mathrm{r}', 'n_\mathrm{r}']:
                norm_ = mpl.colors.LogNorm
                field_min = 0.01
                cmap = cmap_lcpp                
                if ax_title == 'r_\mathrm{r}':
                    field_max = 1.
                    xticks_major = [0.01,0.1,1.]
                    xticks_minor = np.concatenate((
                            np.linspace(2E-2,1E-1,9),
                            np.linspace(2E-1,1,9),
                            ))
                elif ax_title == 'n_\mathrm{r}':
                    field_max = 10.
                    xticks_major = [0.01,0.1,1.,10.]
                    xticks_minor = np.concatenate((
                            np.linspace(2E-2,1E-1,9),
                            np.linspace(2E-1,1,9),
                            np.linspace(2,10,9),
                            ))
            else: norm_ = mpl.colors.Normalize   
            
            if ax_title == r'\Theta':
                field_min = 289.2
                field_max = 292.5
                xticks_major = [290,291,292]
            if ax_title == 'r_\mathrm{v}':
                field_min = 6.5
                field_max = 7.6
                xticks_minor = [6.75,7.25]
            if ax_title == 'r_\mathrm{l}':
                field_min = 0.0
                field_max = 1.3
                xticks_minor = [0.25,0.75,1.25]
                
            if ax_title == 'r_\mathrm{c}':
                field_min = 0.0
                field_max = 1.3
                xticks_major = np.linspace(0,1.2,7)
            if ax_title == 'n_\mathrm{c}':
                field_min = 0.0
                field_max = 150.
                xticks_major = [0,50,100,150]
                xticks_minor = [25,75,125]                
            if ax_title == 'n_\mathrm{aero}':
                field_min = 0.0
                field_max = 150.
                xticks_major = [0,50,100,150]
            if ax_title in [r'R_\mathrm{avg}', r'R_{2/1}', r'R_\mathrm{eff}']:
                xticks_major = [1,5,10,15,20]
                field_min = 1
                field_max = 20.
                # Arabas 2015
                cmap = cmap_lcpp
                unit = r'\si{\micro\meter}'
                
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
            if ax_title == r'\Theta':
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
                if ax_title == 'n_\mathrm{c}':
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
                                f'\\textbf{{{annotations[tg_cell_n]}}}',
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
    
#%% UNUSED PLOTTING FUNCTIONS

#%% PLOT ABSOLUTE DEVIATIONS OF TWO SCALAR FIELDS
# def plot_scalar_field_frames_abs_dev_MA(grid,
#                                         fields_with_time1,
#                                         fields_with_time_std1,
#                                         fields_with_time2,
#                                         fields_with_time_std2,
#                                         save_times,
#                                         field_names,
#                                         units,
#                                         scales,
#                                         solute_type,
#                                         simulation_mode, # for time in label
#                                         compare_type,
#                                         fig_path,
#                                         fig_path_abs_err,
#                                         figsize,
#                                         no_ticks=[6,6],
#                                         alpha = 1.0,
#                                         TTFS = 12, LFS = 10, TKFS = 10,
#                                         cbar_precision = 2,
#                                         show_target_cells = False,
#                                         target_cell_list = None,
#                                         no_cells_x = 0,
#                                         no_cells_z = 0
#                                         ):
    
    
#     for i,fm in enumerate(field_names):
#         print(i,fm)
#     print(save_times)
    
#     tick_ranges = grid.ranges

#     no_rows = len(save_times)
#     no_cols = len(field_names)
    
#     abs_dev_with_time = fields_with_time2 - fields_with_time1
    
#     fig, axes = plt.subplots(nrows=no_rows, ncols=no_cols,
#                        figsize = figsize,
#                        sharex=True, sharey=True)
    
#     vline_pos = (0.33, 1.17, 1.33)
    
#     for time_n in range(no_rows):
#         for field_n in range(no_cols):
#             ax = axes[time_n, field_n]
            
#             if compare_type == 'Nsip' and time_n == 2:
#                 for vline_pos_ in vline_pos:
#                     ax.axvline(vline_pos_, alpha=0.5, c='k',
#                                zorder= 3, linewidth = 1.3)
            
#             field = abs_dev_with_time[time_n, field_n] * scales[field_n]
#             ax_title = field_names[field_n]

#             print(time_n, ax_title, field.min(), field.max())

#             unit = units[field_n]
#             if ax_title in ['T','p',r'Theta']:
#                 cmap = 'coolwarm'
#                 alpha = 1.0
#             else :
#                 cmap = cmap_lcpp
                
#             field_max = field.max()
#             field_min = field.min()
            
#             xticks_major = None
#             xticks_minor = None
            
#             norm_ = mpl.colors.Normalize 

#             if compare_type in ['Ncell', 'solute', 'Kernel']:
#                 if ax_title in ['r_r', 'n_r']: #and field_max > 1E-2:
#                     norm_ = mpl.colors.Normalize
#                     cmap = cmap_lcpp                
#                     if ax_title == 'r_r':
#                         field_max = 0.1
#                         field_min = -field_max
#                         linthresh=0.01
#                         xticks_major = np.linspace(field_min,field_max,5)
#                     elif ax_title == 'n_r':
#                         pass
#                         field_max = 0.9
#                         field_min = -field_max
#                 else: norm_ = mpl.colors.Normalize   
                
#                 if ax_title == r'Theta':
#                     field_min = 289.2
#                     field_max = 292.5
#                     xticks_major = [290,291,292]
#                 if ax_title == 'r_v':
#                     field_min = 6.5
#                     field_max = 7.6
#                     xticks_minor = [6.75,7.25]
#                 if ax_title == 'r_l':
#                     field_min = 0.0
#                     field_max = 1.3
#                     xticks_minor = [0.25,0.75,1.25]
                    
#                 if ax_title == 'r_c':
#                     field_max = 0.3
#                     field_min = -field_max                
#                     xticks_major = np.linspace(field_min,field_max,5)
#                 if ax_title == 'n_c':
#                     field_max = 40.
#                     field_min = -field_max
#                 if ax_title == 'n_mathrm{aero}':
#                     field_max = 40.
#                     field_min = -field_max
#                     xticks_major = np.linspace(field_min,field_max,5)
#                 if ax_title in [r'R_mathrm{avg}',
#                                 r'R_{2/1}', r'R_mathrm{eff}']:
#                     field_max = 8.
#                     field_min = -field_max
#                     xticks_major = np.linspace(field_min,field_max,5)
#                     # Arabas 2015
#                     cmap = cmap_lcpp
#                     unit = r'\si{\micro\meter}'
                    
#             elif compare_type == 'Nsip':
#                 if ax_title in ['r_r', 'n_r']: #and field_max > 1E-2:
#                     norm_ = mpl.colors.Normalize
#                     cmap = cmap_lcpp                
#                     if ax_title == 'r_r':
#                         field_max = 0.08
#                         field_min = -field_max
#                         linthresh=0.01
#                         xticks_major = [field_min, -0.04, 0., 0.04, field_max]
#                     elif ax_title == 'n_r':
#                         field_max = 10.
#                         xticks_major = [0.01,0.1,1.,10.]
#                         xticks_minor = np.concatenate((
#                                 np.linspace(2E-2,1E-1,9),
#                                 np.linspace(2E-1,1,9),
#                                 np.linspace(2,10,9),
#                                 ))
#                 else: norm_ = mpl.colors.Normalize   
                
#                 if ax_title == r'Theta':
#                     field_min = 289.2
#                     field_max = 292.5
#                     xticks_major = [290,291,292]
#                 if ax_title == 'r_v':
#                     field_min = 6.5
#                     field_max = 7.6
#                     xticks_minor = [6.75,7.25]
#                 if ax_title == 'r_l':
#                     field_min = 0.0
#                     field_max = 1.3
#                     xticks_minor = [0.25,0.75,1.25]
                    
#                 if ax_title == 'r_c':
#                     field_max = 0.12
#                     field_min = -field_max                
#                     xticks_major = [-0.1, -0.05, 0., 0.05, 0.1]
#                 if ax_title == 'n_c':
#                     field_min = 0.0
#                     field_max = 150.
#                 if ax_title == 'n_mathrm{aero}':
#                     field_max = 20.
#                     field_min = -field_max
#                     xticks_major = np.linspace(field_min,field_max,5)
#                 if ax_title in [r'R_mathrm{avg}',
#                                 r'R_{2/1}', r'R_mathrm{eff}']:
#                     field_max = 5.
#                     field_min = -field_max
#                     xticks_major = np.linspace(field_min,field_max,5)
#                     # Arabas 2015
#                     cmap = cmap_lcpp
#                     unit = r'\si{\micro\meter}'
#             else:
#                 if ax_title in ['r_r', 'n_r']: #and field_max > 1E-2:
#                     norm_ = mpl.colors.Normalize
#                     cmap = cmap_lcpp                
#                     if ax_title == 'r_r':
#                         field_max = 0.08
#                         field_min = -field_max
#                         linthresh=0.01
#                         xticks_major = [field_min, -0.04, 0., 0.04, field_max]
#                     elif ax_title == 'n_r':
#                         field_max = 10.
#                         xticks_major = [0.01,0.1,1.,10.]
#                         xticks_minor = np.concatenate((
#                                 np.linspace(2E-2,1E-1,9),
#                                 np.linspace(2E-1,1,9),
#                                 np.linspace(2,10,9),
#                                 ))
#                 else: norm_ = mpl.colors.Normalize   
                
#                 if ax_title == r'Theta':
#                     field_min = 289.2
#                     field_max = 292.5
#                     xticks_major = [290,291,292]
#                 if ax_title == 'r_v':
#                     field_min = 6.5
#                     field_max = 7.6
#                     xticks_minor = [6.75,7.25]
#                 if ax_title == 'r_l':
#                     field_min = 0.0
#                     field_max = 1.3
#                     xticks_minor = [0.25,0.75,1.25]
                    
#                 if ax_title == 'r_c':
#                     field_max = 0.1
#                     field_min = -field_max                
#                     xticks_major = [field_min, -0.05, 0., 0.05, field_max]
#                 if ax_title == 'n_c':
#                     field_min = 0.0
#                     field_max = 150.
#                 if ax_title == 'n_mathrm{aero}':
#                     field_max = 8.
#                     field_min = -field_max
#                     xticks_major = [field_min, -4, 0., 4, field_max]
#                 if ax_title in [r'R_mathrm{avg}',
#                                 r'R_{2/1}', r'R_mathrm{eff}']:
#                     field_max = 3.
#                     field_min = -field_max
#                     xticks_major = np.linspace(field_min,field_max,7)
#                     # Arabas 2015
#                     cmap = cmap_lcpp
#                     unit = r'\si{\micro\meter}'
                    
#             oom_max = int(math.log10(field_max))
            
#             my_format = False
#             oom_factor = 1.0
            
#             if oom_max ==2: str_format = '%.0f'
#             else: str_format = '%.2g'
            
#             cmap = 'bwr'
            
#             if False:
#                 norm = norm_(linthresh = linthresh,
#                              vmin=field_min, vmax=field_max)
#             else:
#                 norm = norm_(vmin=field_min, vmax=field_max)
#             CS = ax.pcolormesh(*grid.corners, field*oom_factor,
#                                cmap=cmap, alpha=alpha,
#                                 edgecolor='face', zorder=1,
#                                 norm = norm,
#                                 rasterized=True,
#                                 antialiased=True, linewidth=0.0
#                                 )
#             CS.cmap.set_under('blue')
#             CS.cmap.set_over('red')
            
#             ax.set_xticks( np.linspace( tick_ranges[0,0],
#                                              tick_ranges[0,1],
#                                              no_ticks[0] ) )
#             ax.set_yticks( np.linspace( tick_ranges[1,0],
#                                              tick_ranges[1,1],
#                                              no_ticks[1] ) )

#             ax.tick_params(axis='both', which='major', labelsize=TKFS,
#                            length = 3, width=1)
#             ax.grid(color='gray', linestyle='dashed', zorder = 2)
#             ax.set_aspect('equal')
#             if time_n == no_rows-1:
#                 xticks1 = ax.xaxis.get_major_ticks()
#                 xticks1[-1].label1.set_visible(False)
#                 ax.set_xlabel(r'$x$ (km)', fontsize = LFS)
#             if field_n == 0:            
#                 ax.set_ylabel(r'$z$ (km)', fontsize = LFS)
#             ax.set_title( r'$t$ = {0} min'.format(int(save_times[time_n]/60)),
#                          fontsize = TTFS)
#             if time_n == 0:
#                 axins = inset_axes(ax,
#                                    width='90%', # width=5% of parent_bbox width
#                                    height='8%',  # height
#                                    loc='lower center',
#                                    bbox_to_anchor=(0.0, 1.35, 1, 1),
#                                    bbox_transform=ax.transAxes,
#                                    borderpad=0,
#                                    )      
#                 cbar = plt.colorbar(CS, cax=axins,
#                                     format=mticker.FormatStrFormatter(
#                                             str_format),
#                                     orientation='horizontal'
#                                     )
                
#                 axins.xaxis.set_ticks_position('bottom')
#                 axins.tick_params(axis='x',direction='inout',which='both')
#                 axins.tick_params(axis='x', which='major', labelsize=TKFS,
#                                length = 7, width=1)                
#                 axins.tick_params(axis='x', which='minor', labelsize=TKFS,
#                                length = 5, width=0.5,bottom=True)                
                
#                 if xticks_major is not None:
#                     axins.xaxis.set_ticks(xticks_major)
#                 if xticks_minor is not None:
#                     axins.xaxis.set_ticks(xticks_minor, minor=True)
#                 axins.set_title(r'$Delta {0}$ ({1})'.format(ax_title, unit))

#                 if my_format:
#                     cbar.ax.text(field_min - (field_max-field_min),
#                                  field_max + (field_max-field_min)*0.01,
#                                  r'$\times,10^{{{}}}$'.format(oom_max),
#                                  va='bottom', ha='left', fontsize = TKFS)
#                 cbar.ax.tick_params(labelsize=TKFS)

#             if show_target_cells:
#                 ### add the target cells
#                 no_neigh_x = no_cells_x // 2
#                 no_neigh_z = no_cells_z // 2
#                 dx = grid.steps[0]
#                 dz = grid.steps[1]
                
#                 no_tg_cells = len(target_cell_list[0])
#                 LW_rect = .5
#                 for tg_cell_n in range(no_tg_cells):
#                     x = (target_cell_list[0, tg_cell_n]
#                          - no_neigh_x - 0.1) * dx
#                     z = (target_cell_list[1, tg_cell_n]
#                          - no_neigh_z - 0.1) * dz
                    
#                     rect = plt.Rectangle((x, z), dx*no_cells_x,dz*no_cells_z,
#                                          fill=False,
#                                          linewidth = LW_rect,
#                                          edgecolor='k',
#                                          zorder = 99)        
#                     ax.add_patch(rect)


#     pad_ax_h = 0.1     
#     pad_ax_v = 0.05
#     fig.subplots_adjust(hspace=pad_ax_h, wspace=pad_ax_v)
             
#     if fig_path is not None:
#         fig.savefig(fig_path,
#                     bbox_inches = 'tight',
#                     pad_inches = 0.05,
#                     dpi=600
#                     )     
#     plt.close('all')
    
#     ### PLOT ABS ERROR OF THE DIFFERENCE OF TWO FIELDS:
#     # Var = Var_1 + Var_2 (assuming no correlations)

#     print('plotting ABS ERROR of field1 - field2')
    
#     std1 = np.nan_to_num( fields_with_time_std1 )
#     std2 = np.nan_to_num( fields_with_time_std2 )
#     std = np.sqrt( std1**2 + std2**2 )

#     tick_ranges = grid.ranges
    
#     no_rows = len(save_times)
#     no_cols = len(field_names)
    
#     for time_n in range(no_rows):
#         for field_n in range(no_cols):
#             std[time_n, field_n] *= scales[field_n]

#     field_max_all = np.amax(std, axis=(0,2,3))
# #    field_min_all = np.amin(std, axis=(0,2,3))

#     pad_ax_h = 0.1     
#     pad_ax_v = 0.03

#     ### PLOT ABS ERROR  
#     fig2, axes = plt.subplots(nrows=no_rows, ncols=no_cols,
#                        figsize = figsize,
#                        sharex=True, sharey=True)
            
#     for time_n in range(no_rows):
#         for field_n in range(no_cols):
#             ax = axes[time_n,field_n]
#             field = std[time_n, field_n]
#             ax_title = field_names[field_n]
#             unit = units[field_n]
#             cmap = cmap_lcpp
            
#             field_min = 0.
#             field_max = field_max_all[field_n]

#             xticks_major = None
#             xticks_minor = None
            
#             if compare_type == 'Nsip':
#                 if ax_title == 'n_mathrm{aero}':
#                     xticks_minor = np.array((1.25,3.75))
#                 if ax_title == 'r_c':
#                     field_max = 0.06
#                     xticks_minor = np.linspace(0.01,0.05,3)
#                 if ax_title == 'r_r':
#                     xticks_minor = np.linspace(0.01,0.03,2)
#                 if ax_title in [r'R_mathrm{eff}']:
#                     xticks_major = np.linspace(0.,1.5,4)
            
#             if compare_type == 'dt_col':
#                 if ax_title == 'n_mathrm{aero}':
#                     xticks_minor = np.array((1.25, 3.75, 6.25))
#                 if ax_title == 'r_c':
#                     xticks_major = np.array((0.,0.02,0.04,0.06))
#                 if ax_title == 'r_r':
#                     xticks_minor = np.linspace(0.01,0.05,3)
#                 if ax_title in [r'R_mathrm{eff}']:
#                     xticks_major = np.linspace(0.,1.5,4)
            
#             norm_ = mpl.colors.Normalize 
                
#             str_format = '%.2g'
            
#             my_format = False
#             oom_factor = 1.0
# #            oom_max = oom = 1

#             CS = ax.pcolormesh(*grid.corners,
#                                field*oom_factor,
#                                cmap=cmap, alpha=alpha,
#                                 edgecolor='face', zorder=1,
#                                 norm = norm_(vmin=field_min, vmax=field_max),
#                                 rasterized=True,
#                                 antialiased=True, linewidth=0.0
#                                 )
#             CS.cmap.set_under('white')
            
#             ax.set_xticks( np.linspace( tick_ranges[0,0],
#                                              tick_ranges[0,1],
#                                              no_ticks[0] ) )
#             ax.set_yticks( np.linspace( tick_ranges[1,0],
#                                              tick_ranges[1,1],
#                                              no_ticks[1] ) )

#             ax.tick_params(axis='both', which='major', labelsize=TKFS,
#                            length = 3, width=1)
#             ax.grid(color='gray', linestyle='dashed', zorder = 2)
#             ax.set_aspect('equal')
#             if time_n == no_rows-1:
#                 xticks1 = ax.xaxis.get_major_ticks()
#                 xticks1[-1].label1.set_visible(False)
#                 ax.set_xlabel(r'$x$ (km)', fontsize = LFS)
#             if field_n == 0:            
#                 ax.set_ylabel(r'$z$ (km)', fontsize = LFS)
#             ax.set_title( r'$t$ = {0} min'.format(int(save_times[time_n]/60)),
#                          fontsize = TTFS)
#             if time_n == 0:
#                 axins = inset_axes(ax,
#                                    width='90%', # width=5% of parent_bbox width
#                                    height='8%', # height
#                                    loc='lower center',
#                                    bbox_to_anchor=(0.0, 1.35, 1, 1),
#                                    bbox_transform=ax.transAxes,
#                                    borderpad=0,
#                                    )      
#                 cbar = plt.colorbar(
#                            CS, cax=axins,
#                            format=mticker.FormatStrFormatter(str_format),
#                            orientation='horizontal'
#                            )
#                 axins.xaxis.set_ticks_position('bottom')
#                 axins.tick_params(axis='x',direction='inout',which='both')
#                 axins.tick_params(axis='x', which='major', labelsize=TKFS,
#                                length = 7, width=1)                
#                 axins.tick_params(axis='x', which='minor', labelsize=TKFS,
#                                length = 5, width=0.5,bottom=True)                
                
#                 if xticks_major is not None:
#                     axins.xaxis.set_ticks(xticks_major)
#                 if xticks_minor is not None:
#                     axins.xaxis.set_ticks(xticks_minor, minor=True)
#                 if ax_title in [r'R_mathrm{avg}', r'R_{2/1}',
#                                 r'R_mathrm{eff}']:
#                     unit = r'\si{\micro\meter}'                                            
#                 axins.set_title(r'${0}$ abs. error ({1})'.format(ax_title,
#                                                                  unit))
            
#                 # 'my_format' dos not work with log scale here!!
#                 if my_format:
#                     cbar.ax.text(1.0,1.0,
#                                  r'$\times,10^{{{}}}$'.format(oom_max),
#                                  va='bottom', ha='right', fontsize = TKFS,
#                                  transform=ax.transAxes)
#                 cbar.ax.tick_params(labelsize=TKFS)

#             if show_target_cells:
#                 ### add the target cells
#                 no_neigh_x = no_cells_x // 2
#                 no_neigh_z = no_cells_z // 2
#                 dx = grid.steps[0]
#                 dz = grid.steps[1]
                
#                 no_tg_cells = len(target_cell_list[0])
#                 LW_rect = .5
#                 for tg_cell_n in range(no_tg_cells):
#                     x = (target_cell_list[0, tg_cell_n]
#                          - no_neigh_x - 0.1) * dx
#                     z = (target_cell_list[1, tg_cell_n]
#                          - no_neigh_z - 0.1) * dz
                    
#                     rect = plt.Rectangle((x, z), dx*no_cells_x,dz*no_cells_z,
#                                          fill=False,
#                                          linewidth = LW_rect,
#                                          edgecolor='k',
#                                          zorder = 99)        
#                     ax.add_patch(rect)

#     fig2.subplots_adjust(hspace=pad_ax_h, wspace=pad_ax_v)

#     if fig_path_abs_err is not None:
#         fig2.savefig(fig_path_abs_err,
#                     bbox_inches = 'tight',
#                     pad_inches = 0.05,
#                     dpi=600
#                     )        
        
# #%% PARTICLE TRACKING        
# # traj = [ pos0, pos1, pos2, .. ]
# # where pos0 = [pos_x_0, pos_z_0] is pos at time0
# # where pos_x_0 = [x0, x1, x2, ...]
# # selection = [n1, n2, ...] --> take only these indic. from the list of traj !!!
# # title font size (pt)
# # TTFS = 10
# # # labelsize (pt)
# # LFS = 10
# # # ticksize (pt)
# # TKFS = 10        
# def plot_particle_trajectories(traj, grid, selection=None,
#                                no_ticks=[6,6], figsize=(8,8),
#                                MS=1.0, arrow_every=5,
#                                ARROW_SCALE=12,ARROW_WIDTH=0.005, 
#                                TTFS = 10, LFS=10,TKFS=10,fig_name=None,
#                                t_start=0, t_end=3600):
#     centered_u_field = ( grid.velocity[0][0:-1,0:-1]
#                          + grid.velocity[0][1:,0:-1] ) * 0.5
#     centered_w_field = ( grid.velocity[1][0:-1,0:-1]
#                          + grid.velocity[1][0:-1,1:] ) * 0.5
    
#     pos_x = grid.centers[0][arrow_every//2::arrow_every,
#                             arrow_every//2::arrow_every]
#     pos_z = grid.centers[1][arrow_every//2::arrow_every,
#                             arrow_every//2::arrow_every]
    
#     vel_x = centered_u_field[arrow_every//2::arrow_every,
#                              arrow_every//2::arrow_every]
#     vel_z = centered_w_field[arrow_every//2::arrow_every,
#                              arrow_every//2::arrow_every]
    
#     tick_ranges = grid.ranges
#     fig, ax = plt.subplots(figsize=figsize)
#     if traj[0,0].size == 1:
#         ax.plot(traj[:,0], traj[:,1] ,'o', markersize = MS)
#     else:        
#         if selection == None: selection=range(len(traj[0,0]))
#         for ID in selection:
#             x = traj[:,0,ID]
#             z = traj[:,1,ID]
#             ax.plot(x,z,'o', markersize = MS)
#     ax.quiver(pos_x, pos_z, vel_x, vel_z,
#               pivot = 'mid',
#               width = ARROW_WIDTH, scale = ARROW_SCALE, zorder=99 )
#     ax.set_xticks( np.linspace( tick_ranges[0,0], tick_ranges[0,1],
#                                 no_ticks[0] ) )
#     ax.set_yticks( np.linspace( tick_ranges[1,0], tick_ranges[1,1],
#                                 no_ticks[1] ) )
#     ax.tick_params(axis='both', which='major', labelsize=TKFS)
#     ax.set_xlabel('horizontal position (m)', fontsize = LFS)
#     ax.set_ylabel('vertical position (m)', fontsize = LFS)
#     ax.set_title(
#     'Air velocity field and arbitrary particle trajectories\nfrom $t = $'\
#     + str(t_start) + ' s to ' + str(t_end) + ' s',
#         fontsize = TTFS, y = 1.04)
#     ax.grid(color='gray', linestyle='dashed', zorder = 0)    
#     fig.tight_layout()
#     if fig_name is not None:
#         fig.savefig(fig_name)
        
# #%% PLOT PARTICLE POSITIONS AND VELOCITIES

# def plot_pos_vel_pt(pos, vel, grid,
#                     figsize=(8,8), no_ticks = [6,6],
#                     MS = 1.0, ARRSCALE=2, fig_name=None):
#     fig, ax = plt.subplots(figsize=figsize)
#     ax.plot(grid.corners[0], grid.corners[1], 'x', color='red', markersize=MS)
#     ax.plot(pos[0],pos[1], 'o', color='k', markersize=2*MS)
#     ax.quiver(*pos, *vel, scale=ARRSCALE, pivot='mid')
#     x_min = grid.ranges[0,0]
#     x_max = grid.ranges[0,1]
#     y_min = grid.ranges[1,0]
#     y_max = grid.ranges[1,1]
#     ax.set_xticks( np.linspace(x_min, x_max, no_ticks[0]) )
#     ax.set_yticks( np.linspace(y_min, y_max, no_ticks[1]) )
#     ax.set_xticks(grid.corners[0][:,0], minor = True)
#     ax.set_yticks(grid.corners[1][0,:], minor = True)
#     ax.grid()
#     fig.tight_layout()
#     if fig_name is not None:
#         fig.savefig(fig_name)
        
# # pos = [pos0, pos1, pos2, ..] where pos0[0] = [x0, x1, x2, x3, ..] etc
# def plot_pos_vel_pt_with_time(pos_data, vel_data, grid, save_times,
#                     figsize=(8,8), no_ticks = [6,6],
#                     MS = 1.0, ARRSCALE=2, fig_name=None):
#     no_rows = len(pos_data)
#     fig, axes = plt.subplots(nrows=no_rows, figsize=figsize)
#     for i,ax in enumerate(axes):
#         pos = pos_data[i]
#         vel = vel_data[i]
#         ax.plot(grid.corners[0], grid.corners[1], 'x', color='red',
#                 markersize=MS)
#         ax.plot(pos[0],pos[1], 'o', color='k', markersize=2*MS)
#         ax.quiver(*pos, *vel, scale=ARRSCALE, pivot='mid')
#         x_min = grid.ranges[0,0]
#         x_max = grid.ranges[0,1]
#         y_min = grid.ranges[1,0]
#         y_max = grid.ranges[1,1]
#         ax.set_xticks( np.linspace(x_min, x_max, no_ticks[0]) )
#         ax.set_yticks( np.linspace(y_min, y_max, no_ticks[1]) )
#         ax.set_xticks(grid.corners[0][:,0], minor = True)
#         ax.set_yticks(grid.corners[1][0,:], minor = True)
#         ax.grid()
#         ax.set_title('t = ' + str(save_times[i]) + ' s')
#         ax.set_xlabel('x (m)')
#         ax.set_xlabel('z (m)')
#     fig.tight_layout()
#     if fig_name is not None:
#         fig.savefig(fig_name)
