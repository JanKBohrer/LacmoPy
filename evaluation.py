#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 15:17:55 2019

@author: jdesk
"""

from numba import njit
import numpy as np
import matplotlib.pyplot as plt


#%% PLOT GRID SCALAR FIELDS WITH TIME
### 


def load_grid_scalar_fields(path, save_times):
    fields = []
    for t_ in save_times:
        filename = path + "grid_scalar_fields_t_" + str(int(t_)) + ".npy"
        fields_ = np.load(filename)
        fields.append(fields_)
    fields = np.array(fields)
    return fields

# fields = [fields_t0, fields_t1, ...]
# fields_ti =
# (grid.mixing_ratio_water_vapor, grid.mixing_ratio_water_liquid,
#  grid.potential_temperature, grid.temperature,
#  grid.pressure, grid.saturation)
# input:
# field_indices = (idx1, idx2, ...)  (tuple of int)
# time_indices = (idx1, idx2, ...)  (tuple of int) 
def plot_field_frames(grid, fields, save_times, field_indices, time_indices,
                      no_ticks, fig_path=None):
    
    no_rows = len(time_indices)
    no_cols = len(field_indices)
    
    field_names = ["r_v", "r_l", "Theta", "T", "p", "S"]
    scales = [1000, 1000, 1, 1, 0.01, 1]
    units = ["g/kg", "g/kg", "K", "K", "hPa", "-"]
    
    tick_ranges = grid.ranges
    
    fig, axes = plt.subplots(nrows=no_rows, ncols=no_cols,
                           figsize = (4*no_cols, 4*no_rows))
    for i in range(no_rows):
        for j in range(no_cols):
            ax = axes[i,j]
            idx_t = time_indices[i]
            idx_f = field_indices[j]
            # print("idx_t")
            # print(idx_t)
            # print("idx_f")
            # print(idx_f)
            field = fields[idx_t, idx_f]*scales[idx_f]
            field_min = field.min()
            field_max = field.max()
#                contours = ax[i,j].contour(grid_centers_x_, grid_centers_y_,
#                       field, no_contour_lines_, colors = 'black')
#                ax[i,j].clabel(contours, inline=True, fontsize=8)
            CS = ax.pcolorfast(*grid.corners, field, cmap='coolwarm',
                                    vmin=field_min, vmax=field_max)
            ax.set_title(
                field_names[idx_f] + ' (' + units[idx_f] + '), t = '
                + str(save_times[idx_t]) )
            ax.set_xticks( np.linspace( tick_ranges[0,0],
                                             tick_ranges[0,1],
                                             no_ticks[0] ) )
            ax.set_yticks( np.linspace( tick_ranges[1,0],
                                             tick_ranges[1,1],
                                             no_ticks[1] ) )
            fig.colorbar(CS, ax=ax)
          
    fig.tight_layout()
    if fig_path is not None:
        fig.savefig(fig_path)
    


#%% RUNTIME OF FUNCTIONS
# functions is list of strings,
# e.g. ["compute_r_l_grid_field", "compute_r_l_grid_field_np"]
# pars is string,
# e.g. "m_w, xi, cells, grid.mixing_ratio_water_liquid, grid.mass_dry_inv"
# rs is list of repeats (int)
# ns is number of exec per repeat (int)
# example:
# funcs = ["compute_r_l_grid_field_np", "compute_r_l_grid_field"]
# pars = "m_w, xi, cells, grid.mixing_ratio_water_liquid, grid.mass_dry_inv"
# rs = [5,5,5]
# ns = [100,10000,1000]
# compare_functions_run_time(funcs, pars, rs, ns)
def compare_functions_run_time(functions, pars, rs, ns, globals_=globals()):
    import timeit
    # import numpy as np
    # print (__name__)
    t = []
    for i,func in enumerate(functions):
        print(func + ": repeats =", rs[i], "no reps = ", ns[i])
    for i,func in enumerate(functions):
        statement = func + "(" + pars + ")"
        t_ = timeit.repeat(statement, repeat=rs[i],
                           number=ns[i], globals=globals_)
        t.append(t_)
        print("best = ", f"{min(t_)/ns[i]*1.0E6:.4}", "us;",
              "worst = ", f"{max(t_)/ns[i]*1.0E6:.4}", "us;",
              "mean =", f"{np.mean(t_)/ns[i]*1.0E6:.4}",
              "+-", f"{np.std(t_, ddof = 1)/ns[i]*1.0E6:.3}", "us" )

# @njit()
def sample_masses(m_w, m_s, xi, cells, target_cell, no_cells_x, no_cells_z):
    m_dry = []
    m_wat = []
    multi = []
    
    i_p = []
    j_p = []
    
    dx = no_cells_x // 2
    dz = no_cells_z // 2
    
    i_an = range(target_cell[0] - dx, target_cell[0] + dx + 1)
    j_an = range(target_cell[1] - dz, target_cell[1] + dz + 1)
    # print("cells.shape in sample masses")
    # print(cells.shape)
    
    for ID, m_s_ in enumerate(m_s):
        # print(ID)
        i = cells[0,ID]
        j = cells[1,ID]
        if i in i_an and j in j_an:
            m_dry.append(m_s_)
            m_wat.append(m_w[ID])
            multi.append(xi[ID])
            i_p.append(i)
            j_p.append(j)
    m_wat = np.array(m_wat)
    m_dry = np.array(m_dry)
    multi = np.array(multi)
    i = np.array(i)
    j = np.array(j)
    
    return m_wat, m_dry, multi, i, j

from microphysics import compute_radius_from_mass,\
                         compute_R_p_w_s_rho_p
import constants as c
# we always assume the only quantities stored are m_s, m_w, xi
def sample_radii(m_w, m_s, xi, cells, grid_temperature,
                 target_cell, no_cells_x, no_cells_z):
    m_wat, m_dry, multi, i, j = sample_masses(m_w, m_s, xi, cells,
                                        target_cell, no_cells_x, no_cells_z)
    # print("m_wat")
    # print("m_dry")
    # print("multi")
    # print(m_wat)
    # print(m_dry)
    # print(multi)
    R_s = compute_radius_from_mass(m_dry, c.mass_density_NaCl_dry)
    T_p = grid_temperature[i,j]
    R, w_s, rho_p = compute_R_p_w_s_rho_p(m_wat, m_dry, T_p)
    return R, R_s, multi


