#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 15:17:55 2019

@author: jdesk
"""

from numba import njit
import numpy as np

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
    for func in functions:
        print(func+":")
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


