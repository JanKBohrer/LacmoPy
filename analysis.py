#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 15:17:55 2019

@author: jdesk
"""

from numba import njit
import numpy as np
import matplotlib.pyplot as plt

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
# compare_functions_run_time(funcs, pars, rs, ns, globals_=globals())
# NOTE that we need to call with globals_=globals() explicitly
# a default argument for globals_ cannot be given in the function definition..
# because in that case, the globals are taken from module "analysis.py" and
# not from the environment of the executed program
def compare_functions_run_time(functions, pars, rs, ns, globals_):
    import timeit
    # import numpy as np
    # print (__name__)
    t = []
    for i,func in enumerate(functions):
        print(func + ": repeats =", rs[i], "no reps = ", ns[i])
    # print(globals_)
    for i,func in enumerate(functions):
        statement = func + "(" + pars + ")"
        t_ = timeit.repeat(statement, repeat=rs[i],
                           number=ns[i], globals=globals_)
        t.append(t_)
        print("best = ", f"{min(t_)/ns[i]*1.0E6:.4}", "us;",
              "worst = ", f"{max(t_)/ns[i]*1.0E6:.4}", "us;",
              "mean =", f"{np.mean(t_)/ns[i]*1.0E6:.4}",
              "+-", f"{np.std(t_, ddof = 1)/ns[i]*1.0E6:.3}", "us" )

#%% PARTICLE TRACKING

# traj = [ pos0, pos1, pos2, .. ]
# where pos0 = [pos_x_0, pos_z_0] is pos at time0
# where pos_x_0 = [x0, x1, x2, ...]
# selection = [n1, n2, ...] --> take only these indic. from the list of traj !!!
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
    
    # title font size (pt)
    # TTFS = 10
    # # labelsize (pt)
    # LFS = 10
    # # ticksize (pt)
    # TKFS = 10
    
    # ARROW_SCALE = 12
    # ARROW_WIDTH = 0.005
    # no_major_xticks = 6
    # no_major_yticks = 6
    # tick_every_x = grid.no_cells[0] // (no_major_xticks - 1)
    # tick_every_y = grid.no_cells[1] // (no_major_yticks - 1)
    
    # arrow_every = 5
    
    # part_every = 2
    # loc_every = 1
    
    pos_x = grid.centers[0][arrow_every//2::arrow_every,
                            arrow_every//2::arrow_every]
    pos_z = grid.centers[1][arrow_every//2::arrow_every,
                            arrow_every//2::arrow_every]
    # print(pos_x)
    # pos_x = grid.centers[0][arrow_every//2::arrow_every,
    #                         arrow_every//2::arrow_every]/1000
    # pos_z = grid.centers[1][arrow_every//2::arrow_every,
    #                         arrow_every//2::arrow_every]/1000
    
    # pos_x = grid.centers[0][::arrow_every,::arrow_every]/1000
    # pos_z = grid.centers[1][::arrow_every,::arrow_every]/1000
    
    # pos_x = grid.centers[0][30:40,5:15]
    # pos_z = grid.centers[1][30:40,5:15]
    # pos_x = grid.surface_centers[1][0][30:40,5:15]
    # pos_z = grid.surface_centers[1][1][30:40,5:15]
    
    vel_x = centered_u_field[arrow_every//2::arrow_every,
                             arrow_every//2::arrow_every]
    vel_z = centered_w_field[arrow_every//2::arrow_every,
                             arrow_every//2::arrow_every]
    
    tick_ranges = grid.ranges
    fig, ax = plt.subplots(figsize=figsize)
    if selection == None: selection=range(len(traj[0,0]))
    # print(selection)
    for ID in selection:
        x = traj[:,0,ID]
        z = traj[:,1,ID]
        ax.plot(x,z,"o", markersize = MS)
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
    + str(t_start) + " s to " + str(t_end) + " s",
        fontsize = TTFS, y = 1.04)
    ax.grid(color='gray', linestyle='dashed', zorder = 0)    
    # ax.grid()
    fig.tight_layout()
    if fig_name is not None:
        fig.savefig(fig_name)

#%% BINNING OF SIPs:

def auto_bin_SIPs(masses, xis, no_bins, dV, no_sims, xi_min=1):

    ind = np.nonzero(xis)
    m_sort = masses[ind]
    xi_sort = xis[ind]
    # print(m_sort.shape)
    # print(xi_sort.shape)
    # print()
    
    ind = np.argsort(m_sort)
    m_sort = m_sort[ind]
    xi_sort = xi_sort[ind]
    # print(m_sort.shape)
    # print(xi_sort.shape)
    
    # plt.plot(masses, "o")
    # plt.plot(m_sort, "o")
    # print(np.nonzero(xis))
    
    # plt.loglog(m_sort, xi_sort, "+")
    # plt.plot(m_sort, xi_sort, "+")
    
    ### merge particles with xi < xi_min
    for i in range(len(xi_sort)-1):
        if xi_sort[i] < xi_min:
            xi = xi_sort[i]
            m = m_sort[i]
            xi_left = 0
            j = i
            while(j > 0 and xi_left==0):
                j -= 1
                xi_left = xi_sort[j]
            if xi_left != 0:
                m1 = m_sort[j]
                dm_left = m-m1
            else:
                dm_left = 1.0E18
            m2 = m_sort[i+1]
            if m2-m < dm_left:
                j = i+1
                # assign to m1 since distance is smaller
                # i.e. increase number of xi[i-1],
                # then reweight mass to total mass
            m_sum = m*xi + m_sort[j]*xi_sort[j]
            # print("added pt ", i, "to", j)
            # print("xi_i bef=", xi)
            # print("xi_j bef=", xi_sort[j])
            # print("m_i bef=", m)
            # print("m_j bef=", m_sort[j])
            # xi_sort[j] += xi_sort[i]
            # m_sort[j] = m_sum / xi_sort[j]
            # print("xi_j aft=", xi_sort[j])
            # print("m_j aft=", m_sort[j])
            xi_sort[i] = 0           
    
    if xi_sort[-1] < xi_min:
        i = -1
        xi = xi_sort[i]
        m = m_sort[-1]
        xi_left = 0
        j = i
        while(xi_left==0):
            j -= 1
            xi_left = xi_sort[j]
        
        m_sum = m*xi + m_sort[j]*xi_sort[j]
        xi_sort[j] += xi_sort[i]
        m_sort[j] = m_sum / xi_sort[j]
        xi_sort[i] = 0           
    
    ind = np.nonzero(xi_sort)
    xi_sort = xi_sort[ind]
    m_sort = m_sort[ind]
    
    # print()
    # print("xis.min()")
    # print("xi_sort.min()")
    # print(xis.min())
    # print(xi_sort.min())
    # print()
    
    # print("xi_sort.shape, xi_sort.dtype")
    # print("m_sort.shape, m_sort.dtype")
    # print(xi_sort.shape, xi_sort.dtype)
    # print(m_sort.shape, m_sort.dtype)
    
    # plt.loglog(m_sort, xi_sort, "x")
    # plt.plot(m_sort, xi_sort, "x")
    
    # resort, if masses have changed "places" in the selection process
    ind = np.argsort(m_sort)
    m_sort = m_sort[ind]
    xi_sort = xi_sort[ind]
    
    ### merge particles, which have masses or xis < m_lim, xi_lim
    no_bins0 = no_bins
    no_bins *= 10
    # print("no_bins")
    # print("no_bins0")
    # print(no_bins)
    # print(no_bins0)
    
    no_spc = len(xi_sort)
    n_save = int(no_spc//1000)
    if n_save < 2: n_save = 2
    
    no_rpc = np.sum(xi_sort)
    total_mass = np.sum(xi_sort*m_sort)
    xi_lim = no_rpc / no_bins
    m_lim = total_mass / no_bins
    
    bin_centers = []
    m_bin = []
    xi_bin = []
    
    n_left = no_rpc
    
    i = 0
    while(n_left > 0 and i < len(xi_sort)-n_save):
        bin_mass = 0.0
        bin_xi = 0
        # bin_center = 0.0
        
        # while(bin_xi < n_lim):
        while(bin_mass < m_lim and bin_xi < xi_lim and n_left > 0
              and i < len(xi_sort)-n_save):
            bin_xi += xi_sort[i]
            bin_mass += xi_sort[i] * m_sort[i]
            n_left -= xi_sort[i]
            i += 1
        bin_centers.append(bin_mass / bin_xi)
        m_bin.append(bin_mass)
        xi_bin.append(bin_xi)
            
    # return m_bin, xi_bin, bin_centers    
    
    xi_bin = np.array(xi_bin)
    bin_centers = np.array(bin_centers)
    m_bin = np.array(m_bin)
    
    ### merge particles, whose masses are close together in log space:
    bin_size_log =\
        (np.log10(bin_centers[-1]) - np.log10(bin_centers[0])) / no_bins0
    
    # print("np.sum(xi_bin*bin_centers), m_bin.sum() before")
    # print(np.sum(xi_bin*bin_centers), m_bin.sum())
    
    i = 0
    while(i < len(xi_bin)-1):
        m_next_bin = bin_centers[i] * 10**bin_size_log
        m = bin_centers[i]
        j = i
        while (m < m_next_bin and j < len(xi_bin)-1):
            j += 1
            m = bin_centers[j]
        if m >= m_next_bin:
            j -= 1
        if (i != j):
            m_sum = 0.0
            xi_sum = 0
            for k in range(i,j+1):
                m_sum += m_bin[k]
                xi_sum += xi_bin[k]
                if k > i:
                    xi_bin[k] = 0
            bin_centers[i] = m_sum / xi_sum
            xi_bin[i] = xi_sum
            m_bin[i] = m_sum
        i = j+1            
    
    
    ind = np.nonzero(xi_bin)
    xi_bin = xi_bin[ind]
    bin_centers = bin_centers[ind]        
    m_bin = m_bin[ind]
    
    # print("np.sum(xi_bin*bin_centers), m_bin.sum() after")
    # print(np.sum(xi_bin*bin_centers), m_bin.sum())
    
    ######
    # bin_size = 0.5 * (bin_centers[-1] - bin_centers[0]) / no_bins0
    # bin_size = (bin_centers[-1] - bin_centers[0]) / no_bins0
    
    # print("len(bin_centers) bef =", len(bin_centers))
    
    # for i, bc in enumerate(bin_centers[:-1]):
    #     if bin_centers[i+1] - bc < bin_size and xi_bin[i] != 0:
    #         m_sum = m_bin[i+1] + m_bin[i]
    #         xi_sum = xi_bin[i+1] + xi_bin[i]
    #         bin_centers[i] = m_sum / xi_sum
    #         xi_bin[i] = xi_sum
    #         xi_bin[i+1] = 0
    #         m_bin[i] = m_sum

    # ind = np.nonzero(xi_bin)
    # xi_bin = xi_bin[ind]
    # m_bin = m_bin[ind]
    # bin_centers = bin_centers[ind]
    ######
    
    # print("len(bin_centers) after =", len(bin_centers))

    # radii = compute_radius_from_mass(m_sort, c.mass_density_water_liquid_NTP)
    radii = compute_radius_from_mass_vec(bin_centers,
                                     c.mass_density_water_liquid_NTP)
    
    ###
    # find the midpoints between the masses/radii
    # midpoints = 0.5 * ( m_sort[:-1] + m_sort[1:] )
    # m_left = 2.0 * m_sort[0] - midpoints[0]
    # m_right = 2.0 * m_sort[-1] - midpoints[-1]
    bins = 0.5 * ( radii[:-1] + radii[1:] )
    # add missing bin borders for m_min and m_max:
    R_left = 2.0 * radii[0] - bins[0]
    R_right = 2.0 * radii[-1] - bins[-1]
    
    bins = np.hstack([R_left, bins, R_right])
    bins_log = np.log(bins)
    # print(midpoints)
       
    # mass_per_ln_R = m_sort * xi_sort
    # mass_per_ln_R *= 1.0E-15/no_sims
    
    m_bin = np.array(m_bin)
    
    g_ln_R = m_bin * 1.0E-15 / no_sims / (bins_log[1:] - bins_log[0:-1]) / dV
    
    return g_ln_R, radii, bins, xi_bin, bin_centers

# masses = masses_vs_time[3]
# xis = xis_vs_time[3]
# radii = compute_radius_from_mass(masses, c.mass_density_water_liquid_NTP)
# xi_min = 100
# m_bin, xi_bin, bins = auto_bin_SIPs(masses, xis, xi_min)

# r_bin = compute_radius_from_mass(bins, c.mass_density_water_liquid_NTP)

# plt.plot(r_bin, m_bin, "o")

# masses = masses_vs_time[3]
# xis = xis_vs_time[3]
# radii = compute_radius_from_mass(masses, c.mass_density_water_liquid_NTP)
# print(masses.shape)
# print(xis.shape)
# print()

# xi_min = 100
# no_bins = 40
# g_ln_R, R_sort, bins, xi_bi, m_bins = auto_bin_SIPs(masses,
#                                                     xis, xi_min, no_bins,
#                                                     dV, no_sims)

# fig = plt.figure()
# ax = plt.gca()
# ax.loglog(R_sort, g_ln_R, "x")
# # ax.plot(R_sort, g_ln_R, "x")
# # ax.plot(R_sort, xi_bin)

# ###

# method = "log_R"

# R_min = 0.99*np.amin(radii)
# R_max = 1.01*np.amax(radii)
# # R_max = 3.0*np.amax(radii)
# print("R_min=", R_min)
# print("R_max=", R_max)

# no_bins = 20
# if method == "log_R":
#     bins = np.logspace(np.log10(R_min), np.log10(R_max), no_bins)
# elif method == "lin_R":
#     bins = np.linspace(R_min, R_max, no_bins)
# # print(bins)

# # masses in 10^-15 gram
# mass_per_ln_R, _ = np.histogram(radii, bins, weights=masses*xis)
# # convert to gram
# mass_per_ln_R *= 1.0E-15/no_sims
# # print(mass_per_ln_R)
# # print(mass_per_ln_R.shape, bins.shape)

# bins_log = np.log(bins)
# # bins_mid = np.exp((bins_log[1:] + bins_log[:-1]) * 0.5)
# bins_mid = (bins[1:] + bins[:-1]) * 0.5

# g_ln_R = mass_per_ln_R / (bins_log[1:] - bins_log[0:-1]) / dV

# # print(g_ln_R.shape)
# # print(np.log(bins_mid[1:])-np.log(bins_mid[0:-1]))
# ax.loglog( bins_mid, g_ln_R, "-" )
###

#%% PARTICLE SIZE SPECTRA

def plot_pos_vel_pt(pos, vel, grid,
                    figsize=(8,8), no_ticks = [6,6],
                    MS = 1.0, ARRSCALE=2, fig_name=None):
    # u_g = 0.5 * ( grid.velocity[0,0:-1] + grid.velocity[0,1:] )
    # v_g = 0.5 * ( grid.velocity[1,:,0:-1] + grid.velocity[1,:,1:] )
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(grid.corners[0], grid.corners[1], "x", color="red", markersize=MS)
    ax.plot(pos[0],pos[1], "o", color="k", markersize=2*MS)
    ax.quiver(*pos, *vel, scale=ARRSCALE, pivot="mid")
    # ax.quiver(*grid.centers, u_g[:,0:-1], v_g[0:-1],
              # scale=ARRSCALE, pivot="mid", color="red")
    # ax.quiver(grid.corners[0], grid.corners[1] + 0.5*grid.steps[1],
    #           grid.velocity[0], np.zeros_like(grid.velocity[0]),
    #           scale=0.5, pivot="mid", color="red")
    # ax.quiver(grid.corners[0] + 0.5*grid.steps[0], grid.corners[1],
    #           np.zeros_like(grid.velocity[1]), grid.velocity[1],
    #           scale=0.5, pivot="mid", color="blue")
    x_min = grid.ranges[0,0]
    x_max = grid.ranges[0,1]
    y_min = grid.ranges[1,0]
    y_max = grid.ranges[1,1]
    ax.set_xticks( np.linspace(x_min, x_max, no_ticks[0]) )
    ax.set_yticks( np.linspace(y_min, y_max, no_ticks[1]) )
    # ax.set_xticks(grid.corners[0][:,0])
    # ax.set_yticks(grid.corners[1][0,:])
    ax.set_xticks(grid.corners[0][:,0], minor = True)
    ax.set_yticks(grid.corners[1][0,:], minor = True)
    # plt.minorticks_off()
    # plt.minorticks_on()
    ax.grid()
    fig.tight_layout()
    # ax.grid(which="minor")
    # plt.show()
    if fig_name is not None:
        fig.savefig(fig_name)
        
# pos = [pos0, pos1, pos2, ..] where pos0[0] = [x0, x1, x2, x3, ..] etc
def plot_pos_vel_pt_with_time(pos_data, vel_data, grid, save_times,
                    figsize=(8,8), no_ticks = [6,6],
                    MS = 1.0, ARRSCALE=2, fig_name=None):
    # u_g = 0.5 * ( grid.velocity[0,0:-1] + grid.velocity[0,1:] )
    # v_g = 0.5 * ( grid.velocity[1,:,0:-1] + grid.velocity[1,:,1:] )
    no_rows = len(pos_data)
    fig, axes = plt.subplots(nrows=no_rows, figsize=figsize)
    for i,ax in enumerate(axes):
        pos = pos_data[i]
        vel = vel_data[i]
        ax.plot(grid.corners[0], grid.corners[1], "x", color="red",
                markersize=MS)
        ax.plot(pos[0],pos[1], "o", color="k", markersize=2*MS)
        ax.quiver(*pos, *vel, scale=ARRSCALE, pivot="mid")
        # ax.quiver(*grid.centers, u_g[:,0:-1], v_g[0:-1],
                  # scale=ARRSCALE, pivot="mid", color="red")
        # ax.quiver(grid.corners[0], grid.corners[1] + 0.5*grid.steps[1],
        #           grid.velocity[0], np.zeros_like(grid.velocity[0]),
        #           scale=0.5, pivot="mid", color="red")
        # ax.quiver(grid.corners[0] + 0.5*grid.steps[0], grid.corners[1],
        #           np.zeros_like(grid.velocity[1]), grid.velocity[1],
        #           scale=0.5, pivot="mid", color="blue")
        x_min = grid.ranges[0,0]
        x_max = grid.ranges[0,1]
        y_min = grid.ranges[1,0]
        y_max = grid.ranges[1,1]
        ax.set_xticks( np.linspace(x_min, x_max, no_ticks[0]) )
        ax.set_yticks( np.linspace(y_min, y_max, no_ticks[1]) )
        # ax.set_xticks(grid.corners[0][:,0])
        # ax.set_yticks(grid.corners[1][0,:])
        ax.set_xticks(grid.corners[0][:,0], minor = True)
        ax.set_yticks(grid.corners[1][0,:], minor = True)
        # plt.minorticks_off()
        # plt.minorticks_on()
        ax.grid()
        ax.set_title("t = " + str(save_times[i]) + " s")
        ax.set_xlabel('x (m)')
        ax.set_xlabel('z (m)')
    fig.tight_layout()
    # ax.grid(which="minor")
    # plt.show()
    if fig_name is not None:
        fig.savefig(fig_name)
        
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

from microphysics import compute_radius_from_mass_vec,\
                         compute_R_p_w_s_rho_p_NaCl,\
                         compute_R_p_w_s_rho_p_AS
import constants as c
# we always assume the only quantities stored are m_s, m_w, xi
def sample_radii(m_w, m_s, xi, cells, grid_temperature,
                 target_cell, no_cells_x, no_cells_z, solute_type):
    m_wat, m_dry, multi, i, j = sample_masses(m_w, m_s, xi, cells,
                                        target_cell, no_cells_x, no_cells_z)
    # print("m_wat")
    # print("m_dry")
    # print("multi")
    # print(m_wat)
    # print(m_dry)
    # print(multi)
    T_p = grid_temperature[i,j]
    if solute_type == "AS":
        mass_density_dry = c.mass_density_AS_dry
        R, w_s, rho_p = compute_R_p_w_s_rho_p_AS(m_wat, m_dry, T_p)
    elif solute_type == "NaCl":
        mass_density_dry = c.mass_density_AS_dry
        R, w_s, rho_p = compute_R_p_w_s_rho_p_NaCl(m_wat, m_dry, T_p)
    R_s = compute_radius_from_mass_vec(m_dry, mass_density_dry)
    return R, R_s, multi        

#%% PLOTTING

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
    
def plot_particle_size_spectra(m_w, m_s, xi, cells, grid, solute_type,
                          target_cell, no_cells_x, no_cells_z,
                          no_rows=1, no_cols=1, TTFS=12, LFS=10, TKFS=10,
                          fig_path = None):
    R, Rs, multi = sample_radii(m_w, m_s, xi, cells, grid.temperature,
                                    target_cell, no_cells_x, no_cells_z,
                                    solute_type)

    log_R = np.log10(R)
    log_Rs = np.log10(Rs)
    multi = np.array(multi)
    if no_cells_x % 2 == 0: no_cells_x += 1
    if no_cells_z % 2 == 0: no_cells_z += 1
    
    V_an = no_cells_x * no_cells_z * grid.volume_cell * 1.0E6
    # V_an = (no_neighbors_x * 2 + 1) * (no_neighbors_z * 2 + 1)
    # * grid.volume_cell * 1.0E6
    
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
    # h1, bins1 = np.histogram(log_R, bins=no_bins,
    #                          range=(R_min_log,R_max_log), weights=multi/V_an)
    # h2, bins2 = np.histogram( log_Rs, bins=no_bins_s,
    #                           range=(R_min_log, Rs_max_log),
    #                           weights=multi/V_an )
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
                           # figsize = (figsize_x*no_cols, figsize_y*no_rows),
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
    ax.plot(np.repeat(bins1,2)[:],
            np.hstack( [[0.001], np.repeat(h1,2)[:], [0.001] ] ),
            linewidth = LW, zorder = 6, label = "wet")
    ax.plot(np.repeat(bins2,2)[:],
            np.hstack( [[0.001], np.repeat(h2,2)[:], [0.001] ] ),
            linewidth = LW, label = "dry")
    
    
    ax.tick_params(axis='both', which='major', labelsize=TKFS, length = 5)
    ax.tick_params(axis='both', which='minor', labelsize=TKFS, length = 3)
    
    height = int(grid.compute_location(*target_cell,0.0,0.0)[1])
    ax.set_xlabel(r"particle radius (mu)", fontsize = LFS)
    ax.set_ylabel(r"concentration (${\mathrm{cm}^{3}}$)", fontsize = LFS)
    # ax.set_xlabel(r"particle radius ($\si{\micro m}$)", fontsize = LFS)
    # ax.set_ylabel(r"concentration ($\si{\# / cm^{3}}$)", fontsize = LFS)
    ax.set_title( f'h = {height} m ' +
                 f"tg cell ({target_cell[0]} {target_cell[1]}) "
                    + f"no cells ({no_cells_x}, {no_cells_z})",
                    fontsize = TTFS )
    
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
#        fig.savefig(fig_path + "spectrum_" + str(height) +".pdf")
        fig.savefig(fig_path
                    + f"spectrum_cell_{target_cell[0]}_{target_cell[1]}_"
                    + f"no_cells_{no_cells_x}_{no_cells_z}.pdf")

def plot_particle_size_spectra_tg_list(
        t, m_w, m_s, xi, cells,
        grid_volume_cell,
        grid_step_z,
        grid_temperature,
        solute_type,
        target_cell_list, no_cells_x, no_cells_z,
        no_rows, no_cols,
        TTFS=12, LFS=10, TKFS=10,
        fig_path = None):
    
    i_list = target_cell_list[0]
    j_list = target_cell_list[1]
    
#    no_rows = len(i_list)
#    no_cols = len(j_list)
    
    def cm2inch(*tupl):
        inch = 2.54
        if isinstance(tupl[0], tuple):
            return tuple(i/inch for i in tupl[0])
        else:
            return tuple(i/inch for i in tupl)
    
    fig, axes = plt.subplots(nrows = no_rows, ncols = no_cols,
                             figsize = (no_cols*5, no_rows*4) )
#    fig, axes = plt.subplots(nrows = no_rows, ncols = no_cols,
#                             figsize = cm2inch(10.8,9.3))
    
    plot_n = -1
    for row_n in range(no_rows)[::-1]:
        for col_n in range(no_cols):
            plot_n += 1
            target_cell = (i_list[plot_n], j_list[plot_n])
            if no_cols == 1:
                ax = axes[row_n]
            else:
                ax = axes[row_n, col_n]
            
            R, Rs, multi = sample_radii(m_w, m_s, xi, cells, grid_temperature,
                                            target_cell, no_cells_x, no_cells_z,
                                            solute_type)
        
            log_R = np.log10(R)
            log_Rs = np.log10(Rs)
            multi = np.array(multi)
            if no_cells_x % 2 == 0: no_cells_x += 1
            if no_cells_z % 2 == 0: no_cells_z += 1
            
            V_an = no_cells_x * no_cells_z * grid_volume_cell * 1.0E6
            # V_an = (no_neighbors_x * 2 + 1) * (no_neighbors_z * 2 + 1)
            # * grid.volume_cell * 1.0E6
            
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
            # h1, bins1 = np.histogram(log_R, bins=no_bins,
            #                          range=(R_min_log,R_max_log), weights=multi/V_an)
            # h2, bins2 = np.histogram( log_Rs, bins=no_bins_s,
            #                           range=(R_min_log, Rs_max_log),
            #                           weights=multi/V_an )
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
        
    
        
    
    #    ax = axes
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
            ax.plot(np.repeat(bins1,2)[:],
                    np.hstack( [[0.001], np.repeat(h1,2)[:], [0.001] ] ),
                    linewidth = LW, zorder = 6, label = "wet")
            ax.plot(np.repeat(bins2,2)[:],
                    np.hstack( [[0.001], np.repeat(h2,2)[:], [0.001] ] ),
                    linewidth = LW, label = "dry")
            
            
            ax.tick_params(axis='both', which='major', labelsize=TKFS, length = 5)
            ax.tick_params(axis='both', which='minor', labelsize=TKFS, length = 3)
            
#            height = int(grid.compute_location(*target_cell,0.0,0.0)[1])
            height = int((target_cell[1]+0.5)*grid_step_z)
            ax.set_xlabel(r"particle radius (mu)", fontsize = LFS)
            ax.set_ylabel(r"concentration (${\mathrm{cm}^{3}}$)", fontsize = LFS)
            # ax.set_xlabel(r"particle radius ($\si{\micro m}$)", fontsize = LFS)
            # ax.set_ylabel(r"concentration ($\si{\# / cm^{3}}$)", fontsize = LFS)
            ax.set_title( f'h = {height} m ' +
                         f"tg cell ({target_cell[0]} {target_cell[1]}) "
                            + f"no cells ({no_cells_x}, {no_cells_z})",
                            fontsize = TTFS )
            
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
                      loc='upper center', bbox_to_anchor=(0.5, 1.04), frameon = False)
    fig.tight_layout()
#    plt.show()
    if fig_path is not None:
#        fig.savefig(fig_path + "spectrum_" + str(height) +".pdf")
        fig.savefig(fig_path 
                    + f"spectrum_cell_list_j_from_{j_list[0]}_to_{j_list[-1]}_" 
                    + f"no_cells_{no_cells_x}_{no_cells_z}_t_{int(t)}.pdf")

#%% PLOT GRID SCALAR FIELDS WITH TIME

# fields = [fields_t0, fields_t1, ...]
# fields_ti =
# (grid.mixing_ratio_water_vapor, grid.mixing_ratio_water_liquid,
#  grid.potential_temperature, grid.temperature,
#  grid.pressure, grid.saturation)
# input:
# field_indices = (idx1, idx2, ...)  (tuple of int)
# time_indices = (idx1, idx2, ...)  (tuple of int)
# title size (pt)
# TTFS = 18
# labelsize (pt)
# LFS = 18
# ticksize (pt)
# TKFS = 18
def plot_scalar_field_frames(grid, fields,
                            save_times, field_indices, time_indices,
                            no_ticks=[6,6], fig_path=None,
                            TTFS = 12, LFS = 10, TKFS = 10,):
    
    no_rows = len(time_indices)
    no_cols = len(field_indices)
    
    field_names = ["r_v", "r_l", "Theta", "T", "p", "S"]
    scales = [1000, 1000, 1, 1, 0.01, 1]
    units = ["g/kg", "g/kg", "K", "K", "hPa", "-"]
    
    tick_ranges = grid.ranges
    
    fig, axes = plt.subplots(nrows=no_rows, ncols=no_cols,
                           figsize = (4.5*no_cols, 4*no_rows))
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
            if idx_f == 1: field_min = 0.001
            field_max = field.max()
            if idx_f in [2,3,4]: 
                cmap = "coolwarm"
                alpha = None
            else: 
                cmap = "rainbow"
                alpha = 0.7
#                contours = ax[i,j].contour(grid_centers_x_, grid_centers_y_,
#                       field, no_contour_lines_, colors = 'black')
#                ax[i,j].clabel(contours, inline=True, fontsize=8)
            CS = ax.pcolormesh(*grid.corners, field, cmap=cmap, alpha=alpha,
                                    vmin=field_min, vmax=field_max,
                                    edgecolor="face", zorder=1)
            CS.cmap.set_under("white")
            if idx_f == 1:
                cbar = fig.colorbar(CS, ax=ax, extend = "min")
            else: cbar = fig.colorbar(CS, ax=ax)
            cbar.ax.tick_params(labelsize=TKFS)
            ax.set_title(
                field_names[idx_f] + ' (' + units[idx_f] + '), t = '
                + str(int(save_times[idx_t]//60)) + " min", fontsize = TTFS )
            ax.set_xticks( np.linspace( tick_ranges[0,0],
                                             tick_ranges[0,1],
                                             no_ticks[0] ) )
            ax.set_yticks( np.linspace( tick_ranges[1,0],
                                             tick_ranges[1,1],
                                             no_ticks[1] ) )
            ax.tick_params(axis='both', which='major', labelsize=TKFS)
            if i == no_rows-1:
                ax.set_xlabel(r'x (m)', fontsize = LFS)
            if j == 0:
                ax.set_ylabel(r'z (m)', fontsize = LFS)
            ax.grid(color='gray', linestyle='dashed', zorder = 2)
          
    fig.tight_layout()
    if fig_path is not None:
        fig.savefig(fig_path)

# fields = [fields_t0, fields_t1, ...]
# fields_ti =
# (grid.mixing_ratio_water_vapor, grid.mixing_ratio_water_liquid,
#  grid.potential_temperature, grid.temperature,
#  grid.pressure, grid.saturation)
# input:
# field_indices = (idx1, idx2, ...)  (tuple of int)
# time_indices = (idx1, idx2, ...)  (tuple of int)
# title size (pt)
# TTFS = 18
# labelsize (pt)
# LFS = 18
# ticksize (pt)
# TKFS = 18
        
# need grid for dry air density and ?
# fields are with time
# m_s, w_s, xi       
@njit()
def update_T_p(grid_temp, cells, T_p):
    for ID in range(len(T_p)):
        # T_p_ = grid_temp[cells[0,ID],cells[1,ID]]
        T_p[ID] = grid_temp[cells[0,ID],cells[1,ID]]    
# OLd REFERS TO: no cells and active ids stored with time ("all_particle_data")
def plot_scalar_field_frames_Arabas_old(grid, fields, m_s, m_w, xi, pos,
                                    solute_type,
                                    save_times, field_indices, time_indices,
                                    no_ticks=[6,6], fig_path=None,
                                    TTFS = 12, LFS = 10, TKFS = 10,):
    
    
    if solute_type == "AS":
        compute_R_p_w_s_rho_p = compute_R_p_w_s_rho_p_AS
    elif solute_type == "NaCl":
        compute_R_p_w_s_rho_p = compute_R_p_w_s_rho_p_NaCl
    
    no_rows = len(time_indices)
    no_cols = len(field_indices)
   
    no_SIPs = len(xi[0])
    T_p = np.zeros(no_SIPs, dtype = np.float64)
    
    fig, axes = plt.subplots(nrows=no_rows, ncols=no_cols,
                           figsize = (4.5*no_cols, 4*no_rows))
    # load particle data: stored data: m_s, m_w, xi
    # need R_p -> need rho_p -> need w_s and T_p
    # classify droplets per cell by R_p:
    # aerosol: R_p < 0.5 mu
    # cloud drops: 0.5 < R_p < 25 mu
    # rain drops: 25 mu < R_p
    bins_drop_class = [0.5,25]
    
    for t_n, idx_t in enumerate(time_indices):
        pos_ = pos[idx_t]
        # DEACTIVATE LATER... -> need active ids and cells from storage...
        cells = np.array( [np.floor(pos_[0]/grid.steps[0]),
                           np.floor(pos_[1]/grid.steps[1])] ).astype(int)
        update_T_p(fields[idx_t, 3], cells, T_p)        
        
        R_p, w_s, rho_p = compute_R_p_w_s_rho_p(m_w[idx_t], m_s[idx_t], T_p)
        
        idx_R_p = np.digitize(R_p, bins_drop_class)
        idx_classification = np.arange(3).reshape((3,1))
        
        masks_R_p = idx_classification == idx_R_p
        
        
        
                
    field_names = ["r_v", "r_l", "Theta", "T", "p", "S"]
    scales = [1000, 1000, 1, 1, 0.01, 1]
    units = ["g/kg", "g/kg", "K", "K", "hPa", "-"]
    
    tick_ranges = grid.ranges
    
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
            if idx_f == 1: field_min = 0.001
            field_max = field.max()
            if idx_f in [2,3,4]: 
                cmap = "coolwarm"
                alpha = None
            else: 
                cmap = "rainbow"
                alpha = 0.7
#                contours = ax[i,j].contour(grid_centers_x_, grid_centers_y_,
#                       field, no_contour_lines_, colors = 'black')
#                ax[i,j].clabel(contours, inline=True, fontsize=8)
            CS = ax.pcolormesh(*grid.corners, field, cmap=cmap, alpha=alpha,
                                    vmin=field_min, vmax=field_max,
                                    edgecolor="face", zorder=1)
            CS.cmap.set_under("white")
            if idx_f == 1:
                cbar = fig.colorbar(CS, ax=ax, extend = "min")
            else: cbar = fig.colorbar(CS, ax=ax)
            cbar.ax.tick_params(labelsize=TKFS)
            ax.set_title(
                field_names[idx_f] + ' (' + units[idx_f] + '), t = '
                + str(int(save_times[idx_t]//60)) + " min", fontsize = TTFS )
            ax.set_xticks( np.linspace( tick_ranges[0,0],
                                             tick_ranges[0,1],
                                             no_ticks[0] ) )
            ax.set_yticks( np.linspace( tick_ranges[1,0],
                                             tick_ranges[1,1],
                                             no_ticks[1] ) )
            ax.tick_params(axis='both', which='major', labelsize=TKFS)
            if i == no_rows-1:
                ax.set_xlabel(r'x (m)', fontsize = LFS)
            if j == 0:
                ax.set_ylabel(r'z (m)', fontsize = LFS)
            ax.grid(color='gray', linestyle='dashed', zorder = 2)
          
    fig.tight_layout()
    if fig_path is not None:
        fig.savefig(fig_path)
