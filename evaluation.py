#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 12:28:30 2019

@author: jdesk
"""

#%% MODULE IMPORTS

import numpy as np
import math
import matplotlib.pyplot as plt
# import os
# from datetime import datetime
# import timeit

import constants as c
from init import initialize_grid_and_particles, dst_log_normal
from grid import Grid
from grid import interpolate_velocity_from_cell_bilinear,\
                 interpolate_velocity_from_position_bilinear,\
                 compute_cell_and_relative_position
from microphysics import compute_mass_from_radius,\
                         compute_initial_mass_fraction_solute_NaCl,\
                         compute_radius_from_mass,\
                         compute_density_particle,\
                         compute_dml_and_gamma_impl_Newton_full,\
                         compute_R_p_w_s_rho_p
                         
from atmosphere import compute_kappa_air_moist,\
                       compute_diffusion_constant,\
                       compute_thermal_conductivity_air,\
                       compute_specific_heat_capacity_air_moist,\
                       compute_heat_of_vaporization,\
                       compute_saturation_pressure_vapor_liquid,\
                       compute_pressure_vapor,\
                       compute_pressure_ideal_gas,\
                       epsilon_gc, compute_surface_tension_water,\
                       kappa_air_dry,\
                       compute_beta_without_liquid,\
                       compute_temperature_from_potential_temperature_moist                
from file_handling import save_particles_to_files,\
                          save_grid_and_particles_full,\
                          load_grid_and_particles_full,\
                          load_grid_scalar_fields,\
                          load_particle_data
from analysis import sample_masses, sample_radii, compare_functions_run_time,\
                       plot_scalar_field_frames
from integration import compute_dt_max_from_CFL
    
def plot_pos_vel_pt(pos, vel, grid,
                    figsize=(8,8), no_ticks = [6,6],
                    MS = 1.0, ARRSCALE=2):
    u_g = 0.5 * ( grid.velocity[0,0:-1] + grid.velocity[0,1:] )
    v_g = 0.5 * ( grid.velocity[1,:,0:-1] + grid.velocity[1,:,1:] )
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
    # ax.grid(which="minor")
    plt.show()


### storage directories -> need to assign "simdata_path" and "fig_path"
my_OS = "Linux_desk"
# my_OS = "Mac"

if(my_OS == "Linux_desk"):
    home_path = '/home/jdesk/'
    simdata_path = home_path + "OneDrive/python/sim_data/"
    # fig_path = home_path + 'Onedrive/Uni/Masterthesis/latex/Report/Figures/'
elif (my_OS == "Mac"):
    home_path = "/Users/bohrer/"
    simdata_path = home_path + "OneDrive - bwedu/python/sim_data/"
    # fig_path = home_path + 'OneDrive - bwedu/Uni/Masterthesis/latex/Report/Figures/'
    

#%% particle trajectories
from analysis import plot_particle_trajectories

# folder_load = "190511/grid_75_75_spcm_4_4/"
folder_load = "190512/grid_10_10_spcm_0_4/sim3/"
# folder_load = "190507/test1/"
# folder_load = "190508/grid_75_75_spct_20/"
# folder_save = "190508/grid_10_10_spct_4/"
# folder_save = folder_load
path = simdata_path + folder_load

# reload = False
t = 1800
save_time_every = 4
t_start = 0
t_end = 1800
dt_save = 10
reload = True

if reload:
    path = simdata_path + folder_load
    grid, pos, cells, vel, m_w, m_s, xi, active_ids, removed_ids = \
        load_grid_and_particles_full(t, path)
    R_p, w_s, rho_p = compute_R_p_w_s_rho_p(m_w, m_s,
                                            grid.temperature[tuple(cells)] )
    R_s = compute_radius_from_mass(m_s, c.mass_density_NaCl_dry)

pt_save_times = np.arange(t_start, t_end, dt_save).astype(int)

vecs, scals, xis = load_particle_data(path, pt_save_times)

no_trace_ids = 10
trace_id_dist = math.ceil(len(xi)/(no_trace_ids + 1))
trace_ids = np.arange(trace_id_dist, len(xi), trace_id_dist)

plot_particle_trajectories(vecs[:,0], grid, trace_ids, MS = 1.0)

    
#%% PLOT GRID SCALAR FIELD FRAMES OVER TIME
    
# folder_load = "190511/grid_75_75_spcm_4_4/"
folder_load_base = "190512/grid_75_75_spcm_0_4/"
folder_load = "190512/grid_75_75_spcm_0_4/sim8/"
# folder_load = "190507/test1/"
# folder_load = "190508/grid_75_75_spct_20/"
# folder_save = "190508/grid_10_10_spct_4/"
folder_save = folder_load


# reload = False
t = 0
save_time_every = 1
t_start = 0
t_end = 10800
dt_save = 300
reload = True

if reload:
    path = simdata_path + folder_load_base
    grid, pos, cells, vel, m_w, m_s, xi, active_ids, removed_ids = \
        load_grid_and_particles_full(t, path)
    R_p, w_s, rho_p = compute_R_p_w_s_rho_p(m_w, m_s,
                                            grid.temperature[tuple(cells)] )
    R_s = compute_radius_from_mass(m_s, c.mass_density_NaCl_dry)

grid_save_times =\
    np.arange(t_start, t_end + 0.5 * dt_save, dt_save).astype(int)

# grid.plot_thermodynamic_scalar_fields_grid()

dt_max = compute_dt_max_from_CFL(grid)

print("grid_save_times full: ( len =", len(grid_save_times), ")")
print(grid_save_times)

path = simdata_path + folder_load

fields = load_grid_scalar_fields(path, grid_save_times)

print("fields.shape")
print(fields.shape)

field_ind = np.arange(6)
time_ind = np.arange(0, len(grid_save_times), save_time_every)

print("grid_save_times indices and times chosen:")
for idx_t in time_ind:
    print(idx_t, grid_save_times[idx_t])

no_ticks=[6,6]

fig_name = path + "scalar_fields.png"
# fig_name = None

plot_scalar_field_frames(grid, fields, grid_save_times,
                  field_ind, time_ind, no_ticks, fig_name)

# pos2 = pos[:,::4]
# vel2 = vel[:,::4]
# plot_pos_vel_pt(pos2, vel2, grid,
#                     figsize=(8,8), no_ticks = [6,6],
#                     MS = 1.0, ARRSCALE=30)

#%%

# folder_load = "190511/grid_75_75_spcm_4_4/"
folder_load = "190512/grid_75_75_spcm_0_4/"
# folder_load = "190507/test1/"
# folder_load = "190508/grid_75_75_spct_20/"
# folder_save = "190508/grid_10_10_spct_4/"
folder_save = folder_load


# reload = False
t = 0
save_time_every = 4
t_start = 0
t_end = 1800
dt_save = 45
reload = True


if reload:
    path = simdata_path + folder_load
    grid, pos, cells, vel, m_w, m_s, xi, active_ids, removed_ids = \
        load_grid_and_particles_full(t, path)
    R_p, w_s, rho_p = compute_R_p_w_s_rho_p(m_w, m_s,
                                            grid.temperature[tuple(cells)] )
    R_s = compute_radius_from_mass(m_s, c.mass_density_NaCl_dry)
    

from index import ind
# ind = index.ind

T_bef = np.copy(grid.temperature)
grid_scalar_fields = np.array( ( grid.temperature,
                                 grid.pressure,
                                 grid.potential_temperature,
                                 grid.mass_density_air_dry,
                                 grid.mixing_ratio_water_vapor,
                                 grid.mixing_ratio_water_liquid,
                                 grid.saturation,
                                 grid.saturation_pressure) )

grid_mat_prop = np.array( ( grid.thermal_conductivity,
                            grid.diffusion_constant,
                            grid.heat_of_vaporization,
                            grid.surface_tension,
                            grid.specific_heat_capacity,
                            grid.viscosity,
                            grid.mass_density_fluid ) )
import index as idx
# i_T = 0
# TT = grid_scalar_fields[0]
grid.temperature = grid_scalar_fields[idx.T]
grid.pressure = grid_scalar_fields[ind["p"]]
grid.potential_temperature = grid_scalar_fields[ind["Th"]]
grid.mass_density_air_dry = grid_scalar_fields[ind["rhod"]]
grid.mixing_ratio_water_vapor = grid_scalar_fields[ind["rv"]]
grid.mixing_ratio_water_liquid = grid_scalar_fields[ind["rl"]]
grid.saturation = grid_scalar_fields[ind["S"]]
grid.saturation_pressure = grid_scalar_fields[ind["es"]]

grid.thermal_conductivity = grid_mat_prop[ind["K"]]
grid.diffusion_constant = grid_mat_prop[ind["Dv"]]
grid.heat_of_vaporization = grid_mat_prop[ind["L"]]
grid.surface_tension = grid_mat_prop[ind["sigmaw"]]
grid.specific_heat_capacity = grid_mat_prop[ind["cpf"]]
grid.viscosity = grid_mat_prop[ind["muf"]]
grid.mass_density_fluid = grid_mat_prop[ind["rhof"]]
    
print("T_bef - grid.temperature")
print(T_bef - grid.temperature)

print("grid.temperature += 1.0")
grid.temperature += 1.0
print('grid_scalar_fields[idx.T] - grid.temperature')
print(grid_scalar_fields[idx.T] - grid.temperature)
# print(grid_scalar_fields[0] - grid.temperature)

print("grid_scalar_fields[idx.T] += 3.5")
grid_scalar_fields[idx.T] += 3.5
print('grid_scalar_fields[idx.T] - grid.temperature')
print(grid_scalar_fields[idx.T] - grid.temperature)
grid.temperature = grid.potential_temperature * 1.5
# grid.temperature = np.arange(6)

print(grid.temperature)
print(grid_scalar_fields[idx.T])

# TT += 1.0
# print("grid_scalar_fields[0] - TT")
# print(grid_scalar_fields[0] - TT)

