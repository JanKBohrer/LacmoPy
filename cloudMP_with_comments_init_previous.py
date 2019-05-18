#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 14:07:21 2019

@author: jdesk
"""

# 1. init()
# 2. spinup()
# 3. simulate()

### output:
## full saves:
# grid_parameters, grid_scalars, grid_vectors
# pos, cells, vel, masses, xi
## data:
# grid:
# initial: p, T, Theta, S, r_v, r_l, rho_dry, e_s
# continuous
# p, T, Theta, S, r_v, r_l, 
# particles:
# pos, vel, masses


# set:
# files for full save
# files for data output -> which data?

# 1. generate grid and particles + save initial to file
# grid, pos, cells, vel, masses, xi = init()

# 2. changes grid, pos, cells, vel, masses and save to file after spin up
# data output during spinup if desired
# spinup(grid, pos, cells, vel, masses, xi)
# in here:
# advection(grid) (grid, dt_adv) spread over particle time step h
# propagation(pos, vel, masses, grid) (particles) # switch gravity on/off here!
# condensation() (particle <-> grid) maybe do together with collision 
# collision() maybe do together with condensation

# 3. changes grid, pos, cells, vel, masses,
# data output to file and 
# save to file in intervals chosen
# need to set start time, end time, integr. params, interval of print,
# interval of full save, ...)
# simulate(grid, pos, vel, masses, xi)

#%% MODULE IMPORTS

import numpy as np
import math
import matplotlib.pyplot as plt
import os
from datetime import datetime
# import timeit

import constants as c
from init import initialize_grid_and_particles, dst_log_normal
# from grid import Grid
from grid import interpolate_velocity_from_cell_bilinear,\
                 compute_cell_and_relative_position
                 # interpolate_velocity_from_position_bilinear,\
from microphysics import compute_radius_from_mass,\
                         compute_R_p_w_s_rho_p
#                          compute_initial_mass_fraction_solute_NaCl,\
#                          compute_density_particle,\
#                          compute_dml_and_gamma_impl_Newton_full,\
                         
# from atmosphere import compute_kappa_air_moist,\
#                        compute_pressure_vapor,\
#                        compute_pressure_ideal_gas,\
#                        epsilon_gc, compute_surface_tension_water,\
#                        kappa_air_dry,\
#                        compute_beta_without_liquid,\
#                        compute_temperature_from_potential_temperature_moist                
                       # compute_diffusion_constant,\
                       # compute_thermal_conductivity_air,\
                       # compute_specific_heat_capacity_air_moist,\
                       # compute_heat_of_vaporization,\
                       # compute_saturation_pressure_vapor_liquid,\
from file_handling import save_grid_and_particles_full,\
                          load_grid_and_particles_full
from file_handling import dump_particle_data, save_grid_scalar_fields                          
#                         save_particles_to_files,\
from analysis import compare_functions_run_time

from integration import simulate

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
    fig_path = home_path + 'Onedrive/Uni/Masterthesis/latex/Report/Figures/'
elif (my_OS == "Mac"):
    home_path = "/Users/bohrer/"
    simdata_path = home_path + "OneDrive - bwedu/python/sim_data/"
    fig_path = home_path + 'OneDrive - bwedu/Uni/Masterthesis/latex/Report/Figures/'


### III. SIMULATION
### 3. simulation

#%% SET SIMULATION PARAMETERS

from analysis import compare_functions_run_time
from grid import update_grid_r_l
from file_handling import dump_particle_data, save_grid_scalar_fields
from integration import update_pos_from_vel_BC_PS,\
                        update_cells_and_rel_pos,\
                        compute_divergence_upwind,\
                        propagate_particles_subloop_step,\
                        propagate_grid_subloop_step
import index

####################################
### SET
# # load grid and particle list from directory
# # folder_load = "190508/grid_10_10_spct_4/"
# # folder_load = "190510/grid_15_15_spcm_4_4/"
# folder_load = "190514/grid_75_75_spcm_0_4/"

# # folder_save = "190508/grid_10_10_spct_4/sim4/"
# # folder_save = "190510/grid_15_15_spcm_4_4/sim2/"
# folder_save = "190514/grid_75_75_spcm_0_4/sim2/"
# # folder_save = "190511/grid_75_75_spcm_4_4"

# t_start = 0.0
# t_end = 600.0 # s
# # timestep of advection
# dt = 1.0 # s

# # timescale "scale_dt" = N_h of subloop with timestep h = dt/(2 N_h)
# # => N_h = dt/(2 h)
# # with implicit Newton:
# # from experience: dt_sub <= 0.1 s 
# # dt_sub = 0.1 -> N = 1.0/(0.2) = 5 OR N = 5.0/(0.2) = 25 OR N = 10.0/(0.2) = 50
# scale_dt = 5

# # Newton implicit root finding number of iterations
# Newton_iter = 3

# # set g = 0 for spin up 2 hours
# # not implemented yet... -> NO EFFECT YET
# # spin_up_time = 7200.0 # s 

# # no of frames of grid properties T, p, Theta, r_v, r_l, S
# no_grid_frames = 6

# # full save of grid and particles in time intervals
# full_save_time_interval = 600 # s

# # particles to be traced, evenly distributed over "active_ids"
# # not in use right now, all particle pos are dumped...
# no_trace_ids = 20

# # trace particle positions every "trace_every" advection steps "dt"
# # trace_every = 4
# dump_every = 10

# # phys. constants
# p_ref = 1.0E5
# p_ref_inv = 1.0E-5
# # g must be positive = 9.8... or 0.0
# # g_set = 0.0
# g_set = c.earth_gravity

# adiabatic_index = 1.4
# accomodation_coefficient = 1.0
# condensation_coefficient = 0.0415
### SET END
####################################

#################################### 
### DERIVED
# dt_sub = dt/(2 * scale_dt)
# dt_sub_half = 0.5 * dt_sub
# print("dt_sub = ", dt_sub)

# cnt_max = (t_end - t_start) /dt
# frame_every = int(math.ceil(cnt_max / no_grid_frames))
# full_save_every = int(full_save_time_interval // dt)

# path = simdata_path + folder_load

# grid, pos, cells, vel, m_w, m_s, xi, active_ids, removed_ids = \
#     load_grid_and_particles_full(0, path)

# R_p, w_s, rho_p = compute_R_p_w_s_rho_p(m_w, m_s,
#                                         grid.temperature[tuple(cells)] )
# R_s = compute_radius_from_mass(m_s, c.mass_density_NaCl_dry)

# print(grid.p_ref)          
# print(grid.p_ref_inv)          

# trace_id_dist = int(math.floor(len(xi)/(no_trace_ids)))
# trace_ids = np.arange(int(trace_id_dist*0.5), len(xi), trace_id_dist)
# trace_id_dist = math.ceil(len(xi)/(no_trace_ids + 1))
# trace_id_dist = int(math.ceil(len(xi)/(no_trace_ids + 1)))
# trace_ids = np.arange(trace_id_dist, len(xi), trace_id_dist)

# sim_paras = [dt, dt_sub, Newton_iter]
# sim_par_names = "dt dt_sub Newton_iter"
# # sim_paras = [dt, dt_sub, Newton_iter, no_trace_ids]
# # sim_par_names = "dt dt_sub Newton_iter no_trace_ids"

# sim_para_file = path + "sim_paras_t_" + str(t_start) + ".txt"

# with open(sim_para_file, "w") as f:
#     f.write( sim_par_names + '\n' )
#     for item in sim_paras:
#         if type(item) is list or type(item) is np.ndarray:
#             for el in item:
#                 f.write( f'{el} ' )
#         else: f.write( f'{item} ' )

### DERIVED END
#################################### 

#################################### INIT
# init grid properties
# grid.update_material_properties()
# V0_inv = 1.0 / grid.volume_cell
# grid.rho_dry_inv =\
#     np.ones_like(grid.mass_density_air_dry) / grid.mass_density_air_dry
# grid.mass_dry_inv = V0_inv * grid.rho_dry_inv

# # tau = t_start
# # taus = []
# # taus.append(tau)

# delta_Q_p = np.zeros_like(grid.temperature)
# delta_m_l = np.zeros_like(grid.temperature)
# water_removed = 0.0

# rel_pos = np.zeros_like(pos)
# update_cells_and_rel_pos(pos, cells, rel_pos, grid.ranges, grid.steps)

# T_p = np.ones_like(m_w)

# ind = index.ind
# grid_scalar_fields = np.array( ( grid.temperature,
#                                  grid.pressure,
#                                  grid.potential_temperature,
#                                  grid.mass_density_air_dry,
#                                  grid.mixing_ratio_water_vapor,
#                                  grid.mixing_ratio_water_liquid,
#                                  grid.saturation,
#                                  grid.saturation_pressure,
#                                  grid.mass_dry_inv,
#                                  grid.rho_dry_inv) )

# grid_mat_prop = np.array( ( grid.thermal_conductivity,
#                             grid.diffusion_constant,
#                             grid.heat_of_vaporization,
#                             grid.surface_tension,
#                             grid.specific_heat_capacity,
#                             grid.viscosity,
#                             grid.mass_density_fluid ) )

# grid_scalar_fields[0] = grid.temperature
# grid_scalar_fields[1] = grid.pressure
# grid_scalar_fields[2] = grid.potential_temperature
# grid_scalar_fields[3] = grid.mass_density_air_dry
# grid_scalar_fields[4] = grid.mixing_ratio_water_vapor
# grid_scalar_fields[5] = grid.mixing_ratio_water_liquid
# grid_scalar_fields[6] = grid.saturation
# grid_scalar_fields[7] = grid.saturation_pressure
# grid_scalar_fields[8] = grid.mass_dry_inv
# grid_scalar_fields[9] = grid.rho_dry_inv

# grid_mat_prop[0] = grid.thermal_conductivity
# grid_mat_prop[1] = grid.diffusion_constant
# grid_mat_prop[2] = grid.heat_of_vaporization
# grid_mat_prop[3] = grid.surface_tension
# grid_mat_prop[4] = grid.specific_heat_capacity
# grid_mat_prop[5] = grid.viscosity
# grid_mat_prop[6] = grid.mass_density_fluid

# grid.temperature = grid_scalar_fields[0]
# grid.pressure = grid_scalar_fields[1]
# grid.potential_temperature = grid_scalar_fields[2]
# grid.mass_density_air_dry = grid_scalar_fields[3]
# grid.mixing_ratio_water_vapor = grid_scalar_fields[4]
# grid.mixing_ratio_water_liquid = grid_scalar_fields[5]
# grid.saturation = grid_scalar_fields[6]
# grid.saturation_pressure = grid_scalar_fields[7]

# grid.thermal_conductivity = grid_mat_prop[0]
# grid.diffusion_constant = grid_mat_prop[1]
# grid.heat_of_vaporization = grid_mat_prop[2]
# grid.surface_tension = grid_mat_prop[3]
# grid.specific_heat_capacity = grid_mat_prop[4]
# grid.viscosity = grid_mat_prop[5]
# grid.mass_density_fluid = grid_mat_prop[6]

# grid.temperature = grid_scalar_fields[ind["T"]]
# grid.pressure = grid_scalar_fields[ind["p"]]
# grid.potential_temperature = grid_scalar_fields[ind["Th"]]
# grid.mass_density_air_dry = grid_scalar_fields[ind["rhod"]]
# grid.mixing_ratio_water_vapor = grid_scalar_fields[ind["rv"]]
# grid.mixing_ratio_water_liquid = grid_scalar_fields[ind["rl"]]
# grid.saturation = grid_scalar_fields[ind["S"]]
# grid.saturation_pressure = grid_scalar_fields[ind["es"]]

# grid.thermal_conductivity = grid_mat_prop[ind["K"]]
# grid.diffusion_constant = grid_mat_prop[ind["Dv"]]
# grid.heat_of_vaporization = grid_mat_prop[ind["L"]]
# grid.surface_tension = grid_mat_prop[ind["sigmaw"]]
# grid.specific_heat_capacity = grid_mat_prop[ind["cpf"]]
# grid.viscosity = grid_mat_prop[ind["muf"]]
# grid.mass_density_fluid = grid_mat_prop[ind["rhof"]]

# grid.plot_thermodynamic_scalar_fields_grid()
#################################### INIT end

#%% MAIN SIMULATION LOOP

# from integration import propagate_particles_subloop_step
# from integration import integrate_subloop, simulate

# ####################################
# # MAIN SIMULATION LOOP:
# path = simdata_path + folder_save
# if not os.path.exists(path):
#     os.makedirs(path)
# # dt = dt_advection
# grid_save_times = []
# start_time = datetime.now()

# grid_velocity = grid.velocity
# grid_mass_flux_air_dry = grid.mass_flux_air_dry
# grid_ranges = grid.ranges
# grid_steps = grid.steps
# grid_no_cells = grid.no_cells
# grid_volume_cell = grid.volume_cell
# p_ref = grid.p_ref
# p_ref_inv = grid.p_ref_inv


#%%

####################################
### SET
# load grid and particle list from directory
# folder_load = "190508/grid_10_10_spct_4/"
# folder_load = "190510/grid_15_15_spcm_4_4/"
folder_load = "190514/grid_75_75_spcm_0_4/"

# folder_save = "190508/grid_10_10_spct_4/sim4/"
# folder_save = "190510/grid_15_15_spcm_4_4/sim2/"
folder_save = "190514/grid_75_75_spcm_0_4/sim3/"
# folder_save = "190511/grid_75_75_spcm_4_4"

t_start = 0.0
t_end = 3600.0 # s
# timestep of advection
dt = 1.0 # s

# timescale "scale_dt" = N_h of subloop with timestep h = dt/(2 N_h)
# => N_h = dt/(2 h)
# with implicit Newton:
# from experience: dt_sub <= 0.1 s 
# dt_sub = 0.1 -> N = 1.0/(0.2) = 5 OR N = 5.0/(0.2) = 25 OR N = 10.0/(0.2) = 50
scale_dt = 5

# Newton implicit root finding number of iterations
Newton_iter = 3

# set g = 0 for spin up 2 hours
# not implemented yet... -> NO EFFECT YET
# spin_up_time = 7200.0 # s 

# frames of grid properties T, p, Theta, r_v, r_l, S frame_every steps dt
# MUST be >= than dump_every and an integer multiple of dump every
# NOTE that grid frames are taken at
# t = t_start, t_start + frame_every * dt AND additionally at the end, t = t_end
frame_every = 300
# no_grid_frames = 6

# full save of grid and particles in time intervals
# --> no effect right now, only full save at the end!
# full_save_time_interval = 600 # s

# particles to be traced, evenly distributed over "active_ids"
# not in use right now, all particle pos are dumped...
trace_ids = 20

# trace particle positions every "trace_every" advection steps "dt"
# trace_every = 4
# must be <= frame_every and frame_every/dump_every must be integer
# NOTE that particle is saved at
# t = t_start, t_start + dump_every * dt AND additionally at the end, t = t_end
dump_every = 10

# phys. constants
p_ref = 1.0E5
p_ref_inv = 1.0E-5
# g must be positive = 9.8... or 0.0
# g_set = 0.0
g_set = c.earth_gravity
### SET END
####################################

#################################### 
### INIT
path = simdata_path + folder_load

grid, pos, cells, vel, m_w, m_s, xi, active_ids, removed_ids = \
    load_grid_and_particles_full(0, path)

# init grid properties
grid.update_material_properties()
V0_inv = 1.0 / grid.volume_cell
grid.rho_dry_inv =\
    np.ones_like(grid.mass_density_air_dry) / grid.mass_density_air_dry
grid.mass_dry_inv = V0_inv * grid.rho_dry_inv

delta_Q_p = np.zeros_like(grid.temperature)
delta_m_l = np.zeros_like(grid.temperature)
water_removed = 0.0

rel_pos = np.zeros_like(pos)
update_cells_and_rel_pos(pos, cells, rel_pos, grid.ranges, grid.steps)
T_p = np.ones_like(m_w)

path = simdata_path + folder_save
if not os.path.exists(path):
    os.makedirs(path)
### INIT END
#################################### 

simulate(grid,
         pos, vel, cells, rel_pos, m_w, m_s, xi, T_p,
         delta_m_l, delta_Q_p,
         dt, scale_dt, t_start, t_end,
         Newton_iter, g_set,
         frame_every, dump_every, trace_ids,
         path)
