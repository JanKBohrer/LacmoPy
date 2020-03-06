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
import timeit

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
                          load_grid_and_particles_full
from analysis import sample_masses, sample_radii, compare_functions_run_time

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

#%% I. GENERATE GRID AND PARTICLES
from init import initialize_grid_and_particles
# 1a. generate grid and particles + save initial to file
# domain size
x_min = 0.0
x_max = 1500.0
z_min = 0.0
z_max = 1500.0

# grid steps
dx = 20.0
dy = 1.0
dz = 20.0

p_0 = 101500 # surface pressure in Pa
p_ref = 1.0E5 # ref pressure for potential temperature in Pa
r_tot_0 = 7.5E-3 # kg water / kg dry air
# r_tot_0 = 22.5E-3 # kg water / kg dry air
# r_tot_0 = 7.5E-3 # kg water / kg dry air
Theta_l = 289.0 # K
# number density of particles mode 1 and mode 2:
n_p = np.array([60.0E6, 100.0E6]) # m^3
# n_p = np.array([60.0E6, 40.0E6]) # m^3

# no_super_particles_cell = [N1,N2] is a list with
# N1 = no super part. per cell in mode 1 etc.
no_spcm = np.array([0, 4])
# no_spcm = np.array([0, 4])

### creating random particles
# parameters of log-normal distribution:
dst = dst_log_normal
# in log(mu)
par_sigma = np.log( [1.4,1.6] )
# par_sigma = np.log( [1.0,1.0] )
# in mu
par_r0 = 0.5 * np.array( [0.04, 0.15] )
dst_par = []
for i,sig in enumerate(par_sigma):
    dst_par.append([par_r0[i],sig])
dst_par = np.array(dst_par)
# parameters of the quantile-integration
P_min = 0.001
P_max = 0.999
dr = 1E-4
r0 = dr
r1 = 10 * par_sigma
reseed = False
rnd_seed = 4713

### for saturation adjustment phase
S_init_max = 1.05
dt_init = 0.1 # s
# number of iterations for the root finding
# Newton algorithm in the implicit method
Newton_iterations = 2
# maximal allowed iter counts in initial particle water take up to equilibrium
iter_cnt_limit = 800

folder = "190512/grid_10_10_spcm_0_4/"
path = simdata_path + folder
if not os.path.exists(path):
    os.makedirs(path)

grid, pos, cells, vel, m_w, m_s, xi, active_ids, removed_ids =\
    initialize_grid_and_particles(
        x_min, x_max, z_min, z_max, dx, dy, dz,
        p_0, p_ref, r_tot_0, Theta_l,
        n_p, no_spcm, dst, dst_par, 
        P_min, P_max, r0, r1, dr, rnd_seed, reseed,
        S_init_max, dt_init, Newton_iterations, iter_cnt_limit, path)

#%% simple plots
# IN WORK: ADD AUTOMATIC CREATION OF DRY SIZE SPECTRA COMPARED WITH EXPECTED PDF
# FRO ALL MODES AND SAVE AS FILE 
grid.plot_thermodynamic_scalar_fields_grid()

#%% III. SIMULATION
### 3. simulation
# 3a. load grid from file
folder_load = "190510/grid_15_15_spcm_4_4/"
# folder_load = "190507/test1/"
# folder_load = "190508/grid_75_75_spct_20/"
path = simdata_path + folder_load

grid, pos, cells, vel, m_w, m_s, xi, active_ids, removed_ids = \
    load_grid_and_particles_full(0, path)

R_p, w_s, rho_p = compute_R_p_w_s_rho_p(m_w, m_s,
                                        grid.temperature[tuple(cells)] )
R_s = compute_radius_from_mass(m_s, c.mass_density_NaCl_dry)


#%% TESTS

#%% update_particle_locations_from_velocities_BC_PS
from evaluation import compare_functions_run_time
from grid import update_grid_r_l 
from file_handling import dump_particle_data, save_grid_scalar_fields
from integration import update_pos_from_vel_BC_PS,\
                        update_pos_from_vel_BC_PS_np,\
                        update_pos_from_vel_BC_PS_par

folder_load = "190508/grid_10_10_spct_4/"
# folder_load = "190507/test1/"
# folder_load = "190508/grid_75_75_spct_20/"
path = simdata_path + folder_load

grid, pos, cells, vel, m_w, m_s, xi, active_ids, removed_ids = \
    load_grid_and_particles_full(0, path)

R_p, w_s, rho_p = compute_R_p_w_s_rho_p(m_w, m_s,
                                        grid.temperature[tuple(cells)] )
R_s = compute_radius_from_mass(m_s, c.mass_density_NaCl_dry)
                        
# print(len(xi))
# print(pos)
# print(vel)
dt_sub = 0.1 # s
dt_sub_half = 0.5 * dt_sub
# pos_bef = np.copy(pos)

funcs = ["update_pos_from_vel_BC_PS_np",
         "update_pos_from_vel_BC_PS",
         "update_pos_from_vel_BC_PS_par"]
pars = "pos,vel,xi,grid.ranges,dt_sub_half"
rs = [7,7,7]
ns = [10,1000,1000]
print("no_spt =",len(xi))
compare_functions_run_time(funcs, pars, rs, ns, globals_ = globals())
#%% test subploop steps njit
from evaluation import compare_functions_run_time
from grid import update_grid_r_l 
from file_handling import dump_particle_data, save_grid_scalar_fields
from integration import update_pos_from_vel_BC_PS,\
                        update_pos_from_vel_BC_PS_np,\
                        update_pos_from_vel_BC_PS_par,\
                        update_m_w_and_delta_m_l_impl_Newton,\
                        update_m_w_and_delta_m_l_impl_Newton_np,\
                        update_m_w_and_delta_m_l_impl_Newton_par,\
                        propagate_particles_subloop_step,\
                        propagate_particles_subloop_step_np

folder_load = "190508/grid_10_10_spct_4/"
# folder_load = "190507/test1/"
# folder_load = "190508/grid_75_75_spct_20/"
# folder_load = "190508/grid_75_75_spct_2/"
path = simdata_path + folder_load

grid, pos, cells, vel, m_w, m_s, xi, active_ids, removed_ids = \
    load_grid_and_particles_full(0, path)

R_p, w_s, rho_p = compute_R_p_w_s_rho_p(m_w, m_s,
                                        grid.temperature[tuple(cells)] )
R_s = compute_radius_from_mass(m_s, c.mass_density_NaCl_dry)

cells, rel_pos = compute_cell_and_relative_position(pos, grid.ranges,
                                                    grid.steps)

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
                        
dt_sub = 0.1 # s
dt_sub_half = 0.5 * dt_sub
Newton_iter = 3
delta_Q_p = np.zeros_like(grid.temperature)
delta_m_l = np.zeros_like(grid.temperature)
T_p = grid.temperature[cells[0],cells[1]]

# update_m_w_and_delta_m_l_impl_Newton_np(
#         grid.temperature, grid.pressure, grid.saturation,
#         grid.saturation_pressure, grid.thermal_conductivity, 
#         grid.diffusion_constant, grid.heat_of_vaporization,
#         grid.surface_tension, cells, m_w, m_s, xi, R_p, w_s, rho_p, T_p,
#         delta_m_l, delta_Q_p, dt_sub, Newton_iter)

# pos_bef = np.copy(pos)
funcs = ["update_m_w_and_delta_m_l_impl_Newton_np",
          "update_m_w_and_delta_m_l_impl_Newton",
          "update_m_w_and_delta_m_l_impl_Newton_par"]
          # "update_pos_from_vel_BC_PS_par"]
pars =\
"""
grid.temperature, grid.pressure, grid.saturation,
grid.saturation_pressure, grid.thermal_conductivity, 
grid.diffusion_constant, grid.heat_of_vaporization,
grid.surface_tension, cells, m_w, m_s, xi, R_p, w_s, rho_p, T_p,
delta_m_l, delta_Q_p, dt_sub, Newton_iter
"""
rs = [5,5,5]
ns = [10,100,100]
print("no_spt =",len(xi))
compare_functions_run_time(funcs, pars, rs, ns, globals_ = globals())

#%% LOAD EVERYTHING IMPORTANT
from integration import update_vel_impl
from grid import interpolate_velocity_from_cell_bilinear

folder_load = "190508/grid_10_10_spct_4/"
# folder_load = "190507/test1/"
# folder_load = "190508/grid_75_75_spct_20/"
# folder_load = "190508/grid_75_75_spct_2/"
path = simdata_path + folder_load

grid, pos, cells, vel, m_w, m_s, xi, active_ids, removed_ids = \
    load_grid_and_particles_full(0, path)

R_p, w_s, rho_p = compute_R_p_w_s_rho_p(m_w, m_s,
                                        grid.temperature[tuple(cells)] )
R_s = compute_radius_from_mass(m_s, c.mass_density_NaCl_dry)

cells, rel_pos = compute_cell_and_relative_position(pos, grid.ranges,
                                                    grid.steps)

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

dt_sub = 0.1 # s
dt_sub_half = 0.5 * dt_sub
Newton_iter = 3
delta_Q_p = np.zeros_like(grid.temperature)
delta_m_l = np.zeros_like(grid.temperature)
T_p = grid.temperature[cells[0],cells[1]]

vel_ipol = interpolate_velocity_from_cell_bilinear(cells, rel_pos,
        grid.velocity, grid.no_cells)

#%%

update_vel_impl(vel, cells, rel_pos, xi, R_p, rho_p,
                    grid.velocity, grid.viscosity, grid.mass_density_fluid, 
                    grid.no_cells, c.earth_gravity, dt_sub)

#%% propagate_particles_subloop_step
from integration import propagate_particles_subloop_step_np
from integration import propagate_particles_subloop_step
from integration import propagate_particles_subloop_step_par
from integration import update_T_p
from numba import njit
import index
T_p = np.zeros_like(m_w)
# propagate_particles_subloop_step(grid_scalar_fields, grid_mat_prop,
#                                     grid.velocity,
#                                     grid.no_cells, grid.ranges, grid.steps,
#                                     pos, vel, cells, rel_pos, m_w, m_s, xi,
#                                     T_p,
#                                     delta_m_l, delta_Q_p,
#                                     dt_sub, dt_sub,
#                                     Newton_iter, c.earth_gravity)
@njit(parallel=True)
# @njit()
def update_T_p_par(grid_scalar_fields, cells, T_p):
    update_T_p(grid_scalar_fields[0], cells, T_p)

print(T_p)
update_T_p_par(grid_scalar_fields, cells, T_p)

print(T_p)


#%%

# index_dtype = np.dtype({"names" : ["T","p","Th"],
# import index
# ind = index.ind

#%%
import index as idx
ind = idx.ind
from integration import propagate_particles_subloop_step_par
from integration import propagate_particles_subloop_step
propagate_particles_subloop_step_par(grid_scalar_fields, grid_mat_prop,
                                    grid.velocity,
                                    grid.no_cells, grid.ranges, grid.steps,
                                    pos, vel, cells, rel_pos, m_w, m_s, xi,
                                    T_p,
                                    delta_m_l, delta_Q_p,
                                    dt_sub, dt_sub,
                                    Newton_iter, c.earth_gravity, ind)

#%%

funcs = ["propagate_particles_subloop_step_np",
          "propagate_particles_subloop_step",
          "propagate_particles_subloop_step_par"]
          # "update_pos_from_vel_BC_PS_par"]
pars =\
"""
grid_scalar_fields, grid_mat_prop,
grid.velocity,
grid.no_cells, grid.ranges, grid.steps,
pos, vel, cells, rel_pos, m_w, m_s, xi, T_p,
delta_m_l, delta_Q_p,
dt_sub, dt_sub,
Newton_iter, c.earth_gravity
"""
rs = [5,5,5]
ns = [10,100,100]
print("no_spt =",len(xi))
compare_functions_run_time(funcs, pars, rs, ns, globals_ = globals())

#%% dump_particle_data
from evaluation import compare_functions_run_time
from grid import update_grid_r_l
from file_handling import dump_particle_data, save_grid_scalar_fields
# from integration import update_particle_locations_from_velocities_BC_PS,\
#                         update_particle_locations_from_velocities_BC_PS_np,\
#                         update_particle_locations_from_velocities_BC_PS_par

folder_load = "190508/grid_10_10_spct_4/"
# folder_load = "190507/test1/"
# folder_load = "190508/grid_75_75_spct_20/"
path = simdata_path + folder_load

grid, pos, cells, vel, m_w, m_s, xi, active_ids, removed_ids = \
    load_grid_and_particles_full(0, path)
R_p, w_s, rho_p = compute_R_p_w_s_rho_p(m_w, m_s,
                                        grid.temperature[tuple(cells)] )
R_s = compute_radius_from_mass(m_s, c.mass_density_NaCl_dry)

t = 0.0
dt_sub = 0.1 # s
dt_sub_half = 0.5 * dt_sub

dump_particle_data(t, pos, vel, m_w, m_s, xi,
                   grid.temperature, grid.mixing_ratio_water_vapor, path)

pt_data_scalar = np.load(path + "particle_scalar_data_" + str(int(t)) + ".npy")
pt_data_vector = np.load(path + "particle_vector_data_" + str(int(t)) + ".npy")
grd_data = np.load(path + "grid_T_rv_" + str(int(t)) + ".npy")
print(np.shape(pt_data_scalar))
print(np.shape(pt_data_vector))
print(grd_data.shape)

#%% update_cells_and_rel_pos
from evaluation import compare_functions_run_time
from grid import update_grid_r_l, compute_cell_and_relative_position
from file_handling import dump_particle_data
from integration import update_pos_from_vel_BC_PS,\
                        update_cells_and_rel_pos,\
                        compute_divergence_upwind

folder_load = "190508/grid_10_10_spct_4/"
# folder_load = "190507/test1/"
# folder_load = "190508/grid_75_75_spct_20/"
path = simdata_path + folder_load

grid, pos, cells, vel, m_w, m_s, xi, active_ids, removed_ids = \
    load_grid_and_particles_full(0, path)
R_p, w_s, rho_p = compute_R_p_w_s_rho_p(m_w, m_s,
                                        grid.temperature[tuple(cells)] )
R_s = compute_radius_from_mass(m_s, c.mass_density_NaCl_dry)

c2, rel_pos = compute_cell_and_relative_position(pos, grid.ranges, grid.steps)

t = 0.0
dt_sub = 0.1 # s
dt_sub_half = 0.5 * dt_sub

cells_bef = np.copy(cells)
rel_pos_bef = np.copy(rel_pos)

for cnt in range(1000):
    update_pos_from_vel_BC_PS(pos, vel, xi, grid.ranges, dt_sub_half)
    update_cells_and_rel_pos(pos, cells, rel_pos, grid.ranges, grid.steps)

dif_c = cells - cells_bef
dif_rp = rel_pos - rel_pos_bef
# print(cells - cells_bef)
# print(rel_pos - rel_pos_bef)

print(dif_c[dif_c>0])
print(dif_rp[dif_rp>1E-10])
# print(dif_rp)

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
# load grid and particle list from directory
# folder_load = "190508/grid_10_10_spct_4/"
# folder_load = "190510/grid_15_15_spcm_4_4/"
folder_load = "190512/grid_75_75_spcm_0_4/"

# folder_save = "190508/grid_10_10_spct_4/"
# folder_save = "190510/grid_15_15_spcm_4_4/sim2/"
folder_save = "190512/grid_75_75_spcm_0_4/sim8/"
# folder_save = "190511/grid_75_75_spcm_4_4"

t_start = 0.0
t_end = 7200.0+3600.0 # s
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

# no of frames of grid properties T, p, Theta, r_v, r_l, S
no_frames = 36

# full save of grid and particles in time intervals
full_save_time_interval = 600 # s

# particles to be traced, evenly distributed over "active_ids"
# not in use right now, all particle pos are dumped...
# no_trace_ids = 20

# trace particle positions every "trace_every" advection steps "dt"
# trace_every = 4
dump_every = 10

# phys. constants
p_ref = 1.0E5
p_ref_inv = 1.0E-5
# g must be positive = 9.8... or 0.0
# g_set = 0.0
g_set = c.earth_gravity

# adiabatic_index = 1.4
# accomodation_coefficient = 1.0
# condensation_coefficient = 0.0415
### SET END
####################################

#################################### 
### DERIVED
dt_sub = dt/(2 * scale_dt)
dt_sub_half = 0.5 * dt_sub
print("dt_sub = ", dt_sub)

cnt_max = (t_end - t_start) /dt
frame_every = int(math.ceil(cnt_max / no_frames))
full_save_every = int(full_save_time_interval // dt)

path = simdata_path + folder_load

grid, pos, cells, vel, m_w, m_s, xi, active_ids, removed_ids = \
    load_grid_and_particles_full(0, path)

# R_p, w_s, rho_p = compute_R_p_w_s_rho_p(m_w, m_s,
#                                         grid.temperature[tuple(cells)] )
# R_s = compute_radius_from_mass(m_s, c.mass_density_NaCl_dry)

# print(grid.p_ref)          
# print(grid.p_ref_inv)          

sim_paras = [dt, dt_sub, Newton_iter]
sim_par_names = "dt dt_sub Newton_iter"
# sim_paras = [dt, dt_sub, Newton_iter, no_trace_ids]
# sim_par_names = "dt dt_sub Newton_iter no_trace_ids"

sim_para_file = path + "sim_paras_t_" + str(t_start) + ".txt"

with open(sim_para_file, "w") as f:
    f.write( sim_par_names + '\n' )
    for item in sim_paras:
        if type(item) is list or type(item) is np.ndarray:
            for el in item:
                f.write( f'{el} ' )
        else: f.write( f'{item} ' )

### DERIVED END
#################################### 

#################################### INIT
# init grid properties
grid.update_material_properties()
V0_inv = 1.0 / grid.volume_cell
grid.rho_dry_inv =\
    np.ones_like(grid.mass_density_air_dry) / grid.mass_density_air_dry
grid.mass_dry_inv = V0_inv * grid.rho_dry_inv

tau = t_start
taus = []
taus.append(tau)

delta_Q_p = np.zeros_like(grid.temperature)
delta_m_l = np.zeros_like(grid.temperature)
water_removed = 0.0

rel_pos = np.zeros_like(pos)
update_cells_and_rel_pos(pos, cells, rel_pos, grid.ranges, grid.steps)

T_p = np.ones_like(m_w)

ind = index.ind
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

grid_scalar_fields[0] = grid.temperature
grid_scalar_fields[1] = grid.pressure
grid_scalar_fields[2] = grid.potential_temperature
grid_scalar_fields[3] = grid.mass_density_air_dry
grid_scalar_fields[4] = grid.mixing_ratio_water_vapor
grid_scalar_fields[5] = grid.mixing_ratio_water_liquid
grid_scalar_fields[6] = grid.saturation
grid_scalar_fields[7] = grid.saturation_pressure

grid_mat_prop[0] = grid.thermal_conductivity
grid_mat_prop[1] = grid.diffusion_constant
grid_mat_prop[2] = grid.heat_of_vaporization
grid_mat_prop[3] = grid.surface_tension
grid_mat_prop[4] = grid.specific_heat_capacity
grid_mat_prop[5] = grid.viscosity
grid_mat_prop[6] = grid.mass_density_fluid

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
from integration import propagate_particles_subloop_step
####################################
# MAIN SIMULATION LOOP:
path = simdata_path + folder_save
if not os.path.exists(path):
    os.makedirs(path)
# dt = dt_advection
grid_save_times = []
start_time = datetime.now()
for cnt,t in enumerate(np.arange(t_start, t_end, dt)):
    # trace particles in trace_ids every "trace_every" steps (dt_adv-steps)
    if cnt % dump_every == 0:
        dump_particle_data(t, pos, vel, m_w, m_s, xi, grid.temperature,
                           grid.mixing_ratio_water_vapor, path, start_time)
    #     append_trace_locations_and_velocities(t, particle_list_by_id, trace_ids,
    #                                           locations, velocities, trace_times)
    # append "frame" of grid fields every "frame_every" steps,
    # which is calc. from "no_frames"
    if cnt % frame_every == 0:
        update_grid_r_l(m_w, xi, cells,
                               grid.mixing_ratio_water_liquid,
                               grid.mass_dry_inv)
        save_grid_scalar_fields(t, grid, path, start_time)
        grid_save_times.append(int(t))

        # append_frame_to_arrays(t, grid, times, temps, Thetas, press,
        #                        r_vs, r_ls, sats,
        #                        particle_list_by_id, active_ids, trace_ids,
        #                        trace_times, locations, velocities,
        #                        path, start_time, save_to_file = True)
    # append a full save of particles and grid every interval full_save_time_interval
    # if cnt % full_save_every == 0:
    #     save_grid_and_particles_full(t, grid, particle_list_by_id, active_ids, removed_ids, path)
       
    ### one timestep dt:
    # a) dt_sub is set

    # b) for all particles: x_n+1/2 = x_n + h/2 v_n
    # removed_ids_step = []
    update_pos_from_vel_BC_PS(pos, vel, xi, grid.ranges, dt_sub_half)
    update_cells_and_rel_pos(pos, cells, rel_pos, grid.ranges, grid.steps)
    # for ID in active_ids:
    #     particle = particle_list_by_id[ID]
    #     removed = particle.update_location_from_velocity_BC_PS(dt_sub_half)
    #         # multiply by 1.0E-18 in the end
    #     if removed:
    #         removed_ids_step.append(ID)
    #         removed_ids.append(ID)
    #     particle.update_cell_and_relative_location()

    # for ID in removed_ids_step:
    #     active_ids.remove(ID)
    
    # tau += dt_sub_half
################## check
    # c) advection change of r_v and T
    delta_r_v_ad = -dt_sub\
                   * compute_divergence_upwind(grid,
                                               grid.mixing_ratio_water_vapor,
                                               flux_type = 1,
                                               boundary_conditions = [0,1]) \
                   * grid.rho_dry_inv

    delta_Theta_ad = -dt_sub\
                     * compute_divergence_upwind(grid,
                                                 grid.potential_temperature,
                                                 flux_type = 1,
                                                 boundary_conditions = [0,1]) \
                     * grid.rho_dry_inv
################## check
    # d) subloop 1
    # for n_h = 0, ..., N_h-1
    for n_sub in range(scale_dt):

        # i) for all particles
        # updates delta_m_l and delta_Q_p
        # propagate_particles_subloop_step(grid,
        #                              pos, vel, cells, rel_pos, m_w, m_s, xi,
        #                              delta_m_l, delta_Q_p,
        #                              dt_sub, dt_sub_pos=dt_sub,
        #                              Newton_iter=Newton_iter, g_set=g_set)
        propagate_particles_subloop_step(grid_scalar_fields, grid_mat_prop,
                                         grid.velocity,
                                         grid.no_cells, grid.ranges, grid.steps,
                                         pos, vel, cells, rel_pos, m_w, m_s, xi,
                                         T_p,
                                         delta_m_l, delta_Q_p,
                                         dt_sub, dt_sub,
                                         Newton_iter, g_set, ind)
        
        # propagate_particles_subloop(grid, particle_list_by_id, active_ids, removed_ids,
        #                             delta_m_l, delta_Q_p,
        #                             dt_sub, dt_sub_half, dt_sub, Newton_iter, g_set,
        #                             adiabatic_index,
        #                             accomodation_coefficient,
        #                             condensation_coefficient)

        # ii) to vii)
        propagate_grid_subloop_step(grid, delta_Theta_ad, delta_r_v_ad,
                               delta_m_l, delta_Q_p)
        
        grid_scalar_fields[0] = grid.temperature
        grid_scalar_fields[1] = grid.pressure
        grid_scalar_fields[2] = grid.potential_temperature
        grid_scalar_fields[3] = grid.mass_density_air_dry
        grid_scalar_fields[4] = grid.mixing_ratio_water_vapor
        grid_scalar_fields[5] = grid.mixing_ratio_water_liquid
        grid_scalar_fields[6] = grid.saturation
        grid_scalar_fields[7] = grid.saturation_pressure

        grid_mat_prop[0] = grid.thermal_conductivity
        grid_mat_prop[1] = grid.diffusion_constant
        grid_mat_prop[2] = grid.heat_of_vaporization
        grid_mat_prop[3] = grid.surface_tension
        grid_mat_prop[4] = grid.specific_heat_capacity
        grid_mat_prop[5] = grid.viscosity
        grid_mat_prop[6] = grid.mass_density_fluid
        # viii) to ix) included in "propagate_particles_subloop"
        # delta_Q_p.fill(0.0)
        # delta_m_l.fill(0.0)
        
        tau += dt_sub
    # subloop 1 end
####################### CHECK
    # e) advection change of r_v and T for second subloop
    delta_r_v_ad = -2.0 * dt_sub\
                   * compute_divergence_upwind(grid,
                                               grid.mixing_ratio_water_vapor,
                                               flux_type = 1,
                                               boundary_conditions = [0,1]) \
                    * grid.rho_dry_inv - delta_r_v_ad

    delta_Theta_ad = -2.0 * dt_sub\
                     * compute_divergence_upwind(grid,
                                                 grid.potential_temperature,
                                                 flux_type = 1,
                                                 boundary_conditions = [0,1]) \
                    * grid.rho_dry_inv - delta_Theta_ad

    # f) subloop 2
    # for n_h = 0, ..., N_h-2
    for n_sub in range(scale_dt - 1):
        
        # i) for all particles
        # updates delta_m_l and delta_Q_p
        propagate_particles_subloop_step(grid_scalar_fields, grid_mat_prop,
                                            grid.velocity,
                                            grid.no_cells, grid.ranges, grid.steps,
                                            pos, vel, cells, rel_pos, m_w, m_s, xi,
                                            T_p,
                                            delta_m_l, delta_Q_p,
                                            dt_sub, dt_sub,
                                            Newton_iter, g_set, ind)

        # ii) to vii)
        propagate_grid_subloop_step(grid, delta_Theta_ad, delta_r_v_ad,
                               delta_m_l, delta_Q_p)
        
        grid_scalar_fields[0] = grid.temperature
        grid_scalar_fields[1] = grid.pressure
        grid_scalar_fields[2] = grid.potential_temperature
        grid_scalar_fields[3] = grid.mass_density_air_dry
        grid_scalar_fields[4] = grid.mixing_ratio_water_vapor
        grid_scalar_fields[5] = grid.mixing_ratio_water_liquid
        grid_scalar_fields[6] = grid.saturation
        grid_scalar_fields[7] = grid.saturation_pressure

        grid_mat_prop[0] = grid.thermal_conductivity
        grid_mat_prop[1] = grid.diffusion_constant
        grid_mat_prop[2] = grid.heat_of_vaporization
        grid_mat_prop[3] = grid.surface_tension
        grid_mat_prop[4] = grid.specific_heat_capacity
        grid_mat_prop[5] = grid.viscosity
        grid_mat_prop[6] = grid.mass_density_fluid
        # viii) to ix) included in "propagate_particles_subloop"
#         delta_Q_p.fill(0.0)
#         delta_m_l.fill(0.0)
        
        tau += dt_sub
    # subloop 2 end

    # need to add x_n+1/2 -> x_n
    # i) for all particles -> only one for now
    # updates delta_m_l and delta_Q_p
    propagate_particles_subloop_step(grid_scalar_fields, grid_mat_prop,
                                        grid.velocity,
                                        grid.no_cells, grid.ranges, grid.steps,
                                        pos, vel, cells, rel_pos, m_w, m_s, xi,
                                        T_p,
                                        delta_m_l, delta_Q_p,
                                        dt_sub, dt_sub_half,
                                        Newton_iter, g_set, ind)

    # ii) to vii)
    propagate_grid_subloop_step(grid, delta_Theta_ad, delta_r_v_ad,
                           delta_m_l, delta_Q_p)

    grid_scalar_fields[0] = grid.temperature
    grid_scalar_fields[1] = grid.pressure
    grid_scalar_fields[2] = grid.potential_temperature
    grid_scalar_fields[3] = grid.mass_density_air_dry
    grid_scalar_fields[4] = grid.mixing_ratio_water_vapor
    grid_scalar_fields[5] = grid.mixing_ratio_water_liquid
    grid_scalar_fields[6] = grid.saturation
    grid_scalar_fields[7] = grid.saturation_pressure

    grid_mat_prop[0] = grid.thermal_conductivity
    grid_mat_prop[1] = grid.diffusion_constant
    grid_mat_prop[2] = grid.heat_of_vaporization
    grid_mat_prop[3] = grid.surface_tension
    grid_mat_prop[4] = grid.specific_heat_capacity
    grid_mat_prop[5] = grid.viscosity
    grid_mat_prop[6] = grid.mass_density_fluid    
    # viii) to ix) included in "propagate_particles_subloop"
#    delta_Q_p.fill(0.0)
#    delta_m_l.fill(0.0)
    
    tau += dt_sub_half
    taus.append(tau)
    
t += dt
update_grid_r_l(m_w, xi, cells,
               grid.mixing_ratio_water_liquid,
               grid.mass_dry_inv)
save_grid_scalar_fields(t, grid, path, start_time)
# append_frame_to_arrays(t, grid, times,temps,Thetas,press,r_vs,r_ls,sats,
#                            particle_list_by_id, active_ids, trace_ids,
#                            trace_times, locations, velocities,
#                            path, start_time, save_to_file = True)
# MAIN SIMULATION LOOP END

# convert to arrays for plotting
# r_vs = np.array(r_vs)
# r_ls = np.array(r_ls)
# temps = np.array(temps)
# Thetas = np.array(Thetas)
# press = np.array(press)
# sats = np.array(sats)
# fields = [r_vs, r_ls, Thetas, temps, press, sats]

# locations = np.array(locations)
# velocities = np.array(velocities)

# full save at t_end
save_grid_and_particles_full(t, grid, pos, cells, vel, m_w, m_s, xi,
                                 active_ids, removed_ids,
                                 path)
# total water removed by hitting the ground
# convert to kg
# water_removed *= 1.0E-18
end_time = datetime.now()
print("simuation ended:")
print("t_start = ", t_start)
print("t_end = ", t_end)
print("dt = ", dt, "; dt_sub = ", dt_sub)
print("simulation time:")
print(end_time - start_time)

# lh = [ np.amax(r_v) for r_v in r_vs ]
# print('')
# print('time \t sum(rho_dry * r_v) \t sum(rho_dry * r_l) \t sum(rho_dry * (r_l+r_v)) \t  sum(rho_dry * T)')
# for i,rv in enumerate(r_vs):
#     print( times[i],'\t',
#           np.sum(rv*grid.mass_density_air_dry), '\t',
#           np.sum(r_ls[i]*grid.mass_density_air_dry), '\t',
#           np.sum(rv*grid.mass_density_air_dry) +
#           np.sum(r_ls[i]*grid.mass_density_air_dry), '\t',
#           np.sum(Thetas[i]*grid.mass_density_air_dry), '\t',
#           )



#%% PLOTTING

from evaluation import plot_field_frames
from file_handling import load_grid_scalar_fields

folder_load = "190512/grid_75_75_spcm_4_4/"
# folder_load = "190507/test1/"
# folder_load = "190508/grid_75_75_spct_20/"
# folder_save = "190508/grid_10_10_spct_4/"
folder_save = folder_load
fig_name = path + "scalar_fields.png"

reload = False

if reload:
    path = simdata_path + folder_load
    grid, pos, cells, vel, m_w, m_s, xi, active_ids, removed_ids = \
        load_grid_and_particles_full(0, path)
    R_p, w_s, rho_p = compute_R_p_w_s_rho_p(m_w, m_s,
                                            grid.temperature[tuple(cells)] )
    R_s = compute_radius_from_mass(m_s, c.mass_density_NaCl_dry)
    
    t_start = 0
    t_end = 200
    dt = 5

    grid_save_times = np.arange(t_start, t_end + 0.5 * dt, dt).astype(int)

print(grid_save_times)

path = simdata_path + folder_save

fields = load_grid_scalar_fields(path, grid_save_times)

print(fields.shape)

field_ind = np.arange(6)
time_ind = np.arange(0, len(grid_save_times), 2)
print(len(grid_save_times))
print(time_ind)

for idx_t in time_ind:
    print(grid_save_times[idx_t])

no_ticks=[6,6]

plot_field_frames(grid, fields, grid_save_times,
                  field_ind, time_ind, no_ticks, fig_path )

