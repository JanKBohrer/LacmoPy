#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TROPOS LAGRANGIAN CLOUD MODEL
Super-Droplet method in two-dimensional kinetic framework
(Test Case 1 ICMW 2012)
Author: Jan Bohrer (bohrer@tropos.de)
Further contact: Oswald Knoth (knoth@tropos.de)

time integration algorithms
"""

#%% MODULE IMPORTS
import math
import numpy as np
from numba import njit
from datetime import datetime                      

import constants as c
from grid import interpolate_velocity_from_cell_bilinear, update_grid_r_l

import materialproperties as mat
import atmosphere as atm
import microphysics as mp
from relaxation import compute_relaxation_time_profile, compute_relaxation_term

from collision.AON import \
    collision_step_Long_Bott_Ecol_grid_R_all_cells_2D_multicomp_np
from file_handling import \
    dump_particle_data, save_grid_scalar_fields,\
    dump_particle_tracer_data_block,\
    save_grid_and_particles_full,\
    save_sim_paras_to_file, dump_particle_data_all

#%% ADVECTION
            
# compute timestep estimate by courant number (CFL number)
# CFL = dt * ( u_x/dx + u_z/dz ) < cfl_max
# => dt < cfl_max / | ( u_x/dx + u_z/dz ) | for all u_x, u_z
# rhs is smallest for largest term1:
def compute_dt_max_from_CFL(grid):
    term1 = np.abs( grid.velocity[0] / grid.steps[0] )\
            + np.abs (grid.velocity[1] / grid.steps[1]  )
    term1_max = np.amax(term1)
    # define max CFL number
    cfl_max = 0.5
    dt_max = cfl_max / term1_max
    print("dt_max from CFL = ", dt_max)
    return dt_max

@njit()            
def compute_limiter( r_, delta_ = 2.0 ):
    K = (1.0 + 2.0 * r_) * c.one_third
    # fmin: elementwise minimum when comparing two arrays,
    # and NaN is NOT propagated
    return np.maximum(0.0, np.fmin (2.0 * r_, np.fmin( K, delta_ ) ) )

def compute_limiter_from_scalars_upwind( a0, a1, da12 ):
    if ( np.abs(da12) > 1.0E-16 ): 
        limiter_argument = ( a0 - a1 ) / ( da12 )
    else:
        limiter_argument = 1.0
    return compute_limiter(limiter_argument)

@njit()
def compute_limiter_from_scalar_grid_upwind( a0, a1, a2 ):
    r = np.where( np.abs( a1 - a2 ) > 1.0E-8 * np.abs( a0 - a1 ) ,
                  ( a0 - a1 ) / ( a1 - a2 ),
                  ( a0 - a1 ) * 1.0E8 * np.sign( a1 - a2 )
                 )
    return compute_limiter( r )

# computes divergencence of (a * vec) based on vec at the grid cell "surfaces".
# for scalar field quantity a, it is calculated
# div( a * vec ) = d/dx (a * vec_x) + d/dz (a * vec_z)
# method: 3rd order upwind scheme following Hundsdorfer 1996
# vec = velocity OR mass_flux_air_dry
# possible boundary conditions (BC) as list/array in [x,z]:
# 0 = 'periodic',
# 1 = 'solid'
# flux_field = one of
# grid_velocity
# grid_mass_flux_air_dry
def compute_divergence_upwind_np(field, flux_field,
                                 grid_no_cells, grid_steps,
                                 boundary_conditions = np.array([0, 1])):
    Nx = grid_no_cells[0]
    Nz = grid_no_cells[1]
    
    u = flux_field[0][:, 0:-1]
    # transpose w to get same dimensions and routine
    w = np.transpose(flux_field[1][0:-1, :])
    
    N = Nx
    if (boundary_conditions[0] == 0):
        # periodic BC
        field_x =\
            np.vstack( ( np.vstack((field[N-2], field[N-1])),
                          field, np.vstack((field[0], field[1])) ) )
    elif (boundary_conditions[0] == 1):
        # fixed BC
        field_x =\
            np.vstack( ( np.vstack((field[0], field[0])),
                          field, np.vstack((field[N-1], field[N-1])) ) )        
    else:
        print ('ERROR: invalid boundary type' )
    a0 = field_x[2:N+3]
    a1 = field_x[1:N+2]
    a2 = field_x[0:N+1]
    limiter = compute_limiter_from_scalar_grid_upwind( a0, a1, a2 )
    
    # calc f_i_pos = F_i^(+) / u_i
    # where F_i^(+) = flux through surface cell 'i' LEFT BORDER in case u > 0
    
    f_pos = a1 + 0.5 * limiter * (a1 - a2)
    
    # now negative u_i case:
    # a1 and a0 switch places
    # a1 = a0
    # a0 = a1
    a2 = field_x[3:N+4]
    limiter = compute_limiter_from_scalar_grid_upwind( a1, a0, a2 )
    f_neg = a0 + 0.5 * limiter * (a0 - a2)
    
    # np.where to make cases u <> 0
    # flux through LEFT BORDER cell i
    F = np.where( u >= 0.0,
                   u * f_pos,
                   u * f_neg)
    
    if (boundary_conditions[0] == 1):
        F[0] = 0.0
        F[-1] = 0.0
    
    div = (F[1:] - F[0:-1]) / grid_steps[0]
    
    # now for z / w component
    # transpose to get same dimensions and routine
    N = Nz
    field_x = np.transpose( field )
    if (boundary_conditions[1] == 0):

        field_x = np.vstack( ( np.vstack((field_x[N-2], field_x[N-1])), field_x,
                                np.vstack((field_x[0], field_x[1])) ) )
    elif (boundary_conditions[1] == 1):
        field_x =  np.vstack( ( np.vstack((field_x[0], field_x[0])), field_x,
                                np.vstack((field_x[N-1], field_x[N-1])) ) )
    else:
        print ('ERROR: invalid boundary type' )

    a0 = field_x[2:N+3]
    a1 = field_x[1:N+2]
    a2 = field_x[0:N+1]
    limiter = compute_limiter_from_scalar_grid_upwind( a0, a1, a2 )

    # calc f_i_pos = F_i^(+) / u_i
    # where F_i^(+) = flux through surface cell 'i' LEFT BORDER in case u > 0
    
    f_pos = a1 + 0.5 * limiter * (a1 - a2)
    
    # now negative u_i case:
    # a1 and a0 switch places
    # a1 = a0
    # a0 = a1
    a2 = field_x[3:N+4]
    limiter = compute_limiter_from_scalar_grid_upwind( a1, a0, a2 )
    
    f_neg = a0 + 0.5 * limiter * (a0 - a2)
    
    # np.where to make cases w <> 0
    # flux through LEFT BORDER cell
    F = np.where(  w >= 0.0,
                   w * f_pos,
                   w * f_neg )
    if(boundary_conditions[1] == 1):
        F[0] = 0.0
        F[-1] = 0.0
    return div + np.transpose( (F[1:] - F[0:-1]) / grid_steps[1] )
compute_divergence_upwind = njit()(compute_divergence_upwind_np)

#%% GRID PROPAGATION

@njit()
def update_material_properties(grid_scalar_fields, grid_mat_prop):
    grid_mat_prop[0] = mat.compute_thermal_conductivity_air(
                           grid_scalar_fields[0])
    grid_mat_prop[1] = mat.compute_diffusion_constant(grid_scalar_fields[0],
                                                      grid_scalar_fields[1])
    grid_mat_prop[2] = mat.compute_heat_of_vaporization(grid_scalar_fields[0])
    grid_mat_prop[3] = mat.compute_surface_tension_water(grid_scalar_fields[0])
    grid_mat_prop[4] = atm.compute_specific_heat_capacity_air_moist(
                           grid_scalar_fields[4])
    grid_mat_prop[5] = mat.compute_viscosity_air(grid_scalar_fields[0])
    grid_mat_prop[6] = grid_scalar_fields[3]\
                                  * (1.0 + grid_scalar_fields[4])

def propagate_grid_subloop_step_np(grid_scalar_fields, grid_mat_prop,
                                   p_ref, p_ref_inv,
                                   delta_Theta_ad, delta_r_v_ad,
                                   delta_m_l, delta_Q_p,
                                   grid_volume_cell):
    
    # iii) and iv)
    grid_scalar_fields[2] += delta_Theta_ad
    grid_scalar_fields[4] += delta_r_v_ad - delta_m_l*grid_scalar_fields[8]

    # v)
    Theta_over_T = atm.compute_Theta_over_T(grid_scalar_fields[3],
                                            grid_scalar_fields[2],
                                            p_ref_inv)

    # vi)
    grid_scalar_fields[2] += \
         Theta_over_T * (grid_mat_prop[2] * delta_m_l - delta_Q_p) \
        / (c.specific_heat_capacity_air_dry_NTP * grid_scalar_fields[3]
           * grid_volume_cell
           * ( 1.0 + grid_scalar_fields[4] * atm.c_pv_over_c_pd ) )

    # vii) update other grid properties
    p_dry_over_p_ref = atm.compute_p_dry_over_p_ref(grid_scalar_fields[3],
                                                grid_scalar_fields[2],
                                                p_ref_inv)
    grid_scalar_fields[0] = grid_scalar_fields[2]\
                       * p_dry_over_p_ref**atm.kappa_air_dry
    grid_scalar_fields[1] = p_dry_over_p_ref * p_ref\
                    * ( 1 + grid_scalar_fields[4] / atm.epsilon_gc )
    grid_scalar_fields[7] =\
        mat.compute_saturation_pressure_vapor_liquid(grid_scalar_fields[0])
    grid_scalar_fields[6] =\
        atm.compute_pressure_vapor(
            grid_scalar_fields[3] * grid_scalar_fields[4],
            grid_scalar_fields[0] ) / grid_scalar_fields[7]
    update_material_properties(grid_scalar_fields, grid_mat_prop)
propagate_grid_subloop_step = njit()(propagate_grid_subloop_step_np)

#%% PARTICLE PROPAGATION
    
@njit()
def update_T_p(grid_temp, cells, T_p):
    for ID in range(len(T_p)):
        T_p[ID] = grid_temp[cells[0,ID],cells[1,ID]]    
    
def update_cells_and_rel_pos_np(pos, cells, rel_pos,
                                active_ids,
                                grid_ranges, grid_steps):
    x = pos[0][active_ids]
    y = pos[1][active_ids]
    rel_pos[0][active_ids] = x - grid_ranges[0,0] 
    rel_pos[1][active_ids] = y - grid_ranges[1,0]
    cells[0][active_ids] = np.floor(x/grid_steps[0]).astype(np.int64)
    cells[1][active_ids] = np.floor(y/grid_steps[1]).astype(np.int64)
    
    rel_pos[0] = rel_pos[0] / grid_steps[0] - cells[0]
    rel_pos[1] = rel_pos[1] / grid_steps[1] - cells[1]
update_cells_and_rel_pos = njit()(update_cells_and_rel_pos_np)

# update location by one euler step (timestep of the euler step = dt)
# used boundary conditions (BC): periodic in x, solid in z (BC = PS)
# if particle hits bottom, its xi value it set to 0 (int)
def update_pos_from_vel_BC_PS_np(m_w, pos, vel, xi, cells,
                                 water_removed,
                                 id_list, active_ids,
                                 grid_ranges, grid_steps, dt):
    z_min = grid_ranges[1,0]
    pos += dt * vel
    # periodic BC
    # works only if x_grid_min = 0.0
    # there might be trouble if x is exactly = 1500.0 (?),
    # because then x will stay 1500.0 and the calc. cell will be 75,
    # i.e. one too large for eg. grid.centers
    pos[0] = pos[0] % grid_ranges[0,1]
    # dont allow particles to cross z_max (upper domain boundary)
    pos[1] = np.minimum(pos[1], grid_ranges[1,1]*0.999999)
    # if particle hits ground, set z < 0 and active_ID = False
    # this is the indicator for inactive IDs from here on
    for ID in id_list[active_ids]:
        if pos[1,ID] <= z_min:
            # keep z-position constant just below ground
            pos[1,ID] = z_min - 0.01 * grid_steps[1]
            vel[0,ID] = 0.0
            vel[1,ID] = 0.0
            active_ids[ID] = False
            water_removed[0] += xi[ID] * m_w[ID]
            cells[1,ID] = -1
update_pos_from_vel_BC_PS =\
    njit()(update_pos_from_vel_BC_PS_np)

# note: g_set >= 0
def update_vel_impl_np(vel, cells, rel_pos, xi, id_list, active_ids,
                       R_p, rho_p,
                    grid_vel, grid_viscosity, grid_mass_density_fluid, 
                    grid_no_cells, grav, dt):
    vel_f = interpolate_velocity_from_cell_bilinear(cells, rel_pos,
                                                    grid_vel, grid_no_cells)
    for ID in id_list[active_ids]:
        R_p_ = R_p[ID]
        cell = (cells[0,ID], cells[1,ID])
        mu_f = grid_viscosity[cell]
        rho_f_amb = grid_mass_density_fluid[cell]
        dv = vel_f[:,ID] - vel[:,ID]
        vel_dev = np.sqrt(dv[0]*dv[0] + dv[1]*dv[1])
        
        Re_p = mp.compute_particle_reynolds_number(R_p_, vel_dev, rho_f_amb,
                                                   mu_f )
        k_dt = 4.5E12 * dt * mu_f / ( rho_p[ID] * R_p_ * R_p_)
        
        if Re_p > 0.5:
            if Re_p < 1000.0:
                k_dt *= (1 + 0.15 * Re_p**0.687)
            # 0.018333333 = 0.44/24
            else: k_dt *= 0.0183333333333 * Re_p
        vel[0,ID] = (vel[0,ID] + k_dt * vel_f[0,ID]) / (1.0 + k_dt)
        vel[1,ID] = (vel[1,ID] + k_dt * vel_f[1,ID] - dt * grav)\
                    / (1.0 + k_dt)
update_vel_impl = njit()(update_vel_impl_np)

#%% MASS PROPAGATION

w_s_max_NaCl_inv = 1. / mat.w_s_max_NaCl
w_s_max_AS_inv = 1. / mat.w_s_max_AS

# Full Newton method with no_iter iterations,
# the derivative is calculated every iteration
def compute_dml_and_gamma_impl_Newton_full_np(
        dt_sub, no_iter_impl_mass, m_w, m_s, w_s, R_p, T_p, rho_p,
        T_amb, p_amb, S_amb, e_s_amb, L_v, K, D_v, sigma_p, solute_type):
    m_w_effl = m_s * (w_s_max_AS_inv - 1.0)
    
    if solute_type == "AS":
        gamma0, dgamma_dm = mp.compute_mass_rate_and_derivative_AS(
                m_w, m_s, w_s, R_p, T_p, rho_p,
                T_amb, p_amb, S_amb, e_s_amb,
                L_v, K, D_v, sigma_p)
    elif solute_type == "NaCl":
        gamma0, dgamma_dm = mp.compute_mass_rate_and_derivative_NaCl(
                m_w, m_s, w_s, R_p, T_p, rho_p,
                T_amb, p_amb, S_amb, e_s_amb,
                L_v, K, D_v, sigma_p)
    
    dt_sub_times_dgamma_dm = dt_sub * dgamma_dm
    
    denom_inv = np.where(dt_sub_times_dgamma_dm < 0.9,
                             1.0 / (1.0 - dt_sub_times_dgamma_dm),
                             np.ones_like(dt_sub_times_dgamma_dm) * 10.0)
     
    mass_new = np.maximum(m_w_effl, m_w + dt_sub * gamma0 * denom_inv)
    
    for cnt in range(no_iter_impl_mass-1):
        m_p = mass_new + m_s
        w_s = m_s / m_p
        rho = mat.compute_density_AS_solution(w_s, T_p)
        R = mp.compute_radius_from_mass(m_p, rho)
        sigma = mat.compute_surface_tension_AS(w_s,T_p)
        
        if solute_type == "AS":
            gamma, dgamma_dm = mp.compute_mass_rate_and_derivative_AS(
                                   mass_new, m_s, w_s, R, T_p, rho,
                                   T_amb, p_amb, S_amb, e_s_amb,
                                   L_v, K, D_v, sigma)
        elif solute_type == "NaCl":
            gamma, dgamma_dm = mp.compute_mass_rate_and_derivative_NaCl(
                                   mass_new, m_s, w_s, R, T_p, rho,
                                   T_amb, p_amb, S_amb, e_s_amb,
                                   L_v, K, D_v, sigma)
                               
        dt_sub_times_dgamma_dm = dt_sub * dgamma_dm
        
        denom_inv = np.where(dt_sub_times_dgamma_dm < 0.9,
                             1.0 / (1.0 - dt_sub_times_dgamma_dm),
                             np.ones_like(dt_sub_times_dgamma_dm) * 10.0)

        mass_new += ( dt_sub * gamma + m_w - mass_new) * denom_inv
        mass_new = np.maximum( m_w_effl, mass_new )
        
    return mass_new - m_w, gamma0
compute_dml_and_gamma_impl_Newton_full =\
    njit()(compute_dml_and_gamma_impl_Newton_full_np)

#%% UPDATE m_w

def update_m_w_and_delta_m_l_impl_Newton_np(
        grid_temperature, grid_pressure, grid_saturation,
        grid_saturation_pressure, grid_thermal_conductivity, 
        grid_diffusion_constant, grid_heat_of_vaporization,
        grid_surface_tension, cells, m_w, m_s, xi, solute_type,
        id_list, active_ids,
        R_p, w_s, rho_p, T_p, 
        delta_m_l, delta_Q_p, dt_sub,  no_iter_impl_mass):
    delta_m_l.fill(0.0)
    for ID in id_list[active_ids]:
        cell = (cells[0,ID], cells[1,ID])
        T_amb = grid_temperature[cell]
        p_amb = grid_pressure[cell]
        S_amb = grid_saturation[cell]
        e_s_amb = grid_saturation_pressure[cell]
        L_v = grid_heat_of_vaporization[cell]
        K = grid_thermal_conductivity[cell]
        D_v = grid_diffusion_constant[cell]
        sigma_p = mat.compute_surface_tension_solution(w_s[ID], T_p[ID],
                                                       solute_type)
        
        dm, gamma = compute_dml_and_gamma_impl_Newton_full(
                        dt_sub, no_iter_impl_mass, m_w[ID], m_s[ID],
                        w_s[ID], R_p[ID],
                        T_p[ID], rho_p[ID],
                        T_amb, p_amb, S_amb, e_s_amb, L_v, K, D_v, sigma_p,
                        solute_type)
        m_w[ID] += dm

        delta_m_l[cell] += dm * xi[ID]
        
    delta_m_l *= 1.0E-18
update_m_w_and_delta_m_l_impl_Newton =\
    njit()(update_m_w_and_delta_m_l_impl_Newton_np)

#%% new subloop STEP combined

# dt_sub_pos is ONLY for the particle location step x_new = x_old +v*dt_sub_loc
# -->> give the opportunity to adjust this step, usually dt_sub_pos = dt_sub
# at the end of subloop 2: dt_sub_pos = dt_sub_half
# for velocity use m_(n+1) and v_n in |u-v_n| for calcing the Reynolds number
# grid_scalar_fields = np.array([T, p, Theta, rho_dry, r_v, r_l, S, e_s])
# grid_mat_prop = np.array([K, D_v, L, sigma_w, c_p_f, mu_f, rho_f])
def propagate_particles_subloop_step_np(grid_scalar_fields, grid_mat_prop,
                                        grid_velocity,
                                        grid_no_cells, grid_ranges, grid_steps,
                                        pos, vel, cells, rel_pos, m_w, m_s, xi,
                                        water_removed, id_list, active_ids,
                                        T_p, solute_type,
                                        delta_m_l, delta_Q_p,
                                        dt_sub, dt_sub_pos,
                                        no_iter_impl_mass, g_set):
    ### 1. T_p = T_f
    update_T_p(grid_scalar_fields[0], cells, T_p)
            
    ### 2. to 8. compute mass rate and delta_m and update m_w
    
    R_p, w_s, rho_p = mp.compute_R_p_w_s_rho_p(m_w, m_s, T_p, solute_type)
    
    update_m_w_and_delta_m_l_impl_Newton(
        grid_scalar_fields[0], grid_scalar_fields[1],
        grid_scalar_fields[6], grid_scalar_fields[7],
        grid_mat_prop[0], grid_mat_prop[1],
        grid_mat_prop[2], grid_mat_prop[3],
        cells, m_w, m_s, xi, solute_type,
        id_list, active_ids, R_p, w_s, rho_p, T_p,
        delta_m_l, delta_Q_p, dt_sub, no_iter_impl_mass)
    
    ### 9. v_n -> v_n+1
    R_p, w_s, rho_p = mp.compute_R_p_w_s_rho_p_AS(m_w, m_s, T_p)
    update_vel_impl(vel, cells, rel_pos, xi, id_list, active_ids,
                    R_p, rho_p, grid_velocity,
                    grid_mat_prop[5], grid_mat_prop[6], 
                    grid_no_cells, g_set, dt_sub)
    ### 10.
    update_pos_from_vel_BC_PS(m_w, pos, vel, xi, cells,
                              water_removed, id_list,
                              active_ids,
                              grid_ranges, grid_steps, dt_sub_pos)

    ### 11.
    update_cells_and_rel_pos(pos, cells, rel_pos, active_ids,
                             grid_ranges, grid_steps)
propagate_particles_subloop_step = \
    njit()(propagate_particles_subloop_step_np)

#%% SUBLOOP COMBINED
    
def integrate_subloop_n_steps_np(
            grid_scalar_fields, grid_mat_prop, grid_velocity,
            grid_no_cells, grid_ranges, grid_steps,
            grid_volume_cell, p_ref, p_ref_inv,
            pos, vel, cells, rel_pos, m_w, m_s, xi, solute_type,
            dt_col_over_dV, E_col_grid, no_kernel_bins,
            R_kernel_low_log, bin_factor_R_log, no_cols,
            water_removed,
            id_list, active_ids, T_p,
            delta_m_l, delta_Q_p, delta_Theta_ad, delta_r_v_ad,
            dt_sub, dt_sub_pos, no_cond_steps, no_col_steps,
            no_iter_impl_mass, g_set, act_collisions):
    
    if act_collisions and no_col_steps == 1:
        collision_step_Long_Bott_Ecol_grid_R_all_cells_2D_multicomp_np(
            xi, m_w, m_s, vel, grid_scalar_fields[0], cells, grid_no_cells,
            dt_col_over_dV, E_col_grid, no_kernel_bins,
            R_kernel_low_log, bin_factor_R_log, no_cols, solute_type)
    
    no_col_steps_larger_one = no_col_steps > 1
    
    # d) subloop 1
    # for n_h = 0, ..., N_h-1
    for n_sub in range(no_cond_steps):
        if act_collisions and no_col_steps_larger_one:
            collision_step_Long_Bott_Ecol_grid_R_all_cells_2D_multicomp_np(
                xi, m_w, m_s, vel, grid_scalar_fields[0], cells, grid_no_cells,
                dt_col_over_dV, E_col_grid, no_kernel_bins,
                R_kernel_low_log, bin_factor_R_log, no_cols, solute_type)
        # i) for all particles
        # updates delta_m_l and delta_Q_p
        propagate_particles_subloop_step(
            grid_scalar_fields, grid_mat_prop,
            grid_velocity,
            grid_no_cells, grid_ranges, grid_steps,
            pos, vel, cells, rel_pos, m_w, m_s, xi,
            water_removed, id_list, active_ids,
            T_p, solute_type,
            delta_m_l, delta_Q_p,
            dt_sub, dt_sub_pos,
            no_iter_impl_mass, g_set)

        # ii) to vii)
        propagate_grid_subloop_step(grid_scalar_fields, grid_mat_prop,
                                        p_ref, p_ref_inv,
                                        delta_Theta_ad, delta_r_v_ad,
                                        delta_m_l, delta_Q_p,
                                        grid_volume_cell)
        
        # viii) to ix) included in "propagate_particles_subloop"

    # subloop 1 end    

    # put the additional col step here, if no_col_steps > no_cond_steps
    if act_collisions and no_col_steps > no_cond_steps:
        collision_step_Long_Bott_Ecol_grid_R_all_cells_2D_multicomp_np(
            xi, m_w, m_s, vel, grid_scalar_fields[0], cells, grid_no_cells,
            dt_col_over_dV, E_col_grid, no_kernel_bins,
            R_kernel_low_log, bin_factor_R_log, no_cols, solute_type)        
        
integrate_subloop_n_steps = njit()(integrate_subloop_n_steps_np)

#%% INTEGRATE ADVECTION AND CONDENSATION AND COLLISIONS (opt)
#   for one timestep dt = dt_adv

# dt = dt_adv
# no_col_per_adv = 1,2 OR scale_dt_cond * 2
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

def integrate_adv_step_np(
        grid_scalar_fields, grid_mat_prop, grid_velocity,
        grid_mass_flux_air_dry, p_ref, p_ref_inv,
        grid_no_cells, grid_ranges,
        grid_steps, grid_volume_cell,
        pos, vel, cells, rel_pos, m_w, m_s, xi, solute_type,
        water_removed,
        id_list, active_ids, T_p,
        delta_m_l, delta_Q_p,
        dt, dt_sub, dt_sub_half, dt_col_over_dV, scale_dt_cond,
        no_col_per_adv, act_collisions,
        act_relaxation, init_profile_r_v, init_profile_Theta,
        relaxation_time_profile,
        no_iter_impl_mass, g_set,
        E_col_grid, no_kernel_bins,
        R_kernel_low_log, bin_factor_R_log, no_cols):
    ### one timestep dt:
    # a) dt_sub is set
    
    # b) for all particles: x_n+1/2 = x_n + h/2 v_n
    # the velocity is stored from the step before
    # this is why the collision step is done AFTER this position shift
    update_pos_from_vel_BC_PS(m_w, pos, vel, xi, cells,
                              water_removed, id_list,
                              active_ids, grid_ranges, grid_steps,
                              dt_sub_half)
    update_cells_and_rel_pos(pos, cells, rel_pos,
                             active_ids,
                             grid_ranges, grid_steps)

    # c) advection change of r_v and T
    delta_r_v_ad = -dt_sub\
                   * compute_divergence_upwind(
                         grid_scalar_fields[4],
                         grid_mass_flux_air_dry,
                         grid_no_cells, grid_steps,
                         boundary_conditions = np.array([0, 1]))\
                   * grid_scalar_fields[9]
    delta_Theta_ad = -dt_sub\
                   * compute_divergence_upwind(
                         grid_scalar_fields[2],
                         grid_mass_flux_air_dry,
                         grid_no_cells, grid_steps,
                         boundary_conditions = np.array([0, 1]))\
                     * grid_scalar_fields[9]
    
    ### CAN BE REMOVED
    delta_r_v_wo_relax = np.copy(delta_r_v_ad)
    delta_Theta_wo_relax = np.copy(delta_Theta_ad)
    
    # c1) add relaxation source term, if activated
    if act_relaxation:
        # note that a 1D array is added to a 2D array
        # in this case, the 1D array is added to each "row" of the 2D array
        # since the 2D array[x][z], a "z-profile" is added for each FIXED x pos
        delta_r_v_ad += compute_relaxation_term(
                            grid_scalar_fields[4], init_profile_r_v,
                            relaxation_time_profile, dt_sub)        
        delta_Theta_ad += compute_relaxation_term(
                            grid_scalar_fields[2], init_profile_Theta,
                            relaxation_time_profile, dt_sub)
        ### CAN BE REMOVED
        delta_r_v_w_relax = np.copy(delta_r_v_ad)
        delta_Theta_w_relax = np.copy(delta_Theta_ad)
    
    else:
        delta_r_v_w_relax = delta_r_v_wo_relax
        delta_Theta_w_relax = delta_Theta_wo_relax
    # d1) added collision step to the subloop below
    # d) SUBLOOP 1 START
    # for n_h = 0, ..., N_h-1
    # in here, we add e.g. 5 = 5 collision steps if no_col_per_adv > 2
    
    if no_col_per_adv == 1:
        no_col_steps = 0
    elif no_col_per_adv == 2:
        no_col_steps = 1
    else: no_col_steps = scale_dt_cond

    integrate_subloop_n_steps_np(
                grid_scalar_fields, grid_mat_prop, grid_velocity,
                grid_no_cells, grid_ranges, grid_steps,
                grid_volume_cell, p_ref, p_ref_inv,
                pos, vel, cells, rel_pos, m_w, m_s, xi, solute_type,
                dt_col_over_dV, E_col_grid, no_kernel_bins,
                R_kernel_low_log, bin_factor_R_log, no_cols,
                water_removed,
                id_list, active_ids, T_p,
                delta_m_l, delta_Q_p, delta_Theta_ad, delta_r_v_ad,
                dt_sub, dt_sub, scale_dt_cond, no_col_steps,
                no_iter_impl_mass, g_set, act_collisions)
    # SUBLOOP 1 END
    
    # e) advection change of r_v and T for second subloop
    delta_r_v_ad = -2.0 * dt_sub\
                   * compute_divergence_upwind(
                         grid_scalar_fields[4],
                         grid_mass_flux_air_dry,
                         grid_no_cells, grid_steps,
                         boundary_conditions = np.array([0, 1]))\
                   * grid_scalar_fields[9] - delta_r_v_ad
    delta_Theta_ad = -2.0 * dt_sub\
                   * compute_divergence_upwind(
                         grid_scalar_fields[2],
                         grid_mass_flux_air_dry,
                         grid_no_cells, grid_steps,
                         boundary_conditions = np.array([0, 1]))\
                     * grid_scalar_fields[9] - delta_Theta_ad

    # e1) add relaxation source term, if activated
    if act_relaxation:
        # note that a 1D array is added to a 2D array
        # in this case, the 1D array is added to each "row" of the 2D array
        # since the 2D array[x][z], a "z-profile" is added for each FIXED x pos
        delta_r_v_ad += 2. * compute_relaxation_term(
                                grid_scalar_fields[4], init_profile_r_v,
                                relaxation_time_profile, dt_sub)        
        delta_Theta_ad += 2. * compute_relaxation_term(
                                grid_scalar_fields[2], init_profile_Theta,
                                relaxation_time_profile, dt_sub)

    ### f2) added collision step to the subloop below
    # f) SUBLOOP 2 START
    # for n_h = 0, ..., N_h-2
    # in here, we add e.g. 5 = 4 + 1 collision steps if no_col_per_adv > 2
    # the step which is shifted back for particle condensation and acceleration
    # is already made in here
    # in this example, we have 5 + 4 + 1 col steps in total per adv step
    if no_col_per_adv in (1,2):
        no_col_steps = 1
    else: no_col_steps = scale_dt_cond

    integrate_subloop_n_steps_np(
                grid_scalar_fields, grid_mat_prop, grid_velocity,
                grid_no_cells, grid_ranges, grid_steps,
                grid_volume_cell, p_ref, p_ref_inv,
                pos, vel, cells, rel_pos, m_w, m_s, xi, solute_type,
                dt_col_over_dV, E_col_grid, no_kernel_bins,
                R_kernel_low_log, bin_factor_R_log, no_cols,
                water_removed,
                id_list, active_ids, T_p,
                delta_m_l, delta_Q_p, delta_Theta_ad, delta_r_v_ad,
                dt_sub, dt_sub, scale_dt_cond-1, no_col_steps,
                no_iter_impl_mass, g_set, act_collisions)
    # SUBLOOP 2 END

    # add one step, where pos is moved only by half timestep x_n+1/2 -> x_n
    # i) for all particles
    # updates delta_m_l and delta_Q_p as well
    # NOTE that the additional collisions step is already in the method above
    
    propagate_particles_subloop_step_np(
            grid_scalar_fields, grid_mat_prop,
            grid_velocity,
            grid_no_cells, grid_ranges, grid_steps,
            pos, vel, cells, rel_pos, m_w, m_s, xi,
            water_removed, id_list, active_ids,
            T_p, solute_type,
            delta_m_l, delta_Q_p,
            dt_sub, dt_sub_half,
            no_iter_impl_mass, g_set)    

    # ii) to vii)
    propagate_grid_subloop_step(grid_scalar_fields, grid_mat_prop,
                                p_ref, p_ref_inv,
                                delta_Theta_ad, delta_r_v_ad,
                                delta_m_l, delta_Q_p,
                                grid_volume_cell)    
    ### CAN BE REMOVED
    return delta_r_v_wo_relax, delta_r_v_w_relax, \
           delta_Theta_wo_relax, delta_Theta_w_relax
    
integrate_adv_step = njit()(integrate_adv_step_np)

#%% SIMULATE INTERVAL

### SIMULATE INTERVAL COMBINED (choose if collisions, relaxation, ...)
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

# simulates grid and particles for no_adv_steps advection timesteps dt = dt_adv
# with subloop integration with timestep dt_sub
# since coll step is 2 times faster in np-version, there is no njit() here
# the np version is the "normal" version
# test also to use numpy version of integrate_adv_cond_coll_one_adv_step_np
# should not make much of a difference, but who knows?    
def simulate_interval(grid_scalar_fields, grid_mat_prop, grid_velocity,
                      grid_mass_flux_air_dry, p_ref, p_ref_inv,
                      grid_no_cells, grid_ranges,
                      grid_steps, grid_volume_cell,
                      pos, vel, cells, rel_pos, m_w, m_s, xi,
                      solute_type, water_removed,
                      id_list, active_ids, T_p,
                      delta_m_l, delta_Q_p,
                      dt, dt_sub, dt_sub_half, dt_col,
                      scale_dt_cond, no_col_per_adv, no_adv_steps,
                      act_collisions,
                      act_relaxation, init_profile_r_v, init_profile_Theta,
                      relaxation_time_profile,
                      no_iter_impl_mass, g_set,
                      dump_every, trace_ids,
                      traced_vectors, traced_scalars,
                      traced_xi, traced_water,
                      E_col_grid, no_kernel_bins,
                      R_kernel_low_log, bin_factor_R_log, no_cols):
                         # , traced_grid_fields
    dt_col_over_dV = dt_col / grid_volume_cell
    dump_N = 0
    for cnt in range(no_adv_steps):
        if cnt % dump_every == 0:
            traced_vectors[dump_N,0] = pos[:,trace_ids]
            traced_vectors[dump_N,1] = vel[:,trace_ids]
            traced_scalars[dump_N,0] = m_w[trace_ids]
            traced_scalars[dump_N,1] = m_s[trace_ids]
            traced_scalars[dump_N,2] = T_p[trace_ids]
            traced_xi[dump_N] = xi[trace_ids]
            traced_water[dump_N] = water_removed[0]

            dump_N += 1
        
        ### CAN BE REMOVED
        delta_r_v_wo_relax, delta_r_v_w_relax, \
        delta_Theta_wo_relax, delta_Theta_w_relax = \
        integrate_adv_step_np(
            grid_scalar_fields, grid_mat_prop, grid_velocity,
            grid_mass_flux_air_dry, p_ref, p_ref_inv,
            grid_no_cells, grid_ranges,
            grid_steps, grid_volume_cell,
            pos, vel, cells, rel_pos, m_w, m_s, xi, solute_type,
            water_removed,
            id_list, active_ids, T_p,
            delta_m_l, delta_Q_p,
            dt, dt_sub, dt_sub_half, dt_col_over_dV, scale_dt_cond,
            no_col_per_adv, act_collisions,
            act_relaxation, init_profile_r_v, init_profile_Theta,
            relaxation_time_profile,
            no_iter_impl_mass, g_set, 
            E_col_grid, no_kernel_bins,
            R_kernel_low_log, bin_factor_R_log, no_cols)
    
    ### CAN BE REMOVED
    return delta_r_v_wo_relax, delta_r_v_w_relax, \
           delta_Theta_wo_relax, delta_Theta_w_relax
    
    ###### SAVE    
#    dt_col_over_dV = dt_col / grid_volume_cell
##    dt_col_over_dV = 0.5 * dt / grid_volume_cell
#    dump_N = 0
#    for cnt in range(no_adv_steps):
#        if cnt % dump_every == 0:
#            traced_vectors[dump_N,0] = pos[:,trace_ids]
#            traced_vectors[dump_N,1] = vel[:,trace_ids]
#            traced_scalars[dump_N,0] = m_w[trace_ids]
#            traced_scalars[dump_N,1] = m_s[trace_ids]
#            traced_scalars[dump_N,2] = T_p[trace_ids]
#            traced_xi[dump_N] = xi[trace_ids]
#            traced_water[dump_N] = water_removed[0]
#            # traced_grid_fields[dump_N,0] = np.copy(grid_scalar_fields[0])
#            # traced_grid_fields[dump_N,1] = np.copy(grid_scalar_fields[4])
#
#            dump_N += 1
#            
#        integrate_adv_step_np(
#            grid_scalar_fields, grid_mat_prop, grid_velocity,
#            grid_mass_flux_air_dry, p_ref, p_ref_inv,
#            grid_no_cells, grid_ranges,
#            grid_steps, grid_volume_cell,
#            pos, vel, cells, rel_pos, m_w, m_s, xi, solute_type,
#            water_removed,
#            id_list, active_ids, T_p,
#            delta_m_l, delta_Q_p,
#            dt, dt_sub, dt_sub_half, dt_col_over_dV, scale_dt_cond,
#            no_col_per_adv, act_collisions,
#            act_relaxation, init_profile_r_v, init_profile_Theta,
#            relaxation_time_profile,
#            no_iter_impl_mass, g_set, 
#            E_col_grid, no_kernel_bins,
#            R_kernel_low_log, bin_factor_R_log, no_cols)


#%% SIMULATE FULL
### SIMULATE FULL WITH COLLISIONS POSSIBLE (activated by act_collisions bool)
### AND RELAXATION POSSIBLE  (act. by act_relaxation)
    
# NOTE THAT
# not possible with numba: getitem<> from array with (tuple(int64) x 2)
# storing the indices in meaningful classes, modules or dicts is not poss.
# with numby at this point for example i.T for the "T"-index:
# T_p = grid_scalar_fields[i.T][cells[0],cells[1]] leads to an error

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

### INPUT
# takes (grid, pos, vel, cells, m_w, m_s, xi) in a certain state
# and integrates the EOM between t_start and t_end with advection time step dt
# i.e. for a number of (t_start - t_end) / dt advection time steps
# the particle subloop timestep is defined by dt_sub = dt / (2 * scale_dt_cond)
# no_iter_impl_mass (int) = number of iterations in the implicit mass condensation
# algorithm
# g_set = gravity constant (>=0): can be either 0.0 for spin up or 9.8...
# frame_every, dump every, trace_ids, path -> see below at OUTPUT
### OUTPUT
# saves grid fields every "frame_every" advection steps (int)
# saves ("dumps") trajectories and velocities of a number of traced particles
# every "dump every" advection steps (int)
# trace ids can be an np.array = [ID0, ID1, ..] or an integer
# -> if integer: spread this number of tracers uniformly over the ID-range
# path: path to save data, the file notation is chosen internally

def simulate(grid, pos, vel, cells, m_w, m_s, xi, active_ids,
             water_removed, no_cols, simpar, E_col_grid, data_paths):
    
    par_keys = [ 'solute_type', 'dt_adv', 'dt_col', 'scale_dt_cond',
                 'no_col_per_adv', 't_start', 't_end', 'no_iter_impl_mass',
                 'g_set', 'act_collisions',
                 'act_relaxation',
                 'frame_every', 'dump_every',
                 'trace_ids', 'no_kernel_bins', 'R_kernel_low_log',
                 'bin_factor_R_log', 'kernel_type', 'kernel_method',                 
                 'seed_sim', 'simulation_mode' ]
    
    solute_type, dt, dt_col, scale_dt_cond, \
    no_col_per_adv, t_start, t_end, no_iter_impl_mass, \
    g_set, act_collisions, act_relaxation, frame_every, dump_every, \
    trace_ids, no_kernel_bins, R_kernel_low_log, \
    bin_factor_R_log, kernel_type, kernel_method, \
    seed_sim, simulation_mode = \
        [simpar.get(key) for key in par_keys]
    
    output_path = data_paths['output']
    init_path = data_paths['init']
    
    log_file = output_path + f"log_sim_t_{int(t_start)}_{int(t_end)}.txt"
    
    start_time = datetime.now()
    
    # init particles
    rel_pos = np.zeros_like(pos)
    update_cells_and_rel_pos(pos, cells, rel_pos, active_ids,
                             grid.ranges, grid.steps)
    T_p = np.ones_like(m_w)
    
    id_list = np.arange(xi.shape[0])
    
    # init grid properties
    grid.update_material_properties()
    V0_inv = 1.0 / grid.volume_cell
    grid.rho_dry_inv =\
        np.ones_like(grid.mass_density_air_dry) / grid.mass_density_air_dry
    grid.mass_dry_inv = V0_inv * grid.rho_dry_inv
    
    delta_Q_p = np.zeros_like(grid.temperature)
    delta_m_l = np.zeros_like(grid.temperature)
    
    # for relaxation source term: initial profiles and relax time profile
    
    # need cell centers (z)
    # need profile0 of Theta and r_v
    
    if act_relaxation:
        scalars_path = init_path + "arr_file1_" + str(simpar['t_init']) +".npy"
        scalars0 = np.load(scalars_path)
        init_profile_r_v = np.average(scalars0[3], axis = 0)
        init_profile_Theta = np.average(scalars0[7], axis = 0)
        relaxation_time_profile = \
            compute_relaxation_time_profile(grid.centers[1][0])
        
    else:
        init_profile_r_v = None
        init_profile_Theta = None
        relaxation_time_profile = None
        
    # prepare for jit-compilation
    grid_scalar_fields = np.array( ( grid.temperature,
                                     grid.pressure,
                                     grid.potential_temperature,
                                     grid.mass_density_air_dry,
                                     grid.mixing_ratio_water_vapor,
                                     grid.mixing_ratio_water_liquid,
                                     grid.saturation,
                                     grid.saturation_pressure,
                                     grid.mass_dry_inv,
                                     grid.rho_dry_inv ) )
    
    grid_mat_prop = np.array( ( grid.thermal_conductivity,
                                grid.diffusion_constant,
                                grid.heat_of_vaporization,
                                grid.surface_tension,
                                grid.specific_heat_capacity,
                                grid.viscosity,
                                grid.mass_density_fluid ) )
    
    # constants of the grid
    grid_velocity = grid.velocity
    grid_mass_flux_air_dry = grid.mass_flux_air_dry
    grid_ranges = grid.ranges
    grid_steps = grid.steps
    grid_no_cells = grid.no_cells
    grid_volume_cell = grid.volume_cell
    p_ref = grid.p_ref
    p_ref_inv = grid.p_ref_inv
    
    dt_sub = dt/(2 * scale_dt_cond)
    dt_sub_half = 0.5 * dt_sub
    print("dt = ", dt)
    print("dt_col = ", dt_col)
    print("dt_sub = ", dt_sub)
    with open(log_file, "w+") as f:
        f.write(f"simulation mode = {simulation_mode}\n")
        f.write(f"gravitation const = {g_set}\n")
        f.write(f"collisions activated = {act_collisions}\n")
        f.write(f"relaxation activated = {act_relaxation}\n")
        f.write(f"kernel_type = {kernel_type}\n")
        f.write(f"kernel_method = {kernel_method}")
        if kernel_method == "Ecol_const":
            f.write(f", E_col = {E_col_grid}")
        f.write(f"\nsolute material = {solute_type}\n")        
        f.write(f"dt = {dt}\n")    
        f.write(f"dt_col = {dt_col}\n")    
        f.write(f"dt_sub = {dt_sub}\n")    
    cnt_max = (t_end - t_start) /dt
    no_grid_frames = int(math.ceil(cnt_max / frame_every))
    np.save(output_path+"data_saving_paras.npy",
            (frame_every, no_grid_frames, dump_every))
    dt_save = int(frame_every * dt)
    grid_save_times =\
        np.arange(t_start, t_end + 0.5 * dt_save, dt_save).astype(int)
    np.save(output_path+"grid_save_times.npy", grid_save_times)
    
    if isinstance(trace_ids, int):
        trace_id_dist = int(math.floor(len(xi)/(trace_ids)))
        trace_ids = np.arange(int(trace_id_dist*0.5), len(xi), trace_id_dist)
    np.save(output_path+"trace_ids.npy", trace_ids)
    no_trace_ids = len(trace_ids)
    
    dump_factor = frame_every // dump_every
    print("frame_every, no_grid_frames")
    print(frame_every, no_grid_frames)
    print("dump_every, dump_factor")
    print(dump_every, dump_factor)
    with open(log_file, "a") as f:
        f.write( f"frame_every, no_grid_frames\n" )
        f.write( f"{frame_every} {no_grid_frames}\n" )
        f.write( f"dump_every, dump_factor\n" )
        f.write( f"{dump_every} {dump_factor}\n" )
    traced_vectors = np.zeros((dump_factor, 2, 2, no_trace_ids))
    traced_scalars = np.zeros((dump_factor, 3, no_trace_ids))
    traced_xi = np.zeros((dump_factor, no_trace_ids))
    traced_water = np.zeros(dump_factor)
    
    sim_paras = [dt, dt_sub, no_iter_impl_mass, seed_sim]
    sim_par_names = "dt dt_sub no_iter_impl_mass rnd_seed_sim"
    save_sim_paras_to_file(sim_paras, sim_par_names, t_start, output_path)
    
    date = datetime.now()
    print("### simulation starts ###")
    print("start date and time =", date)
    print("sim time =", datetime.now() - start_time)
    print()
    with open(log_file, "a") as f:
        f.write( f"### simulation starts ###\n" )    
        f.write( f"start date and time = {date}\n" )    
        f.write( f"sim time = {date-start_time}\n" )    
    
    ### INTEGRATION LOOP START
    if act_collisions: np.random.seed(seed_sim)
    for frame_N in range(no_grid_frames):
        t = t_start + frame_N * frame_every * dt 
        update_grid_r_l(m_w, xi, cells,
                        grid_scalar_fields[5],
                        grid_scalar_fields[8],
                        active_ids,
                        id_list)
        save_grid_scalar_fields(t, grid_scalar_fields, output_path, start_time)
        dump_particle_data_all(t, pos, vel, cells,
                               m_w, m_s, xi, active_ids, output_path)
        np.save(output_path + f"no_cols_{int(t)}.npy", no_cols)

        ### CAN BE REMOVED
        delta_r_v_wo_relax, delta_r_v_w_relax, \
        delta_Theta_wo_relax, delta_Theta_w_relax = \
        simulate_interval(grid_scalar_fields, grid_mat_prop, grid_velocity,
                      grid_mass_flux_air_dry, p_ref, p_ref_inv,
                      grid_no_cells, grid_ranges,
                      grid_steps, grid_volume_cell,
                      pos, vel, cells, rel_pos, m_w, m_s, xi,
                      solute_type, water_removed,
                      id_list, active_ids, T_p,
                      delta_m_l, delta_Q_p,
                      dt, dt_sub, dt_sub_half, dt_col,
                      scale_dt_cond, no_col_per_adv, frame_every,
                      act_collisions,
                      act_relaxation, init_profile_r_v, init_profile_Theta,
                      relaxation_time_profile,
                      no_iter_impl_mass, g_set,
                      dump_every, trace_ids,
                      traced_vectors, traced_scalars,
                      traced_xi, traced_water,
                      E_col_grid, no_kernel_bins,
                      R_kernel_low_log, bin_factor_R_log, no_cols)
        
        ### CAN BE REMOVED
        tt = t + frame_every * dt
        np.save(output_path + f"delta_r_v_wo_relax_{int(tt)}.npy",
                delta_r_v_wo_relax)
        np.save(output_path + f"delta_r_v_w_relax_{int(tt)}.npy",
                delta_r_v_w_relax)
        np.save(output_path + f"delta_Theta_w_relax_{int(tt)}.npy",
                delta_Theta_w_relax)
        np.save(output_path + f"delta_Theta_wo_relax_{int(tt)}.npy",
                delta_Theta_wo_relax)
            
        time_block =\
            np.arange(t, t + frame_every * dt, dump_every * dt).astype(int)
        dump_particle_tracer_data_block(time_block,
                                 traced_vectors, traced_scalars, traced_xi,
                                 traced_water,
                                 output_path)
    ### INTEGRATION LOOP END
    
    t = t_start + no_grid_frames * frame_every * dt
    update_grid_r_l(m_w, xi, cells,
                   grid_scalar_fields[5],
                   grid_scalar_fields[8],
                   active_ids,
                   id_list)
    save_grid_scalar_fields(t, grid_scalar_fields, output_path, start_time)        
    dump_particle_data_all(t, pos, vel, cells, m_w, m_s, xi, active_ids, output_path)
    np.save(output_path + f"no_cols_{int(t)}.npy", no_cols)
    dump_particle_data(t, pos[:,trace_ids], vel[:,trace_ids],
                       m_w[trace_ids], m_s[trace_ids], xi[trace_ids],
                       grid_scalar_fields[0], grid_scalar_fields[4], output_path)
    
    np.save(output_path + f"water_removed_{int(t)}", water_removed)
    
    # full save at t_end
    grid.temperature = grid_scalar_fields[0]
    grid.pressure = grid_scalar_fields[1]
    grid.potential_temperature = grid_scalar_fields[2]
    grid.mass_density_air_dry = grid_scalar_fields[3]
    grid.mixing_ratio_water_vapor = grid_scalar_fields[4]
    grid.mixing_ratio_water_liquid = grid_scalar_fields[5]
    grid.saturation = grid_scalar_fields[6]
    grid.saturation_pressure = grid_scalar_fields[7]
    
    grid.thermal_conductivity = grid_mat_prop[0]
    grid.diffusion_constant = grid_mat_prop[1]
    grid.heat_of_vaporization = grid_mat_prop[2]
    grid.surface_tension = grid_mat_prop[3]
    grid.specific_heat_capacity = grid_mat_prop[4]
    grid.viscosity = grid_mat_prop[5]
    grid.mass_density_fluid = grid_mat_prop[6]
    
    save_grid_and_particles_full(t, grid, pos, cells, vel, m_w, m_s, xi,
                                     active_ids,
                                     output_path)

    print()
    print("### simulation ended ###")
    print("t_start = ", t_start)
    print("t_end = ", t_end)
    print("dt = ", dt, "; dt_sub = ", dt_sub)
    print("simulation time:")
    end_time = datetime.now()
    print(end_time - start_time)
    with open(log_file, "a") as f:
        f.write( f"### simulation ended ###\n" )    
        f.write( f"t_start = {t_start}\n" )    
        f.write( f"t_end = {t_end}\n" ) 
        f.write( f"dt = {dt}; dt_sub = {dt_sub}\n" ) 
        f.write( f"simulation time = {end_time - start_time}\n" )  
        