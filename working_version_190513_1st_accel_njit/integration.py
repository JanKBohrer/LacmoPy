#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 14:28:39 2019

@author: jdesk
"""

import numpy as np
from numba import njit,jit

import constants as c
from atmosphere import compute_Theta_over_T, c_pv_over_c_pd,\
                       compute_p_dry_over_p_ref,\
                       compute_specific_heat_capacity_air_moist,\
                       kappa_air_dry, epsilon_gc,\
                       compute_saturation_pressure_vapor_liquid,\
                       compute_pressure_vapor

################################################################
#%% ADVECTION
            
# get timestep estimate by courant number (cfl number)
# cfl = dt * ( u_x/dx + u_z/dz ) < cfl_max
# -> dt < cfl_max / | ( u_x/dx + u_z/dz ) | for all u_x, u_z
# rhs is smallest for largest term1:
def compute_dt_max_from_CFL(grid):
    term1 = np.abs( grid.velocity[0] / grid.steps[0] ) + np.abs (grid.velocity[1] / grid.steps[1]  )
    # term1 = np.abs(np.sum(np.array(grid.velocity)/grid.steps))
    term1_max = np.amax(term1)
    # print(term1_max)
    # define
    cfl_max = 0.5
    dt_max = cfl_max / term1_max
    print("dt_max from CFL = ", dt_max)
    return dt_max
            
def compute_limiter( r_, delta_ = 2.0 ):
    K = (1.0 + 2.0 * r_) * c.one_third
    # fmin: elementwise minimum when comparing two arrays,
    # and NaN is NOT propagated
    return np.maximum(0.0, np.fmin (2.0 * r_, np.fmin( K, delta_ ) ) )
#    return np.maximum(0.0, np.minimum (2.0 * r_, np.minimum( K, delta_ ) ) )

def compute_limiter_from_scalars_upwind( a0, a1, da12 ):
#    da12 = a1 - a2
    if ( np.abs(da12) > 1.0E-16 ): 
        limiter_argument = ( a0 - a1 ) / ( da12 )
    else:
        limiter_argument = 1.0
    return compute_limiter(limiter_argument)

def compute_limiter_from_scalar_grid_upwind( a0, a1, a2 ):
#    r = ( a0 - a1 ) / (a1 - a2)
#    da01 = a0 - a1
    r = np.where( np.abs( a1 - a2 ) > 1.0E-8 * np.abs( a0 - a1 ) ,
                  ( a0 - a1 ) / ( a1 - a2 ),
                  ( a0 - a1 ) * 1.0E8 * np.sign( a1 - a2 )
                 )
#        r = np.where( np.abs(da12) > 1.0E-8 * np.abs( a0-a1 ) ,
#                      ( a0 - a1 ) / da12,
#                      ( a0 - a1 ) * 1.0E8 * np.sign(da12)
#                    )
    return compute_limiter( r )
#        r = np.where(   np.abs(delta_field_x) > 1.0E16,
#                    ( field_x[2:Nx+3] - field_x[1:Nx+2] ) / ( delta_field_x ),
#                    1.0
#                )
#        return compute_limiter( r )

#def compute_limiter_from_scalar_grid_upwind( a0, a1, da12 ):
#    r = ( a0 - a1 ) / da12
#    
##        r = np.where( np.abs(da12) > 1.0E-8 * np.abs( a0-a1 ) ,
##                      ( a0 - a1 ) / da12,
##                      ( a0 - a1 ) * 1.0E8 * np.sign(da12)
##                    )
#    return compute_limiter( r )
##        r = np.where(   np.abs(delta_field_x) > 1.0E16,
##                    ( field_x[2:Nx+3] - field_x[1:Nx+2] ) / ( delta_field_x ),
##                    1.0
##                )
##        return compute_limiter( r )
    


#######################################
# computes divergencence of (a * vec) based on vec at the grid cell "surfaces".
# for quantity a, it is calculated
# div( a * vec ) = d/dx (a * vec_x) + d/dz (a * vec_z)
# method: 3rd order upwind scheme following Hundsdorfer 1996
# vec = velocity OR mass_flux_air_dry given by flux_type (s.b.)
# grid has corners and centers
# grid is needed for
# Nx, Nz, grid.steps and grid.velocity or grid.mass_flux_air_dry
# possible boundary conditions as list in [x,z]:
# 0 = 'periodic',
# 1 = 'solid'
# possible flux_type =
# 0 = 'velocity',
# 1 = 'mass_flux_air_dry'

def compute_divergence_upwind(grid, field, flux_type = 1,
                              boundary_conditions = [0, 1]):
    Nx = grid.no_cells[0]
    Nz = grid.no_cells[1]
    
    if (flux_type == 0):
        u = grid.velocity[0][:, 0:-1]
        # transpose w to get same dimensions and routine
        w = np.transpose(grid.velocity[1][0:-1, :])
    elif (flux_type == 1):
        u = grid.mass_flux_air_dry [0][:, 0:-1]
        # transpose w to get same dimensions and routine
        w = np.transpose(grid.mass_flux_air_dry[1][0:-1, :])
    else:
        print ('ERROR: invalid flux type' )
        return 0
    
#    slice_i = [24,24,24,25,25,25,26,26,26] 
#    slice_j = [77,78,79,77,78,79,77,78,79] 
#    slice_i = [21,22,23,24,25,26,27]
#    slice_j = [76,77,78,79]
#    slicer = np.meshgrid( slice_i, slice_j, indexing = 'ij' )
    
    N = Nx
    if (boundary_conditions[0] == 0):
        # periodic bc
        field_x =\
            np.vstack( ( field[N-2], field[N-1], field, field[0], field[1] ) )
    elif (boundary_conditions[0] == 1):
        # fixed bc
        field_x =\
            np.vstack( ( field[0], field[0], field, field[N-1], field[N-1] ) )
    else:
        print ('ERROR: invalid boundary type' )
        return 0
    # i_old range from 0 .. Nx - 1
    # i_new = i_old + 2, and ranges from 0 ... Nx + 3
    # Nx + 2 is last entry of u[i,j]
    # need limiter in every cell of u[i,j] : i = 0 .. Nx, j = 0 .. Nz-1
    # i_old = 0 -> i_new = 2
    # field_x[2:4] is NOT including 4
    a0 = field_x[2:N+3]
    a1 = field_x[1:N+2]
#        a2 = field_x[0:N+1]
    a2 = field_x[0:N+1]
    limiter = compute_limiter_from_scalar_grid_upwind( a0, a1, a2 )
#    da12 = field_x[1:N+2] - field_x[0:N+1]
#    limiter = compute_limiter_from_scalar_grid_upwind( a0, a1, da12 )
#    r = (a0 - a1) / (a1 - a2)
#    print('limiter argument x pos')
#    print(r[slicer])
#    print('limiter x pos')
#    print(limiter[slicer])
    
    # calc f_i_pos = F_i^(+) / u_i
    # where F_i^(+) = flux through surface cell 'i' LEFT BORDER in case u > 0
    
    f_pos = a1 + 0.5 * limiter * (a1 - a2)
    
    # now negative u_i case:
    # a1 and a0 switch places
    # a1 = a0
    # a0 = a1
#    da12 = a0 - field_x[3:N+4]
#    limiter = compute_limiter_from_scalar_grid_upwind( a1, a0, da12 )
    a2 = field_x[3:N+4]
    limiter = compute_limiter_from_scalar_grid_upwind( a1, a0, a2 )
#    print(limiter[slicer])
    f_neg = a0 + 0.5 * limiter * (a0 - a2)
    
    # np.where to make cases u <> 0
    # flux through LEFT BORDER cell i
    F = np.where( u >= 0.0,
                   u * f_pos,
                   u * f_neg)
    
    if (boundary_conditions[0] == 1):
        F[0] = 0.0
        F[-1] = 0.0
    
    div = (F[1:] - F[0:-1]) / grid.steps[0]
#    print( 'div_x' )
#    print(div[24:27,77:])
    
#        print( '' )
#        print( 'div_x' )
#        print( div_x )
    
    ###
    # now for z / w component
    # transpose to get same dimensions and routine
    N = Nz
    field_x = np.transpose( field )
    if (boundary_conditions[1] == 0):
#        field_x = np.vstack( ( field[N-2], field[N-1], field,
#                               field[0], field[1] ) )
        field_x = np.vstack( ( field_x[N-2], field_x[N-1], field_x,
                               field_x[0], field_x[1] ) )
    elif (boundary_conditions[1] == 1):
#        field_x = np.vstack( ( field[0], field[0], field,
#                               field[N-1], field[N-1] ) )
        field_x =  np.vstack( ( field_x[0], field_x[0], field_x,
                                field_x[N-1], field_x[N-1] ) )
    else:
        print ('ERROR: invalid boundary type' )
        return 0

    a0 = field_x[2:N+3]
    a1 = field_x[1:N+2]
#        a2 = field_x[0:N+1]
#    da12 = field_x[1:N+2] - field_x[0:N+1]
#    limiter = compute_limiter_from_scalar_grid_upwind( a0, a1, da12 )
    a2 = field_x[0:N+1]
    limiter = compute_limiter_from_scalar_grid_upwind( a0, a1, a2 )
#    print('limiter z pos')
#    print(limiter[slicer])
    
    # calc f_i_pos = F_i^(+) / u_i
    # where F_i^(+) = flux through surface cell 'i' LEFT BORDER in case u > 0
    
    f_pos = a1 + 0.5 * limiter * (a1 - a2)
    
    # now negative u_i case:
    # a1 and a0 switch places
    # a1 = a0
    # a0 = a1
#    da12 = a0 - field_x[3:N+4]
#    limiter = compute_limiter_from_scalar_grid_upwind( a1, a0, da12 )
    a2 = field_x[3:N+4]
    limiter = compute_limiter_from_scalar_grid_upwind( a1, a0, a2 )
#    print(limiter[slicer])
    
    f_neg = a0 + 0.5 * limiter * (a0 - a2)
    
    # np.where to make cases w <> 0
    # flux through LEFT BORDER cell
    F = np.where(  w >= 0.0,
                   w * f_pos,
                   w * f_neg )
    if(boundary_conditions[1] == 1):
        F[0] = 0.0
        F[-1] = 0.0
#        div += np.transpose( (F[1:] - F[0:-1]) / grid.steps[1] )
    
#        print('')
#        print('div_z')
#        print(div_z)
    
#        divs.append(div_x + np.transpose(div_z))
#    div_z = np.transpose( (F[1:] - F[0:-1]) / grid.steps[1] )
#    print('div_z')
#    print(div_z[24:27,77:])
    return div + np.transpose( (F[1:] - F[0:-1]) / grid.steps[1] )
#    return div + div_z

def compute_new_Theta_and_r_v_advection_and_condensation(
        grid, delta_m_l, delta_Q_con_f, dt,
        flux_type=1, boundary_conditions=[0,1] ):
    
    # RK2 term for T
    # calc k_T first, since it req. r_v at beginning of timestep
    k_T = ( -dt * compute_divergence_upwind(
                      grid, grid.potential_temperature,
                      flux_type=flux_type,
                      boundary_conditions=boundary_conditions) \
            - delta_Q_con_f \
              / ( compute_specific_heat_capacity_air_moist(
                      grid.mixing_ratio_water_vapor)
                  * (1 + grid.mixing_ratio_water_vapor) * grid.volume_cell ) )\
          / grid.mass_density_air_dry
    # RK2 term for r_v
    k_r_v = ( -1.0 * dt * compute_divergence_upwind(
                              grid, grid.mixing_ratio_water_vapor,
                              flux_type = flux_type,
                              boundary_conditions=boundary_conditions) \
              - delta_m_l / grid.volume_cell) / grid.mass_density_air_dry
    # new r_v array
    r_v = grid.mixing_ratio_water_vapor\
          - ( dt * compute_divergence_upwind(
                       grid, grid.mixing_ratio_water_vapor + 0.5 * k_r_v,
                       flux_type = flux_type,
                       boundary_conditions=boundary_conditions) 
              + delta_m_l / grid.volume_cell ) / grid.mass_density_air_dry
    
#     delta_r_v = ( -1.0 * dt * compute_divergence_upwind(
#                                   grid,
#                                   grid.mixing_ratio_water_vapor + 0.5 * k_r_v,
#                                   flux_type = 1) \
#                   - delta_m_l / grid.volume_cell) / grid.mass_density_air_dry
#     grid.mixing_ratio_water_vapor += delta_r_v
    # calc T last, because it req. r_v at end of timestep
    T = grid.potential_temperature\
            - ( dt * compute_divergence_upwind(
                         grid, 
                         grid.potential_temperature + 0.5 * k_T,
                         flux_type = flux_type,
                         boundary_conditions=boundary_conditions) 
                + delta_Q_con_f \
                  / ( compute_specific_heat_capacity_air_moist(r_v)
                      * (1.0 + r_v) * grid.volume_cell ) )\
              / grid.mass_density_air_dry
#     delta_T = ( -1.0 * dt * compute_divergence_upwind(
#                                 grid, 
#                                 grid.temperature + 0.5 * k_T,
#                                 flux_type = 1) \
#             - delta_Q_con_f \
#             / ( compute_specific_heat_capacity_air_moist(r_v) \
#                 * (1 + r_v) * grid.volume_cell ) ) / grid.mass_density_air_dry
    return T, r_v

def propagate_grid_subloop_step(grid, delta_Theta_ad, delta_r_v_ad,
                                delta_m_l, delta_Q_p):
    # iv) and v)
    grid.potential_temperature += delta_Theta_ad
    grid.mixing_ratio_water_vapor += delta_r_v_ad - delta_m_l*grid.mass_dry_inv

    # vi)
    Theta_over_T = compute_Theta_over_T(grid)

    # vii) and viii)
    grid.potential_temperature += \
         Theta_over_T * (grid.heat_of_vaporization * delta_m_l - delta_Q_p) \
        / (c.specific_heat_capacity_air_dry_NTP * grid.mass_density_air_dry
           * grid.volume_cell
           * ( 1.0 + grid.mixing_ratio_water_vapor * c_pv_over_c_pd ) )

    # ix) update other grid properties
    p_dry_over_p_ref = compute_p_dry_over_p_ref(grid)
    grid.temperature = grid.potential_temperature\
                       * p_dry_over_p_ref**kappa_air_dry
#     grid.pressure = specific_gas_constant_air_dry\
#                     * grid.mass_density_air_dry * grid.temperature \
    grid.pressure = p_dry_over_p_ref * grid.p_ref\
                    * ( 1 + grid.mixing_ratio_water_vapor / epsilon_gc )
#         grid.pressure = compute_pressure_ideal_gas(
#                             grid.mass_density_air_dry,
#                             grid.temperature,
#                             specific_gas_constant_air_dry\
#                             *(1 + grid.mixing_ratio_water_vapor / epsilon_gc))
    grid.saturation_pressure =\
        compute_saturation_pressure_vapor_liquid(grid.temperature)
    grid.saturation =\
        compute_pressure_vapor(
            grid.mass_density_air_dry * grid.mixing_ratio_water_vapor,
            grid.temperature ) / grid.saturation_pressure

    grid.update_material_properties()
# IN WORK: if function takes only np arrays as arguments, like Theta, rv,...
# we can try this with njit (!!??) -> then to paralize the numpy functions ?
# OK, then we cant use grid.update... function, hmmm
propagate_grid_subloop_step_par = jit(parallel=True)(propagate_grid_subloop_step)

#%% PARTICLE PROPAGATION

@njit()
def update_cells_and_rel_pos(pos, cells, rel_pos, grid_ranges, grid_steps):
    x = pos[0]
    y = pos[1]
    # cells = np.empty( (2,len(x)) , dtype = np.int64)
    # rel_pos = np.empty( (2,len(x)) , dtype = np.float64 )
    rel_pos[0] = x - grid_ranges[0,0] # gridranges = arr [[x_min, x_max], [y_min, y_max]]
    rel_pos[1] = y - grid_ranges[1,0]
    cells[0] = np.floor(x/grid_steps[0]).astype(np.int64)
    cells[1] = np.floor(y/grid_steps[1]).astype(np.int64)
    
    rel_pos[0] = rel_pos[0] / grid_steps[0] - cells[0]
    rel_pos[1] = rel_pos[1] / grid_steps[1] - cells[1]
    # return cells, rel_pos

#   update location by one euler step (timestep of the euler step = dt)
#   using BC: periodic in x, solid in z (BC = PS)
#   if particle hits bottom, its xi value it set to 0 (int)
#   requires "vel" to be right
# TESTS:
# no_spt = 400
# update_particle_locations_from_velocities_BC_PS_np:
# update_particle_locations_from_velocities_BC_PS:
# update_particle_locations_from_velocities_BC_PS_par:
# best =  124.1 us; worst =  161.5 us; mean = 134.7 +- 13.7 us
# best =  4.704 us; worst =  5.287 us; mean = 4.986 +- 0.222 us
# best =  17.74 us; worst =  29.68 us; mean = 21.34 +- 5.3 us
# no_spt = 112500
# update_particle_locations_from_velocities_BC_PS_np:
# update_particle_locations_from_velocities_BC_PS:
# update_particle_locations_from_velocities_BC_PS_par:
# best =  3.021e+04 us; worst =  3.136e+04 us; mean = 3.064e+04 +- 3.5e+02 us
# best =  1.071e+03 us; worst =  1.082e+03 us; mean = 1.076e+03 +- 4.62 us
# best =  641.5 us; worst =  743.3 us; mean = 699.8 +- 33.3 us
def update_pos_from_vel_BC_PS_np(pos, vel, xi, grid_ranges, dt):
    z_min = grid_ranges[1,0]
    # removed = False
    pos += dt * vel
    # periodic BC
    # works only if x_min_grid =  0.0
    # there might be trouble if x is exactly = 1500.0,
    # because then x will stay 1500.0 and the calc. cell will be 75, i.e. one too large for eg.g grid.centers
    pos[0] = pos[0] % grid_ranges[0,1]
    pos[1] = np.minimum(pos[1], grid_ranges[1,1]*0.999999)
    # if particle hits ground, set xi = 0
    # this is the indicator that the particle has "vanished"
    # also to use for collision later
    for ID,xi_ in enumerate(xi):
        if xi_ != 0:
            if pos[1,ID] <= z_min:
                xi[ID] = 0
                pos[1,ID] = z_min
                vel[0,ID] = 0.0
                vel[1,ID] = 0.0

#            removed_ids.append(ID)
#            active_ids.remove(ID)
update_pos_from_vel_BC_PS =\
    njit()(update_pos_from_vel_BC_PS_np)
update_pos_from_vel_BC_PS_par =\
    njit(parallel = True)(update_pos_from_vel_BC_PS_np)

from microphysics import compute_R_p_w_s_rho_p,\
                         compute_dml_and_gamma_impl_Newton_full,\
                         compute_particle_reynolds_number
from grid import interpolate_velocity_from_cell_bilinear
import math

# g_set >= 0 !!
@njit()
def update_vel_impl(vel, cells, rel_pos, xi, R_p, rho_p,
                    grid_vel, grid_viscosity, grid_mass_density_fluid, 
                    grid_no_cells, grav, dt):
    vel_f = interpolate_velocity_from_cell_bilinear(cells, rel_pos,
                                                    grid_vel, grid_no_cells)
    for ID, xi_ in enumerate(xi):
        if xi_ != 0:
            R_p_ = R_p[ID]
            cell = (cells[0,ID], cells[1,ID])
            mu_f = grid_viscosity[cell]
            rho_f_amb = grid_mass_density_fluid[cell]
            dv = vel_f[:,ID] - vel[:,ID]
            vel_dev = math.sqrt(dv[0]*dv[0] + dv[1]*dv[1])
            
            Re_p = compute_particle_reynolds_number(R_p_, vel_dev, rho_f_amb,
                                     mu_f )
            k_dt = 4.5E12 * dt * mu_f / ( rho_p[ID] * R_p_ * R_p_)
            
            if Re_p > 0.5:
                if Re_p < 1000.0:
                    k_dt *= (1 + 0.15 * Re_p**0.687)
                # 0.018333333  = 0.44/24
                else: k_dt *= 0.0183333333333 * Re_p
            vel[0,ID] = (vel[0,ID] + k_dt * vel_f[0,ID]) / (1.0 + k_dt)
            vel[1,ID] = (vel[1,ID] + k_dt * vel_f[1,ID] - dt * grav)\
                        / (1.0 + k_dt)

# runtime test:
# no_spt = 400
# update_m_w_and_delta_m_l_impl_Newton_np: repeats = 7 no reps =  100
# update_m_w_and_delta_m_l_impl_Newton: repeats = 7 no reps =  1000
# update_m_w_and_delta_m_l_impl_Newton_par: repeats = 7 no reps =  1000
# best =  2.856e+03 us; worst =  6.707e+03 us; mean = 3.427e+03 +- 1.45e+03 us
# best =  390.7 us; worst =  751.5 us; mean = 443.3 +- 1.36e+02 us
# best =  391.3 us; worst =  721.1 us; mean = 439.3 +- 1.24e+02 us
# no_spt = 11250
# update_m_w_and_delta_m_l_impl_Newton_np: repeats = 7 no reps =  100
# update_m_w_and_delta_m_l_impl_Newton: repeats = 7 no reps =  1000
# update_m_w_and_delta_m_l_impl_Newton_par: repeats = 7 no reps =  1000
# best =  8.1e+04 us; worst =  8.393e+04 us; mean = 8.213e+04 +- 9.13e+02 us
# best =  1.102e+04 us; worst =  1.136e+04 us; mean = 1.115e+04 +- 1.56e+02 us
# best =  1.103e+04 us; worst =  1.142e+04 us; mean = 1.114e+04 +- 1.29e+02 us            
def update_m_w_and_delta_m_l_impl_Newton_np(
        grid_temperature, grid_pressure, grid_saturation,
        grid_saturation_pressure, grid_thermal_conductivity, 
        grid_diffusion_constant, grid_heat_of_vaporization,
        grid_surface_tension, cells, m_w, m_s, xi, R_p, w_s, rho_p, T_p,
        delta_m_l, delta_Q_p, dt_sub,  Newton_iter):
    delta_m_l.fill(0.0)
    ### ACTIVATE
    # delta_Q_p.fill(0.0)
    for ID, xi_ in enumerate(xi):
        if xi_ != 0:
            cell = (cells[0,ID], cells[1,ID])
            T_amb = grid_temperature[cell]
            p_amb = grid_pressure[cell]
            S_amb = grid_saturation[cell]
            e_s_amb = grid_saturation_pressure[cell]
            # rho_f_amb = grid.mass_density_fluid[cell]
            L_v = grid_heat_of_vaporization[cell]
            K = grid_thermal_conductivity[cell]
            D_v = grid_diffusion_constant[cell]
            # sigma w is right now calc. with the ambient temperature...
            # can be changed to the particle temp, if tracked
            sigma_w = grid_surface_tension[cell]
            # c_p_f = grid.specific_heat_capacity[cell]
            # mu_f = grid.viscosity[cell]
            
            # req. w_s, R_p, rho_p, T_p (check)
            dm, gamma = compute_dml_and_gamma_impl_Newton_full(
                            dt_sub, Newton_iter, m_w[ID], m_s[ID],
                            w_s[ID], R_p[ID],
                            T_p[ID], rho_p[ID],
                            T_amb, p_amb, S_amb, e_s_amb, L_v, K, D_v, sigma_w)
            # ### 3.
            # ### ACTIVATE
            # T_eq_old = particle.equilibrium_temperature
    
            # ### 4.
            # # T_eq_new req. radius, which req. density,
            # which req. self.temperature, self.mass_fraction_solute
            # ### ACTIVATE
            # particle.equilibrium_temperature =\
            #   T_amb + L_v * mass_rate * 1.0E-12\
            #           / (4.0 * np.pi * particle.radius * K)
    
            # ### 5.
            # ### ACTIVATE
            # delta_Q_p[cell] += particle.compute_heat_capacity()\
            #                    * particle.multiplicity \
            #                    * particle.mass\
            #                    * (particle.equilibrium_temperature - T_eq_old)

            ### 6.
            m_w[ID] += dm
    
            ### 7. 
            delta_m_l[cell] += dm * xi_
            
            ### 8.
            ### ACTIVATE
            # delta_Q_p[cell] += particle.compute_heat_capacity()\
            #                    * delta_m * particle.multiplicity\
            #                    * (particle.equilibrium_temperature - T_amb)    
    delta_m_l *= 1.0E-18
update_m_w_and_delta_m_l_impl_Newton =\
    njit()(update_m_w_and_delta_m_l_impl_Newton_np)
update_m_w_and_delta_m_l_impl_Newton_par =\
    njit(parallel=True)(update_m_w_and_delta_m_l_impl_Newton_np)

@njit()
def update_T_p(grid_temp, cells, T_p):
    for ID in range(len(T_p)):
        # T_p_ = grid_temp[cells[0,ID],cells[1,ID]]
        T_p[ID] = grid_temp[cells[0,ID],cells[1,ID]]

# import index as idx
# # from index import idx_T
# ind = idx.ind
# i_T = ind.T
# dt_sub_pos is ONLY for the particle location step x_new = x_old + v * dt_sub_loc
# -->> give the opportunity to adjust this step, usually dt_sub_pos = dt_sub
# at the end of subloop 2: dt_sub_pos = dt_sub_half
# UPDATE 25.02.19: use full Newton implicit for delta_m
# UPDATE 26.02: use one step implicit for velocity and mass, for velocity use m_(n+1) and v_n in |u-v_n| for calcing the Reynolds number
# IN WORK: NOTE THAT dt_sub_half is not used anymore...
# grid_scalar_fields = np.array([T, p, Theta, rho_dry, r_v, r_l, S, e_s])
# grid_mat_prop = np.array([K, D_v, L, sigma_w, c_p_f, mu_f, rho_f])
def propagate_particles_subloop_step_np(grid_scalar_fields, grid_mat_prop,
                                        grid_velocity,
                                        grid_no_cells, grid_ranges, grid_steps,
                                        pos, vel, cells, rel_pos, m_w, m_s, xi,
                                        T_p,
                                        delta_m_l, delta_Q_p,
                                        dt_sub, dt_sub_pos,
                                        Newton_iter, g_set, ind):
    # removed_ids_step = []
    # delta_m_l.fill(0.0)
    ### 1. T_p = T_f
    # use this in vectorized version, because the subfunction compute_radius ...
    # in compute_R_p_w_s_rho_p (below) is defined by vectorize() and not by jit
    # update_T_p(grid_scalar_fields[ind["T"]], cells, T_p)
    # update_T_p(grid_scalar_fields[idx_T], cells, T_p)
    # update_T_p(grid_scalar_fields[i_T], cells, T_p)
    update_T_p(grid_scalar_fields[0], cells, T_p)
    # update_T_p(grid_scalar_fields[i_T], cells, T_p)
    # update_T_p(grid_scalar_fields[ind[0]], cells, T_p)
    # update_T_p(grid_scalar_fields[0], cells, T_p)
    
    # not possible with numba: getitem<> from array with (tuple(int64) x 2)
    # T_p = grid_scalar_fields[i.T][cells[0],cells[1]]
    
    ### 2. to 8. compute mass rate and delta_m and update m_w
    # use this in vectorized version, because the subfunction compute_radius ...
    # is defined by vectorize() and not by jit
    R_p, w_s, rho_p = compute_R_p_w_s_rho_p(m_w, m_s, T_p)
    
    update_m_w_and_delta_m_l_impl_Newton(
        grid_scalar_fields[0], grid_scalar_fields[1],
        grid_scalar_fields[6], grid_scalar_fields[7],
        grid_mat_prop[0], grid_mat_prop[1],
        grid_mat_prop[2], grid_mat_prop[3],
        cells, m_w, m_s, xi, R_p, w_s, rho_p, T_p,
        delta_m_l, delta_Q_p, dt_sub, Newton_iter)
    # update_m_w_and_delta_m_l_impl_Newton(
    #     grid_scalar_fields[ind["T"]], grid_scalar_fields[ind["p"]],
    #     grid_scalar_fields[ind["S"]], grid_scalar_fields[ind["es"]],
    #     grid_mat_prop[ind["K"]], grid_mat_prop[ind["Dv"]],
    #     grid_mat_prop[ind["L"]], grid_mat_prop[ind["sigmaw"]],
    #     cells, m_w, m_s, xi, R_p, w_s, rho_p, T_p,
    #     delta_m_l, delta_Q_p, dt_sub, Newton_iter)
    
    # update_m_w_and_delta_m_l_impl_Newton(
    #     grid_scalar_fields[i.T], grid_scalar_fields[i.p],
    #     grid_scalar_fields[i.S],
    #     grid_scalar_fields[i.es], grid_mat_prop[i.K], 
    #     grid_mat_prop[i.Dv], grid_mat_prop[i.L],
    #     grid_mat_prop[i.sigmaw], cells, m_w, m_s, xi, R_p, w_s, rho_p, T_p,
    #     delta_m_l, delta_Q_p, dt_sub, Newton_iter)
    
    ### 9. v_n -> v_n+1
    # (CHANGED TO FULL TIMESTEP WITH k_d(m_(n+1), x_(n+1/2), |u_n+1/2 - v_n|) )
    # req. R_p -> self.density + m_p (changed)
    # -> mass_fraction (changed) + temperature (changed)
    # req. cell (check, unchanged) + location (check, unchanged)
    # use this in vectorized version, because the subfunction compute_radius...
    # is defined by vectorize() and not by jit
    R_p, w_s, rho_p = compute_R_p_w_s_rho_p(m_w, m_s, T_p)
    ### ACTIVATE
    # particle.temperature = particle.equilibrium_temperature
    # g_set >= 0 !!
    update_vel_impl(vel, cells, rel_pos, xi, R_p, rho_p, grid_velocity,
                    grid_mat_prop[5], grid_mat_prop[6], 
                    grid_no_cells, g_set, dt_sub)
    # update_vel_impl(vel, cells, rel_pos, xi, R_p, rho_p, grid_velocity,
    #                 grid_mat_prop[ind["muf"]], grid_mat_prop[ind["rhof"]], 
    #                 grid_no_cells, g_set, dt_sub)
    ### 10.
    update_pos_from_vel_BC_PS(pos, vel, xi, grid_ranges, dt_sub_pos)
    ### 11.
    update_cells_and_rel_pos(pos, cells, rel_pos, grid_ranges, grid_steps)
propagate_particles_subloop_step = njit()(propagate_particles_subloop_step_np)
propagate_particles_subloop_step_par =\
    njit(parallel = True)(propagate_particles_subloop_step_np)

# propagate_particles_subloop_step_par =\
#     jit(parallel=True)(propagate_particles_subloop_step)
    
### OLD unjitted version
# def propagate_particles_subloop_step(grid,
#                                      pos, vel, cells, rel_pos, m_w, m_s, xi,
#                                      delta_m_l, delta_Q_p,
#                                      dt_sub, dt_sub_pos,
#                                      Newton_iter, g_set):
#     # removed_ids_step = []
#     # delta_m_l.fill(0.0)
#     ### 1. T_p = T_f
#     # use this in vectorized version, because the subfunction compute_radius ...
#     # in compute_R_p_w_s_rho_p (below) is defined by vectorize() and not by jit
#     T_p = grid.temperature[cells[0],cells[1]]
#     ### 2. to 9. compute mass rate and delta_m and update m_w
#     # use this in vectorized version, because the subfunction compute_radius ...
#     # is defined by vectorize() and not by jit
#     R_p, w_s, rho_p = compute_R_p_w_s_rho_p(m_w, m_s, T_p)
#     update_m_w_and_delta_m_l_impl_Newton(
#         grid.temperature, grid.pressure, grid.saturation,
#         grid.saturation_pressure, grid.thermal_conductivity, 
#         grid.diffusion_constant, grid.heat_of_vaporization,
#         grid.surface_tension, cells, m_w, m_s, xi, R_p, w_s, rho_p, T_p,
#         delta_m_l, delta_Q_p, dt_sub, Newton_iter)
    
#     ### 9. v_n -> v_n+1  (CHANGED TO FULL TIMESTEP WITH k_d(m_(n+1), x_(n+1/2), |u_n+1/2 - v_n|) )
#     # req. R_p -> self.density + m_p (changed) -> mass_fraction (changed) + temperature (changed)
#     # req. cell (check, unchanged) + location (check, unchanged)
#     # use this in vectorized version, because the subfunction compute_radius ...
#     # is defined by vectorize() and not by jit
#     R_p, w_s, rho_p = compute_R_p_w_s_rho_p(m_w, m_s, T_p)
#     ### ACTIVATE
#     # particle.temperature = particle.equilibrium_temperature
#     # g_set >= 0 !!
#     update_vel_impl(vel, cells, rel_pos, xi, R_p, rho_p,
#                     grid.velocity, grid.viscosity, grid.mass_density_fluid, 
#                     grid.no_cells, g_set, dt_sub)
#     ### 10.
#     update_pos_from_vel_BC_PS(pos, vel, xi, grid.ranges, dt_sub_pos)
#     # removed = particle.update_location_from_velocity_BC_PS(dt_sub_loc)
#         # multiply by 1.0E-18 in the end
#     # if removed:
#         # removed_ids_step.append(ID)
#         # removed_ids.append(ID)
# #             active_ids.remove(ID)
# #             water_removed += particle.mass_water * particle.multiplicity
#     ### 11.
#     update_cells_and_rel_pos(pos, cells, rel_pos, grid.ranges, grid.steps)
#     # particle.update_cell_and_relative_location()
# ######
# ######            
# ######            
# #         particle = particle_list_by_id[ID]
# #         cell = tuple(particle.cell)
        
# #         T_amb = grid.temperature[cell]
# #         p_amb = grid.pressure[cell]
# #         S_amb = grid.saturation[cell]
# #         e_s_amb = grid.saturation_pressure[cell]
# #         rho_f_amb = grid.mass_density_fluid[cell]
# #         K = grid.thermal_conductivity[cell]
# #         D_v = grid.diffusion_constant[cell]
# #         L_v = grid.heat_of_vaporization[cell]
# #         # sigma w is right now calc. with the ambient temperature... can be changed to the particle temp, if tracked
# #         sigma_w = grid.surface_tension[cell]
# #         c_p_f = grid.specific_heat_capacity[cell]
# #         mu_f = grid.viscosity[cell]
   
# #         ### 1. T_p = T_f
# #         particle.temperature = T_amb

# #         ### 2. compute mass rate and delta_m
# #         # inserted Newton implicit with "Newton_iter" iterations here
# #         # req. self.density + m_p (check) -> mass_fraction (check) + temperature (changed right above)
# #         # req. cell (check) + location (check)   
# #         particle.density = particle.compute_density()
# #         particle.radius = particle.compute_radius_from_mass()
# #         delta_m, mass_rate = \
# #             compute_delta_water_liquid_and_mass_rate_implicit_Newton_full(dt_sub, Newton_iter,
# #                                                                           particle.mass_water, particle.mass_solute,
# #                                                                           particle.mass, particle.mass_fraction_solute, particle.radius,
# #                                                                           particle.temperature, particle.density,
# #                                                                           T_amb, p_amb,
# #                                                                           S_amb, e_s_amb,
# #                                                                           D_v,
# #                                                                           K,
# #                                                                           c_p_f,
# #                                                                           L_v,
# #                                                                           sigma_w,
# #                                                                           adiabatic_index,
# #                                                                           accomodation_coefficient,
# #                                                                           condensation_coefficient)
# # ######################################### check
# #          # replaced by full Newton method
# # #        delta_m, mass_rate = \
# # #            compute_delta_water_liquid_and_mass_rate_imex_linear( dt_sub,
# # #                                                                  particle.mass_water, # in femto gram
# # #                                                                  particle.mass_solute, 
# # #                                                                  particle.temperature,
# # #                                                                  T_amb, p_amb,
# # #                                                                  S_amb, e_s_amb,
# # #                                                                  D_v, K, c_p_f,
# # #                                                                  adiabatic_index,
# # #                                                                  accomodation_coefficient,
# # #                                                                  condensation_coefficient, 
# # #                                                                  L_v )

# #         ### 3.
# #         ### ACTIVATE
# # #        T_eq_old = particle.equilibrium_temperature

# #         ### 4.
# #         # T_eq_new req. radius, which req. density, which req. self.temperature, self.mass_fraction_solute
# #         ### ACTIVATE
# # #        particle.equilibrium_temperature = T_amb + L_v * mass_rate * 1.0E-12 / (4.0 * np.pi * particle.radius * K)

# #         ### 5.
# #         ### ACTIVATE
# # #        delta_Q_p[cell] += particle.compute_heat_capacity() * particle.multiplicity \
# # #                           * particle.mass * (particle.equilibrium_temperature - T_eq_old)

# #         ### 6. 
# #         particle.mass_water += delta_m
# #         particle.mass += delta_m
# #         particle.mass_fraction_solute = particle.mass_solute / particle.mass

# #         ### 7. 
# #         delta_m_l[cell] += delta_m * particle.multiplicity
# # ####################### check
# #         ### 8.
# #         ### ACTIVATE
# # #        delta_Q_p[cell] += particle.compute_heat_capacity() * delta_m * particle.multiplicity\
# # #                           * (particle.equilibrium_temperature - T_amb)

# #         ### 9. v_n -> v_n+1  (CHANGED TO FULL TIMESTEP WITH k_d(m_(n+1), x_(n+1/2), |u_n+1/2 - v_n|) )
# #         # req. R_p -> self.density + m_p (changed) -> mass_fraction (changed) + temperature (changed)
# #         # req. cell (check, unchanged) + location (check, unchanged)   
# #         ### ACTIVATE
# # #             particle.temperature = particle.equilibrium_temperature
# #         particle.density = particle.compute_density()
# #         particle.radius = particle.compute_radius_from_mass()
# #         particle.velocity = particle.compute_velocity_new_implicit(dt_sub, rho_f_amb, mu_f, g_set)
        
# #         ### 10.
# #         removed = particle.update_location_from_velocity_BC_PS(dt_sub_loc)
# #             # multiply by 1.0E-18 in the end
# #         if removed:
# #             removed_ids_step.append(ID)
# #             removed_ids.append(ID)
# # #             active_ids.remove(ID)
# # #             water_removed += particle.mass_water * particle.multiplicity
# #         ### 11.
# #         particle.update_cell_and_relative_location()
    
# #     for ID in removed_ids_step:
# #         active_ids.remove(ID)
            
# #     delta_m_l *= 1.0E-18
# #     ### ACTIVATE
# # #    delta_Q_p *= 1.0E-18
# propagate_particles_subloop_step_par = jit(parallel=True)(propagate_particles_subloop_step)

