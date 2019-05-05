#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 14:28:39 2019

@author: jdesk
"""

import numpy as np
from numba import jit

import constants as c
from atmosphere import compute_Theta_over_T, c_pv_over_c_pd,\
                       compute_p_dry_over_p_ref,\
                       compute_specific_heat_capacity_air_moist,\
                       kappa_air_dry, epsilon_gc,\
                       compute_saturation_pressure_vapor_liquid,\
                       compute_pressure_vapor

################################################################
## ADVECTION
            
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

def propagate_grid_subloop(grid, delta_Theta_ad, delta_r_v_ad,
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
propagate_grid_subloop_par = jit(parallel=True)(propagate_grid_subloop)