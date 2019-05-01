#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:23:02 2019
copied everything from /python/help_functions.py
@author: jdesk
"""

# HELP FUNCTIONS
import pickle
import numpy as np
import matplotlib.pyplot as plt
from physical_relations_and_constants import *
#import grid_class

#import particle_class


# VECTOR LENGTH
# vector must be a list or an array of more than one component
# the components may be scalars or np.arrays
def vector_length(vector):
    r = 0.0
    for el in vector:
        r += el*el
    return np.sqrt(r)

# MAGNITUDE OF THE DEVIATION BETWEEN TWO VECTORS
# dev = sqrt( (v1 - v2)**2 )
# v1 and v2 may be lists or np.arrays of the same dimensions > 1
def deviation_magnitude_between_vectors(v1,v2):
    dev = 0.0
    
    for i, comp1 in enumerate(v1):
        dev += (v2[i]-comp1)*(v2[i]-comp1)
    
    return np.sqrt(dev)

# print to file
# note that there must be an open file handle present
# e.g. position_file = open("position_vs_time.txt","w")
# which must be closed at some time later
def print_position_from_weights_vs_time_to_file(t, i, j, weight_x, weight_y, file_handle):
    pos_x, pos_y = compute_position_from_weights(i, j, weight_x, weight_y)
    
    string = f'{t:.6} {pos_x:.6} {pos_y:.6}\n'
    file_handle.write(string)

def print_position_vs_time_to_file(t, x, y, file_handle):
    
    string = f'{t:.6} {x:.6} {y:.6}\n'
    file_handle.write(string)

def print_data_vs_time_to_file(t, x, y, u, v, file_handle):
    
    string = f'{t:.6} {x:.6} {y:.6} {u:.6} {v:.6}\n'
    file_handle.write(string)

######

def compute_no_grid_cells_from_step_sizes( gridranges_list_, stepsizes_list_ ):
    no_cells = []
    for i, range_i in enumerate(gridranges_list_):
        no_cells.append( int(np.ceil( (range_i[1] - range_i[0]) / stepsizes_list_[i] ) ) )
    return no_cells

def weight_velocities_linear(i, j, a, b, u_n, v_n):
    return a * u_n[i + 1, j] + (1 - a) * u_n[i, j], \
            b * v_n[i, j + 1] + (1 - b) * v_n[i, j]
            
# OLD version with inverted meshgrid... changed 10.01.19
#def weight_velocities_linear(i, j, a, b, u_n, v_n):
#    return a * u_n[j, i + 1] + (1 - a) * u_n[j, i], \
#            b * v_n[j + 1, i] + (1 - b) * v_n[j, i]
    
# f is a 2D scalar array, giving values for the grid cell [i,j]
# i.e. f[i,j] = scalar value of cell [i,j]
# the interpolation is given for the normalized position [a,b] in a cell with 4 corners 
# [i, j+1]    [i+1, j+1]
# [i, j]      [i+1, j]
# where a = 0..1,  b = 0..1   
def bilinear_weight(i, j, a, b, f):
    return a * (b * f[i+1, j+1] + (1 - b) * f[i+1, j]) + \
            (1 - a) * (b * f[i, j+1] + (1 - b) * f[i, j])

# f is a 2D scalar array, giving values for the grid cell [j,i]
# i.e. f[j,i] = scalar value of cell [i,j]
# the interpolation is given for the normalized position [a,b] in a cell with 4 corners 
# [i, j+1]    [i+1, j+1]
# [i, j]      [i+1, j]
# where a = 0..1,  b = 0..1   
# OLD version with inverted meshgrid... changed 10.01.19
#def bilinear_weight_inverse_indices(i, j, a, b, f):
#    return a * (b * f[j+1, i+1] + (1 - b) * f[j, i+1]) + \
#            (1 - a) * (b * f[j+1, i] + (1 - b) * f[j, i])

################################################################
## ADVECTION
            
def compute_limiter( r_, delta_ = 2.0 ):
    K = (1.0 + 2.0 * r_) * 0.333333333333333
    # fmin: elementwise minimum when comparing two arrays, and NaN is NOT propagated
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
#                    ( field_x[2:Nx+3] - field_x[1:Nx+2] ) / ( delta_field_x  ),
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
##                    ( field_x[2:Nx+3] - field_x[1:Nx+2] ) / ( delta_field_x  ),
##                    1.0
##                )
##        return compute_limiter( r )
    


#######################################
# computes divergencence of ( a * vec ) based on vec at the grid cell "surfaces".
# for quantity a, calculated is div( a * vev ) = d/dx (a * vec_x) + d/dz (a * vec_z)
# method: 3rd order upwind scheme following Hundsdorfer 1996
# vec = velocity OR mass_flux_air_dry given by flux_type (s.b.)
# grid has corners and centers
# grid is needed for Nx, Nz, grid.steps and grid.velocity or grid.mass_flux_air_dry
# possible boundary conditions as list in [x,z]:
# 0 = 'periodic',
# 1 = 'solid'
# possible flux_type =
# 0 = 'velocity',
# 1 = 'mass_flux_air_dry'

def compute_divergence_upwind(grid, field, flux_type = 1, boundary_conditions = [0, 1]):
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
        field_x = np.vstack( ( field[N-2], field[N-1], field, field[0], field[1] ) )
    elif (boundary_conditions[0] == 1):
        # fixed bc
        field_x = np.vstack( ( field[0], field[0], field, field[N-1], field[N-1] ) )
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
#        field_x = np.vstack( ( field[N-2], field[N-1], field, field[0], field[1] ) )
        field_x =  np.vstack( ( field_x[N-2], field_x[N-1], field_x, field_x[0], field_x[1] ) )
    elif (boundary_conditions[1] == 1):
#        field_x = np.vstack( ( field[0], field[0], field, field[N-1], field[N-1] ) )
        field_x =  np.vstack( ( field_x[0], field_x[0], field_x, field_x[N-1], field_x[N-1] ) )
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
        
        
################################################################

#### OLD, dont know if working
# grid has corners and centers
# j = mass flux dry air given at the cell surface centers
# fields is an np ARRAY of scalar fields given at the cell centers, note that it has to be an array not a list
# fields = np.array([field0, field1, ...])
# fields[0,i,j] = field0[i,j]

#def compute_divergence_mass_flux_upwind(grid, fields):
#    Fx = np.zeros_like( fields )
#    Fz = np.zeros_like( fields )
#    Nx = grid.no_cells[0]
#    Nz = grid.no_cells[1]
##    for i in range(grid.no_cells[0]):        
#    for i in range(Nx):        
##        for j in range(grid.no_cells[1]):
#        for j in range(Nz):
#            jx = grid.mass_flux_air_dry[0][i,j]
#            if( jx >= 0.0 ):
#                i0 = i
#                # modulo for periodic bc in x
#                i1 = (i - 1) % Nx
#                i2 = (i - 2) % Nx
#            else:
#                # modulo for periodic bc in x
#                i0 = (i - 1) % Nx
#                i1 = i
#                i2 = (i + 1) % Nx
#            
#            a0 =  fields[:, i0, j]
#            a1 =  fields[:, i1, j]
#            a2 =  fields[:, i2, j]
#            
##            print(i,j, 'a_x')
##            print(a0,a1,a2)
#            
##            limiter_argument = ( field[i0] - field[i1] ) / ( field[i1] - field[i2] )
#            da12 = a1 - a2
#            limiter_argument = ( a0 - a1 ) / ( da12 )
#            limiter = compute_limiter(limiter_argument)
#            
#            Fx[:,i,j] = jx * ( a1 + 0.5 * limiter * ( da12 ) )
#            
#            jz = grid.mass_flux_air_dry[1][i,j]
#            if( jz >= 0.0 ):
#                j0 = j
#                # no inflow in z-domain boundaries
#                if ( j > 1 ):
#                    j1 = (j - 1) # % Nz
#                    j2 = (j - 2) # % Nz
#                else:
#                    j1 = 0
#                    j2 = 0
#            else:
#                # no inflow in z-domain boundaries
#                # could also use modulo as for pbc, since
#                # for j = 0 or j = Nz-1 and jz < 0 it doesnt matter because Fz[:,0] = Fz[:,Nz] = 0 anyways below
#                if j == 0:
#                    j0 = 0
#                else:
#                    j0 = (j - 1) # % Nz
#                j1 = j
#                if j == Nz - 1:
#                    j2 = Nz - 1
#                else:
#                    j2 = (j + 1) # % Nz
#            
##            print(i,j, 'index js')
##            print(j0,j1,j2)
#            
#            a0 =  fields[:, i, j0]
#            a1 =  fields[:, i, j1]
#            a2 =  fields[:, i, j2]
#            
##            a0 =  field[i, j0]
##            a1 =  field[i, j1]
##            a2 =  field[i, j2]
#            
#            da12 = a1 - a2
#            limiter_argument = ( a0 - a1 ) / ( da12 )
#            limiter = compute_limiter(limiter_argument)
#            
#            Fz[:,i,j] = jz * ( a1 + 0.5 * limiter * ( da12 ) )
#            
##    dx = grid.steps[0]
##    dz = grid.steps[1]
#    
##    print('Fx')
##    print(Fx)
##    print('Fz')
##    print(Fz)
#    
#    # IN WORK
#    # concet along axis = 1 (second axis)
#    # Fx[:,0][:] is dim = (no_fields, Nz array)
#    # broadcast to (no_fields, 1, Nz),
#    # then we can stack along second axis
#    Fx = np.hstack( ( Fx, Fx[:, 0][:, None] ) )
##    print('Fx')
##    print(Fx)
#    # no inflow in z-domain boundaries
#    Fz[:,:,0] = 0.0
#    # add a column to each of the elements of Fz, which are 2dim arrays 
#    Fz = np.dstack( ( Fz, Fz[:, :, 0][:, : None] )  )
##    print('Fz')
##    print(Fz)
#    
#    div_x =  ( Fx[:,1:, :] - Fx[:,:-1, :] ) / grid.steps[0]
#    div_z =  ( Fz[:,:, 1:] - Fz[:,:, :-1] ) / grid.steps[1]
#    
#    return div_x + div_z

################################################################
def compute_new_Theta_and_r_v_advection_and_condensation(grid, delta_m_l, delta_Q_con_f, dt, flux_type=1, boundary_conditions = [0,1]  ):
    
    # RK2 term for T
    # calc k_T first, since it req. r_v at beginning of timestep
    k_T = ( -1.0 * dt * compute_divergence_upwind(grid, grid.potential_temperature,
                                                  flux_type = flux_type,
                                                  boundary_conditions=boundary_conditions) \
            - delta_Q_con_f \
            / ( compute_specific_heat_capacity_air_moist(grid.mixing_ratio_water_vapor) \
                * (1 + grid.mixing_ratio_water_vapor) * grid.volume_cell ) ) / grid.mass_density_air_dry
    # RK2 term for r_v
    k_r_v = ( -1.0 * dt * compute_divergence_upwind(grid, grid.mixing_ratio_water_vapor,
                                                    flux_type = flux_type, boundary_conditions=boundary_conditions) \
              - delta_m_l / grid.volume_cell) / grid.mass_density_air_dry
    # new r_v array
    r_v = grid.mixing_ratio_water_vapor - (1.0 * dt * compute_divergence_upwind(grid,
                                              grid.mixing_ratio_water_vapor + 0.5 * k_r_v,
                                              flux_type = flux_type,
                                              boundary_conditions=boundary_conditions) \
                  + delta_m_l / grid.volume_cell) / grid.mass_density_air_dry
    
#     delta_r_v = ( -1.0 * dt * compute_divergence_upwind(grid,
#                                               grid.mixing_ratio_water_vapor + 0.5 * k_r_v,
#                                               flux_type = 1) \
#                   - delta_m_l / grid.volume_cell) / grid.mass_density_air_dry
#     grid.mixing_ratio_water_vapor += delta_r_v
    # calc T last, because it req. r_v at end of timestep
    T = grid.potential_temperature - (1.0 * dt * compute_divergence_upwind(grid, 
                                                      grid.potential_temperature + 0.5 * k_T,
                                                      flux_type = flux_type,
                                                      boundary_conditions=boundary_conditions) \
            + delta_Q_con_f \
            / ( compute_specific_heat_capacity_air_moist(r_v) \
                * (1 + r_v) * grid.volume_cell ) ) / grid.mass_density_air_dry
#     delta_T = ( -1.0 * dt * compute_divergence_upwind(grid, 
#                                                       grid.temperature + 0.5 * k_T,
#                                                       flux_type = 1) \
#             - delta_Q_con_f \
#             / ( compute_specific_heat_capacity_air_moist(r_v) \
#                 * (1 + r_v) * grid.volume_cell ) ) / grid.mass_density_air_dry
    return T, r_v    

################################################################


# working for 1 field only
#def compute_divergence_upwind(grid, field):
#    Fx = np.zeros_like( field )
#    Fz = np.zeros_like( field )
#    Nx = grid.no_cells[0]
#    Nz = grid.no_cells[1]
##    for i in range(grid.no_cells[0]):        
#    for i in range(Nx):        
##        for j in range(grid.no_cells[1]):
#        for j in range(Nz):
#            jx = grid.mass_flux_air_dry[0][i,j]
#            if( jx >= 0.0 ):
#                i0 = i
#                # modulo for periodic bc in x
#                i1 = (i - 1) % Nx
#                i2 = (i - 2) % Nx
#            else:
#                # modulo for periodic bc in x
#                i0 = (i - 1) % Nx
#                i1 = i
#                i2 = (i + 1) % Nx
#            
#            a0 =  field[i0, j]
#            a1 =  field[i1, j]
#            a2 =  field[i2, j]
#            
#            print(i,j, 'a_x')
#            print(a0,a1,a2)
#            
##            limiter_argument = ( field[i0] - field[i1] ) / ( field[i1] - field[i2] )
#            da12 = a1 - a2
#            limiter_argument = ( a0 - a1 ) / ( da12 )
#            limiter = compute_limiter(limiter_argument)
#            
#            Fx[i,j] = jx * ( a1 + 0.5 * limiter * ( a1 - a2 ) )
#            
#            jz = grid.mass_flux_air_dry[1][i,j]
#            if( jz >= 0.0 ):
#                j0 = j
#                # no inflow in z-domain boundaries
#                if ( j > 1 ):
#                    j1 = (j - 1) # % Nz
#                    j2 = (j - 2) # % Nz
#                else:
#                    j1 = 0
#                    j2 = 0
#            else:
#                # no inflow in z-domain boundaries
#                # could also use modulo as for pbc, since
#                # for j = 0 or j = Nz-1 and jz < 0 it doesnt matter because Fz[:,0] = Fz[:,Nz] = 0 anyways below
#                if j == 0:
#                    j0 = 0
#                else:
#                    j0 = (j - 1) # % Nz
#                j1 = j
#                if j == Nz - 1:
#                    j2 = Nz - 1
#                else:
#                    j2 = (j + 1) # % Nz
#            
##            print(i,j, 'index js')
##            print(j0,j1,j2)
#            
#            a0 =  field[i, j0]
#            a1 =  field[i, j1]
#            a2 =  field[i, j2]
#            
#            da12 = a1 - a2
#            limiter_argument = ( a0 - a1 ) / ( da12 )
#            limiter = compute_limiter(limiter_argument)
#            
#            Fz[i,j] = jz * ( a1 + 0.5 * limiter * ( a1 - a2 ) )
#            
##    dx = grid.steps[0]
##    dz = grid.steps[1]
#    
#    print('Fx')
#    print(Fx)
#    print('Fz')
#    print(Fz)
#    
#    # IN WORK
#    Fx = np.vstack( ( Fx, Fx[0] ) )
##    print('Fx')
##    print(Fx)
#    # no inflow in z-domain boundaries
#    Fz[:,0] = 0.0
#    Fz = np.hstack( ( Fz, Fz[:,0][:, None] )  )
##    print('Fz')
##    print(Fz)
#    
#    div_x =  ( Fx[1:, :] - Fx[:-1, :] ) / grid.steps[0]
#    div_z =  ( Fz[:, 1:] - Fz[:, :-1] ) / grid.steps[0]
#    
#    return [div_x, div_z]
################################################################


################################################################
### PLOTTING

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
                         colorbar_fraction_=0.046, colorbar_pad_ = 0.02 ):
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
    ax.set_xticks( np.linspace( tick_ranges_[0,0], tick_ranges_[0,1], no_ticks_[0] ) )
    ax.set_yticks( np.linspace( tick_ranges_[1,0], tick_ranges_[1,1], no_ticks_[1] ) )
    plt.colorbar(CS, fraction=colorbar_fraction_ , pad=colorbar_pad_)

############ imshow plot:
    # note that we need 'xy' meshgrid arangement, i.e. reverse indices for imshow...
    
#from mpl_toolkits import axes_grid1
#from mpl_toolkits.axes_grid1 import make_axes_locatable
#
#def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
#    """Add a vertical color bar to an image plot."""
#    divider = axes_grid1.make_axes_locatable(im.axes)
#    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
#    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
#    current_ax = plt.gca()
#    cax = divider.append_axes("right", size=width, pad=pad)
#    plt.sca(current_ax)
#    return im.axes.figure.colorbar(im, cax=cax, **kwargs)
#
#fig, ax = plt.subplots(figsize=(8,8))
#
#contours = plt.contour(*grid.centers, grid.pressure, 20, colors = 'black')
#ax.clabel(contours, inline=True, fontsize=8)
#
#IS = ax.imshow(  grid.pressure,
#            extent = [grid.centers[0].min(), grid.centers[0].max(),
#                      grid.centers[1].min(), grid.centers[1].max()],
#            origin = 'lower',
#            cmap = plt.cm.coolwarm,
#            aspect = 1)
##             aspect = 'equal')
##             aspect = 'auto' )
## ax.axis(aspect = 'image')
## divider = make_axes_locatable(ax)
## cax = divider.append_axes("right", size="2%", pad = 0.1)
## fig.colorbar(IS, cax=cax)
## fig.colorbar(IS)
#fig.colorbar(IS,fraction=0.046, pad=0.02)
## fig.colorbar(IS,fraction=0.046, pad=0.04)
## add_colorbar(IS)
## IS = ax.imshow( *grid.centers, grid.temperature, 50
##                  vmax = grid.temperature.max(),
##                  vmin = grid.temperature.min())
## plt.colorbar(IS)

############## imshow plot end


########################## file saving
#from grid_class import Grid
#from particle_class import Particle

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
###





    
### fitting

# fitting function of two variables
# rho_sol(w_s, T)
# fit for values between
# T = 0 .. 60°C
# w_s = 0 .. 0.22


def rho_NaCl_fit(X, *p):
    w,T = X
    return p[0] + p[1] * w + p[2] * T + p[3] * T*T + p[4] * w * T 

def rho_NaCl_fit2(X, *p):
    w,T = X
    return p[0] + p[1] * w + p[2] * T + p[3] * w*w + p[4] * w * T + p[5] * T*T
def lin_fit(x,a,b):
    return a + b*x
def quadratic_fit(x,a,b,c):
    return a + b*x + c*x*x
def quadratic_fit_shift(x,y0,a,x0):
    return y0 + a * (x - x0) * (x - x0)
def quadratic_fit_shift_set(T,a):
    return rho0h + a * (T - T0h) * (T - T0h)
def quadratic_fit_shift_set2(T,a,rhomax):
    return rhomax + a * (T - T0h) * (T - T0h)
def cubic_fit(x, a,b,c,d):
    return a + b*x + c*x**2 + d*x**3


