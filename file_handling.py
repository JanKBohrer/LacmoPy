#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TROPOS LAGRANGIAN CLOUD MODEL
Super-Droplet method in two-dimensional kinematic framework
(Test Case 1 ICMW 2012)
Author: Jan Bohrer (bohrer@tropos.de)
Further contact: Oswald Knoth (knoth@tropos.de)

FILE HANDLING, WRITING AND READING OF DATA TO/FROM HARD DISC
"""

#%% MODULE IMPORTS
import numpy as np
from datetime import datetime

from grid import Grid

#%% GENERAL WRITE FUNCTIONS
def save_sim_paras_to_file(sim_paras, sim_par_names, t, path):
    """Write simulation parameters to file
    
    Parameters
    ----------
    sim_paras: list
        List of variables, which are written to file
    sim_par_names: list of str
        List of strings with the parameter names
    t: float
        Time used in the filename
    path: str
        Path to directory, where the file is stored
        Provide in format '/path/to/directory/'
    
    """
    
    sim_para_file = path + 'sim_paras_t_' + str(int(t)) + '.txt'
    with open(sim_para_file, 'w') as f:
        f.write( sim_par_names + '\n' )
        for item in sim_paras:
            if type(item) is list or type(item) is np.ndarray:
                for el in item:
                    f.write( f'{el} ' )
            else: f.write( f'{item} ' )

#%% SAVE AND LOAD PARTICLE DATA
def save_particles_to_files(pos, cells, vel, m_w, m_s, xi,
                            active_ids, 
                            vector_filename, scalar_filename, cells_filename, 
                            xi_filename,
                            active_ids_filename):
    """Write data of all particles to hard disc in .npy format
    
    Parameters
    ----------
    pos: ndarray, dtype=float
        2D array, where
        pos[0] = 1D array of horizontal coordinates (m)
        pos[1] = 1D array of vertical coordinates (m)
        (pos[0,n], pos[1,n]) is the position of particle 'n'
    cells: ndarray, dtype=int
        2D array, holding the particle cell indices, i.e.
        cells[0] = 1D array of horizontal indices
        cells[1] = 1D array of vertical indices
        (cells[0,n], cells[1,n]) gives the cell of particle 'n'    
    vel: ndarray, dtype=float
        2D array, where
        vel[0] = 1D array of horizontal velocity components (m/s)
        vel[1] = 1D array of vertical velocity components (m/s)
        (vel[0,n], vel[1,n]) is the velocity of particle 'n'
    m_w: ndarray, dtype=float
        1D array holding the particle water masses (1E-18 kg)
        This array gets updated by the function.
    m_s: ndarray, dtype=float
        1D array holding the particle solute masses (1E-18 kg)
    xi: ndarray, dtype=float
        1D array holding the particle multiplicities
    active_ids: ndarray, dtype=bool
        1D mask-array. Each particle gets a flag 'True' or 'False', defining
        if it still resides in the simulation domain or has already hit the
        ground and is thereby removed from the simulation
    vector_filename: str
        Full path for the filename of vector data
    scalar_filename: str
        Full path for the filename of scalar data
    cells_filename: str
        Full path for the filename of cells-data
    xi_filename: str
        Full path for the filename of multiplicities
    active_ids_filename: str
        Full path for the filename of active ids
    
    """
    
    np.save(vector_filename, [pos, vel] )
    np.save(cells_filename, cells )
    np.save(scalar_filename, [m_w, m_s] )
    np.save(xi_filename, xi )
    np.save(active_ids_filename, active_ids)

def dump_particle_data(t, pos, vel, m_w, m_s, xi, T_grid, rv_grid, path):
    """Dump particle tracer data to hard disc in .npy format
    
    Parameters
    ----------
    t: float
        Time used in filenames
    pos: ndarray, dtype=float
        2D array, where
        pos[0] = 1D array of horizontal coordinates (m)
        pos[1] = 1D array of vertical coordinates (m)
        (pos[0,n], pos[1,n]) is the position of particle 'n'
    vel: ndarray, dtype=float
        2D array, where
        vel[0] = 1D array of horizontal velocity components (m/s)
        vel[1] = 1D array of vertical velocity components (m/s)
        (vel[0,n], vel[1,n]) is the velocity of particle 'n'
    m_w: ndarray, dtype=float
        1D array holding the particle water masses (1E-18 kg)
        This array gets updated by the function.
    m_s: ndarray, dtype=float
        1D array holding the particle solute masses (1E-18 kg)
    xi: ndarray, dtype=float
        1D array holding the particle multiplicities
    T_grid: ndarray, dtype=float
        2D array holding the discretized temperature of the atmosphere
    rv_grid: ndarray, dtype=float
        2D array holding the discretized water vapor mixing ratio
        of the atmosphere
    path: str
        Path to directory, where data is stored.
        Provide in format '/path/to/directory/'
    
    """
    
    filename_pt_vec = path + 'particle_vector_data_' + str(int(t)) + '.npy'
    filename_pt_scal = path + 'particle_scalar_data_' + str(int(t)) + '.npy'
    filename_pt_xi = path + 'particle_xi_data_' + str(int(t)) + '.npy'
    filename_grid = path + 'grid_T_rv_' + str(int(t)) + '.npy'
    np.save(filename_pt_vec, (pos, vel) )
    np.save(filename_pt_scal, (m_w, m_s) )
    np.save(filename_pt_xi, xi )
    np.save(filename_grid, (T_grid, rv_grid) )
    print('particle data saved at t =', t)

def load_particle_data(path, save_times):
    """Load particle tracer data from hard disc
    
    Parameters
    ----------
    path: str
        Path to directory, where data is stored.
        Provide in format '/path/to/directory/'
    save_times: ndarray
        1D array of times, at which particle data should be loaded
    
    Returns
    -------
    vec_data: ndarray
        vec_data[0] = 2D array with particle positions
        vec_data[1] = 2D array with particle velocities
    scal_data: ndarray
        scal_data[0] = 1D array with particle water masses
        scal_data[1] = 1D array with particle solute masses
    xi_data: ndarray
        1D array with particle multiplicities
    
    """
    
    vec_data = []
    scal_data = []
    xi_data = []
    for t in save_times:
        filename_pt_vec = path + 'particle_vector_data_' + str(int(t)) + '.npy'
        filename_pt_scal = path + 'particle_scalar_data_' + str(int(t)) +'.npy'
        filename_pt_xi = path + 'particle_xi_data_' + str(int(t)) + '.npy'
        vec = np.load(filename_pt_vec)
        scal = np.load(filename_pt_scal)
        xi = np.load(filename_pt_xi)
        vec_data.append(vec)
        scal_data.append(scal)
        xi_data.append(xi)
    vec_data = np.array(vec_data)
    scal_data = np.array(scal_data)
    xi_data = np.array(xi_data)
    return vec_data, scal_data, xi_data

def dump_particle_tracer_data_block(time_block,
                                    traced_vectors, traced_scalars, traced_xi,
                                    traced_water,
                                    path):
    """Dump particle tracer data at multiple times to hard disc in .npy format
    
    Parameters
    ----------
    time_block: ndarray
        1D array of times, where particle tracer data is provided
    traced_vectors: ndarray, dtype=float
        Array which collects vector data (position, velocity) for the tracked
        particles (tracers) at every 'dump'
        traced_vectors[dump_counter,0] = 2D position-array of tracers
        traced_vectors[dump_counter,1] = 2D velocity-array of tracers
    traced_scalars: ndarray, dtype=float
        Array which collects scalar data (m_w, m_s, T_p) for the tracked
        particles (tracers) at every 'dump'
        traced_scalars[dump_counter,0] = 1D array of particle water masses
        traced_scalars[dump_counter,1] = 1D array of particle solute masses
        traced_scalars[dump_counter,2] = 1D array of particle temperatures
    traced_xi: ndarray, dtype=float
        Array which collects the multiplicities of the tracked particles
    traced_water: ndarray, dtype=float
        Array, which stores the total water, which is removed
        from the system in time intervals of the tracer particle dumps.    
    path: str
        Path to directory, where data is stored.
        Provide in format '/path/to/directory/'
    
    """
    
    t = int(time_block[0])
    filename_pt_vec = path + 'particle_vector_data_' + str(t) + '.npy'
    filename_pt_scal = path + 'particle_scalar_data_' + str(t) + '.npy'
    filename_pt_xi = path + 'particle_xi_data_' + str(t) + '.npy'
    filename_water_rem = path + 'water_removed_' + str(t) + '.npy'
    filename_time_block = path + 'particle_time_block_' + str(t) + '.npy'
    np.save(filename_pt_vec, traced_vectors )
    np.save(filename_pt_scal, traced_scalars )
    np.save(filename_pt_xi, traced_xi )
    np.save(filename_water_rem, traced_water )
    np.save(filename_time_block, time_block )
    print('particle data block saved at times = ', time_block)
    print('water removed:', traced_water)
    
def load_particle_data_from_blocks(path, grid_save_times,
                                   pt_dumps_per_grid_frame):
    """Load particle tracer data from hard disc for multiple times
    
    Loads several time blocks. One block at each 'grid_save_time',
    corresponding to 'ind_t'. Each block contains particle tracer data
    of several time steps, corresponding to 'dump_counter'
    
    Parameters
    ----------
    path: str
        Path to directory, where data is stored.
        Provide in format '/path/to/directory/'
    save_times: ndarray
        1D array of times, at which particle data should be loaded

    Returns
    -------
    vec_data: ndarray
        Particle vector data at several time indices and dump indices
        vec_data[ind_t,dump_counter,0] = 2D array with particle positions
        vec_data[ind_t,dump_counter,1] = 2D array with particle velocities
    scal_data: ndarray
        Particle scalar data at several time indices and dump indices
        scal_data[ind_t,dump_counter,0] = 1D array with particle water masses
        scal_data[ind_t,dump_counter,1] = 1D array with particle solute masses
    xi_data: ndarray
        Particle multiplicity data at several time indices and dump indices
        xi_data[ind_t,dump_counter] = 1D array with particle multiplicities
        
    """
    
    vec_data = []
    scal_data = []
    xi_data = []
    for t in grid_save_times:
        filename_pt_vec = path + 'particle_vector_data_' + str(int(t)) + '.npy'
        filename_pt_scal = path + 'particle_scalar_data_' + str(int(t)) +'.npy'
        filename_pt_xi = path + 'particle_xi_data_' + str(int(t)) + '.npy'
        vec = np.load(filename_pt_vec)
        scal = np.load(filename_pt_scal)
        xi = np.load(filename_pt_xi)
        if len(vec.shape) == 4:
            for n in range(pt_dumps_per_grid_frame):
                vec_data.append(vec[n])
                scal_data.append(scal[n])
                xi_data.append(xi[n])
        elif len(vec.shape) == 3:
            vec_data.append(vec)
            scal_data.append(scal)
            xi_data.append(xi)
        else: print('vec.shape is not as expected')
    
    vec_data = np.array(vec_data)
    scal_data = np.array(scal_data)
    xi_data = np.array(xi_data)
    return vec_data, scal_data, xi_data

def dump_particle_data_all(t, pos, vel, cells, m_w, m_s, xi, active_ids, path):
    """Dump data of all particles to hard disc in .npy format
    
    Parameters
    ----------
    t: float
        Time used in filenames
    pos: ndarray, dtype=float
        2D array, where
        pos[0] = 1D array of horizontal coordinates (m)
        pos[1] = 1D array of vertical coordinates (m)
        (pos[0,n], pos[1,n]) is the position of particle 'n'
    vel: ndarray, dtype=float
        2D array, where
        vel[0] = 1D array of horizontal velocity components (m/s)
        vel[1] = 1D array of vertical velocity components (m/s)
        (vel[0,n], vel[1,n]) is the velocity of particle 'n'
    cells: ndarray, dtype=int
        2D array, holding the particle cell indices, i.e.
        cells[0] = 1D array of horizontal indices
        cells[1] = 1D array of vertical indices
        (cells[0,n], cells[1,n]) gives the cell of particle 'n' 
    m_w: ndarray, dtype=float
        1D array holding the particle water masses (1E-18 kg)
        This array gets updated by the function.
    m_s: ndarray, dtype=float
        1D array holding the particle solute masses (1E-18 kg)
    xi: ndarray, dtype=float
        1D array holding the particle multiplicities
    active_ids: ndarray, dtype=bool
        1D mask-array. Each particle gets a flag 'True' or 'False', defining
        if it still resides in the simulation domain or has already hit the
        ground and is thereby removed from the simulation
    path: str
        Path to directory, where data is stored.
        Provide in format '/path/to/directory/'
    
    """
    
    filename_pt_vec = path + 'particle_vector_data_all_' + str(int(t)) + '.npy'
    filename_pt_cells = path + 'particle_cells_data_all_' + str(int(t)) +'.npy'
    filename_pt_scal = path + 'particle_scalar_data_all_' + str(int(t)) +'.npy'
    filename_pt_xi = path + 'particle_xi_data_all_' + str(int(t)) + '.npy'
    filename_pt_act_ids = path + \
        'particle_active_ids_data_all_' + str(int(t)) + '.npy'
    np.save(filename_pt_vec, (pos, vel) )
    np.save(filename_pt_cells, cells)
    np.save(filename_pt_scal, (m_w, m_s) )
    np.save(filename_pt_xi, xi )
    np.save(filename_pt_act_ids, active_ids )
    
    print('all particle data saved at t =', t)

def load_particle_data_all(path, save_times):
    """Load data of all particles from hard disc at several time steps
    
    Loads data for all times in 'save_times', corresponding to 'ind_t' below.
    
    Parameters
    ----------
    path: str
        Path to directory, where data is stored.
        Provide in format '/path/to/directory/'
    save_times: ndarray
        1D array of times, at which particle data should be loaded

    Returns
    -------
    vec_data: ndarray
        Particle vector data at several time indices
        vec_data[ind_t,0] = 2D array with particle positions
        vec_data[ind_t,1] = 2D array with particle velocities
    cells_data: ndarray
        Particle cell data at several time indices
        cells_data[ind_t] = 2D array with particle positions
    scal_data: ndarray
        Particle scalar data at several time indices
        scal_data[ind_t,0] = 1D array with particle water masses
        scal_data[ind_t,1] = 1D array with particle solute masses
    xi_data: ndarray
        Particle multiplicity data at several time indices
        xi_data[ind_t] = 1D array with particle multiplicities
    active_ids_data: ndarray, dtype=bool
        Active IDs mask data at several time indices
        active_ids_data[ind_t] = 1D mask-array with active IDs
        
    """
    
    vec_data = []
    cells_data = []
    scal_data = []
    xi_data = []
    active_ids_data = []
    for t in save_times:
        filename_pt_vec =\
            path + 'particle_vector_data_all_' + str(int(t)) + '.npy'
        filename_pt_cells =\
            path + 'particle_cells_data_all_' + str(int(t)) + '.npy'                    
        filename_pt_scal =\
            path + 'particle_scalar_data_all_' + str(int(t)) + '.npy'
        filename_pt_xi = path + 'particle_xi_data_all_' + str(int(t)) + '.npy'
        filename_pt_act_ids = path + \
            'particle_active_ids_data_all_' + str(int(t)) + '.npy'        
        vec = np.load(filename_pt_vec)
        cells = np.load(filename_pt_cells)
        scal = np.load(filename_pt_scal)
        xi = np.load(filename_pt_xi)
        act_ids = np.load(filename_pt_act_ids)
        
        vec_data.append(vec)
        cells_data.append(cells)
        scal_data.append(scal)
        xi_data.append(xi)
        active_ids_data.append(act_ids)
        
    vec_data = np.array(vec_data)
    scal_data = np.array(scal_data)
    xi_data = np.array(xi_data)
    cells_data = np.array(cells_data)
    active_ids_data = np.array(active_ids_data)
    return vec_data, cells_data, scal_data, xi_data, active_ids_data

#%% SAVE AND LOAD GRID DATA (ATMOSPHERIC VARIABLES)
### IN WORK 
def save_grid_basics_to_textfile(grid_, t_, filename):
    with open(filename, 'w') as f:
        f.write(f'\
{grid_.ranges[0][0]} {grid_.ranges[0][1]} \
{grid_.ranges[1][0]} {grid_.ranges[1][1]} \
{grid_.steps[0]} {grid_.steps[1]} {grid_.step_y} {t_}')
        
def save_grid_arrays_to_npy_file(grid, filename1, filename2):
    arr1 = np.array([grid.pressure, grid.temperature,
                     grid.mass_density_air_dry,
                     grid.mixing_ratio_water_vapor,
                     grid.mixing_ratio_water_liquid,
                     grid.saturation_pressure, grid.saturation,
                     grid.potential_temperature])
    arr2 = np.array([grid.velocity[0], grid.velocity[1], 
                     grid.mass_flux_air_dry[0], grid.mass_flux_air_dry[1]])
    np.save(filename1, arr1)
    np.save(filename2, arr2)

# save fields in this order:
# 0 = r_v
# 1 = r_l
# 2 = Theta    
# 3 = T
# 4 = p
# 5 = S
def save_grid_scalar_fields(t, grid_scalar_fields, path, start_time):
    filename = path + 'grid_scalar_fields_t_' + str(int(t)) + '.npy'
    np.save(filename,
            (grid_scalar_fields[4],
             grid_scalar_fields[5],
             grid_scalar_fields[2],
             grid_scalar_fields[0],
             grid_scalar_fields[1],
             grid_scalar_fields[6]) )
    print('grid fields saved at t =', t,
          ', sim time:', datetime.now()-start_time)

def load_grid_scalar_fields(path, save_times):
    fields = []
    for t_ in save_times:
        filename = path + 'grid_scalar_fields_t_' + str(int(t_)) + '.npy'
        fields_ = np.load(filename)
        fields.append(fields_)
    fields = np.array(fields)
    return fields
   
def save_grid_to_files(grid, t_, basics_file, arr_file1, arr_file2):
    save_grid_basics_to_textfile(grid, t_, basics_file)
    save_grid_arrays_to_npy_file(grid, arr_file1, arr_file2)
    
def load_grid_from_files(basics_file, arr_file1, arr_file2):
    basics = np.loadtxt(basics_file)
    scalars = np.load(arr_file1)
    vectors = np.load(arr_file2)
    grid = Grid( [ [ basics[0], basics[1] ], [ basics[2], basics[3] ] ],
                 [ basics[4], basics[5] ], basics[6] )
    
    grid.pressure = scalars[0]
    grid.temperature = scalars[1]
    grid.mass_density_air_dry = scalars[2]
    grid.mixing_ratio_water_vapor = scalars[3]
    grid.mixing_ratio_water_liquid = scalars[4]
    grid.saturation_pressure = scalars[5]
    grid.saturation = scalars[6]
    grid.potential_temperature = scalars[7]
    
    grid.update_material_properties()
    V0_inv = 1.0 / grid.volume_cell
    grid.rho_dry_inv =\
        np.ones_like(grid.mass_density_air_dry) / grid.mass_density_air_dry
    grid.mass_dry_inv = V0_inv * grid.rho_dry_inv
    
    grid.velocity = np.array( [vectors[0], vectors[1]] )
    grid.mass_flux_air_dry = np.array( [vectors[2], vectors[3]] )
    return grid

#%% GRID AND PARTICLES LOAD/SAVE
       
def save_grid_and_particles_full(t, grid, pos, cells, vel, m_w, m_s, xi,
                                 active_ids,
                                 path):
    grid.mixing_ratio_water_liquid.fill(0.0)
    
    for ID in np.arange(len(xi))[active_ids]:
        grid.mixing_ratio_water_liquid[cells[0,ID],cells[1,ID]] +=\
            m_w[ID] * xi[ID]
    grid.mixing_ratio_water_liquid *= 1.0E-18 * grid.mass_dry_inv

    grid_file_list = ['grid_basics_' + str(int(t)) + '.txt',
                      'arr_file1_' + str(int(t)) + '.npy',
                      'arr_file2_' + str(int(t)) + '.npy']
    grid_file_list = [path + s for s in grid_file_list  ]
    vector_filename = 'particle_vectors_' + str(int(t)) + '.npy'
    vector_filename = path + vector_filename
    cells_filename = 'particle_cells_' + str(int(t)) + '.npy'
    cells_filename = path + cells_filename
    scalar_filename = 'particle_scalars_' + str(int(t)) + '.npy'
    scalar_filename = path + scalar_filename    
    xi_filename = 'multiplicity_' + str(int(t)) + '.npy'
    xi_filename = path + xi_filename    
    active_ids_file = 'active_ids_' + str(int(t)) + '.npy'
    active_ids_file = path + active_ids_file
    save_grid_to_files(grid, t, *grid_file_list)
    
    save_particles_to_files(pos, cells, vel, m_w, m_s, xi,
                            active_ids, 
                            vector_filename, scalar_filename,
                            cells_filename, xi_filename,
                            active_ids_file)
    
def load_grid_and_particles_full(t, path):
    grid_file_list = ['grid_basics_' + str(int(t)) + '.txt',
                      'arr_file1_' + str(int(t)) + '.npy',
                      'arr_file2_' + str(int(t)) + '.npy']
    grid_file_list = [path + s for s in grid_file_list  ]
    vector_filename = 'particle_vectors_' + str(int(t)) + '.npy'
    vector_filename = path + vector_filename
    scalar_filename = 'particle_scalars_' + str(int(t)) + '.npy'
    scalar_filename = path + scalar_filename
    cells_filename = 'particle_cells_' + str(int(t)) + '.npy'
    cells_filename = path + cells_filename  
    xi_filename = 'multiplicity_' + str(int(t)) + '.npy'
    xi_filename = path + xi_filename    
    active_ids_file = 'active_ids_' + str(int(t)) + '.npy'
    active_ids_file = path + active_ids_file
    grid = load_grid_from_files(*grid_file_list)
    vectors = np.load(vector_filename)
    pos = vectors[0]
    vel = vectors[1]
    cells = np.load(cells_filename)
    scalars = np.load(scalar_filename)
    m_w = scalars[0]
    m_s = scalars[1]
    xi = np.load(xi_filename)
    active_ids = np.load(active_ids_file)
    return grid, pos, cells, vel, m_w, m_s, xi, active_ids 

#%% For parameter input file
def load_kernel_data_Ecol(kernel_method, save_dir_Ecol_grid, E_col_const=0.5):
    if kernel_method == 'Ecol_grid_R':
        radius_grid = \
            np.load(save_dir_Ecol_grid + 'radius_grid_out.npy')
        E_col_grid = \
            np.load(save_dir_Ecol_grid + 'E_col_grid.npy' )        
        R_kernel_low = radius_grid[0]
        bin_factor_R = radius_grid[1] / radius_grid[0]
        R_kernel_low_log = np.log(R_kernel_low)
        bin_factor_R_log = np.log(bin_factor_R)
        no_kernel_bins = len(radius_grid)
    elif kernel_method == 'Ecol_const':
        E_col_grid = E_col_const
        radius_grid = None
        R_kernel_low = None
        bin_factor_R = None
        R_kernel_low_log = None
        bin_factor_R_log = None
        no_kernel_bins = None    
    return E_col_grid, radius_grid, \
           R_kernel_low, bin_factor_R, \
           R_kernel_low_log, bin_factor_R_log, \
           no_kernel_bins
