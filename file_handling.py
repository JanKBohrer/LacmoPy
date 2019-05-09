#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 14:43:19 2019

@author: jdesk
"""

import pickle
import numpy as np

from grid import save_grid_to_files
from grid import load_grid_from_files
# from particle_class import save_particle_list_to_files
# from particle_class import load_particle_list_from_files
from datetime import datetime

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

#stores properties of particle_list_by_id and active_ids list
# particle_list_by_id: list of Particle objects
# active_ids: list of integers
def save_particles_to_files(pos, cells, vel, m_w, m_s, xi,
                            active_ids, removed_ids,
                            vector_filename, scalar_filename, cells_filename, 
                            xi_filename,
                            active_ids_filename, removed_ids_filename):
#     with open(pt_filename, "w") as f:
#         for p in p_list:
#             string1 = f'{p.id} {p.multiplicity} {p.location[0]} {p.location[1]} \
# {p.velocity[0]} {p.velocity[1]} {p.radius_solute_dry} {p.temperature} {p.equilibrium_temperature} {p.mass}\n'
#             f.write(string1)
    np.save(vector_filename, [pos, vel] )
    np.save(cells_filename, cells )
    np.save(scalar_filename, [m_w, m_s] )
    np.save(xi_filename, xi )
    np.save(active_ids_filename, active_ids)
    np.save(removed_ids_filename, removed_ids)

def dump_particle_data(t, pos, vel, m_w, m_s, xi, T_grid, rv_grid, path):
    filename_pt_vec = path + "particle_vector_data_" + str(int(t)) + ".npy"
    filename_pt_scal = path + "particle_scalar_data_" + str(int(t)) + ".npy"
    filename_grid = path + "grid_T_rv_" + str(int(t)) + ".npy"
    np.save(filename_pt_vec, (pos, vel) )
    np.save(filename_pt_scal, (m_w, m_s, xi) )
    np.save(filename_grid, (T_grid, rv_grid) )
    
        
def save_grid_and_particles_full(t, grid, pos, cells, vel, m_w, m_s, xi,
                                 active_ids, removed_ids,
                                 path):
    grid.mixing_ratio_water_liquid.fill(0.0)
    
    # print("active_ids")
    # print(active_ids)
    # print("cells")
    # print(cells)
    # print(cells[0,0])
    # print(cells[1,0])
    # print("m_w")
    # print(m_w)
    # print("xi")
    # print(xi)
    
    for ID in active_ids:
        # par = particle_list_by_id[ID]
        # cell = tuple(par.cell)
        grid.mixing_ratio_water_liquid[cells[0,ID],cells[1,ID]] +=\
            m_w[ID] * xi[ID]
    grid.mixing_ratio_water_liquid *= 1.0E-18 * grid.mass_dry_inv

    grid_file_list = ["grid_basics_" + str(int(t)) + ".txt",
                      "arr_file1_" + str(int(t)) + ".npy", "arr_file2_" + str(int(t)) + ".npy"]
    grid_file_list = [path + s for s in grid_file_list  ]
    vector_filename = "particle_vectors_" + str(int(t)) + ".npy"
    vector_filename = path + vector_filename
    cells_filename = "particle_cells_" + str(int(t)) + ".npy"
    cells_filename = path + cells_filename
    scalar_filename = "particle_scalars_" + str(int(t)) + ".npy"
    scalar_filename = path + scalar_filename    
    xi_filename = "multiplicity_" + str(int(t)) + ".npy"
    xi_filename = path + xi_filename    
    active_ids_file = "active_ids_" + str(int(t)) + ".npy"
    active_ids_file = path + active_ids_file
    rem_ids_file = "removed_ids_" + str(int(t)) + ".npy"
    rem_ids_file = path + rem_ids_file
    
#    np.save(path + "trace_ids_" + str(int(t)) + ".npy", trace_ids)
    save_grid_to_files(grid, t, *grid_file_list)
    
    save_particles_to_files(pos, cells, vel, m_w, m_s, xi,
                            active_ids, removed_ids,
                            vector_filename, scalar_filename,
                            cells_filename, xi_filename,
                            active_ids_file, rem_ids_file)
    
def load_grid_and_particles_full(t, path):
    grid_file_list = ["grid_basics_" + str(int(t)) + ".txt",
                      "arr_file1_" + str(int(t)) + ".npy", "arr_file2_" + str(int(t)) + ".npy"]
    grid_file_list = [path + s for s in grid_file_list  ]
    vector_filename = "particle_vectors_" + str(int(t)) + ".npy"
    vector_filename = path + vector_filename
    scalar_filename = "particle_scalars_" + str(int(t)) + ".npy"
    scalar_filename = path + scalar_filename
    cells_filename = "particle_cells_" + str(int(t)) + ".npy"
    cells_filename = path + cells_filename  
    xi_filename = "multiplicity_" + str(int(t)) + ".npy"
    xi_filename = path + xi_filename    
    active_ids_file = "active_ids_" + str(int(t)) + ".npy"
    active_ids_file = path + active_ids_file
    rem_ids_file = "removed_ids_" + str(int(t)) + ".npy"
    rem_ids_file = path + rem_ids_file
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
    removed_ids = np.load(rem_ids_file)
    # pt_lst, act_ids, rem_ids = load_particle_list_from_files(grid, particle_file, active_ids_file, rem_ids_file)
    return grid, pos, cells, vel, m_w, m_s, xi, active_ids, removed_ids




