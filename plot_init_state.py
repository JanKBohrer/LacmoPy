#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""


import numpy as np
import matplotlib.pyplot as plt


from file_handling import load_grid_and_particles_full


simdata_path = "/Users/bohrer/sim_data_cloudMP_ab_Jan20/"

solute_type = "AS"
spcm = [10,14]
no_cells = [50,50]
gseed = 1001


grid_path = simdata_path + solute_type + "/" \
            + f"grid_{no_cells[0]}_{no_cells[1]}_spcm_{spcm[0]}_{spcm[1]}/" \
            + f"{gseed}/"


grid, pos, cells, vel, m_w, m_s, xi, active_ids =\
    load_grid_and_particles_full(0, grid_path)

