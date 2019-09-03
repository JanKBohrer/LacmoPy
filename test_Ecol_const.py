#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 10:59:00 2019

@author: bohrer
"""

import numpy as np

my_OS = "Mac"

if len(sys.argv) > 1:
    my_OS = sys.argv[1]

if(my_OS == "Linux_desk"):
    home_path = '/home/jdesk/'
    simdata_path = "/mnt/D/sim_data_cloudMP/"
#    fig_path = home_path + 'Onedrive/Uni/Masterthesis/latex/Report/Figures/'
elif (my_OS == "Mac"):
    simdata_path = "/Users/bohrer/sim_data_cloudMP/"
elif (my_OS == "TROPOS_server"):
    simdata_path = "/vols/fs1/work/bohrer/sim_data_cloudMP/"  


#%% GRID PARAMETERS
no_cells = (10, 10)
#no_cells = (15, 15)
#no_cells = (75, 75)

# solute material: NaCl OR ammonium sulfate
#solute_type = "NaCl"
solute_type = "AS"

# no_super_particles_cell_mode = [N1,N2] is a list with
# N1 = no super part. per cell in mode 1 etc.
# with init method = SingleSIP, this is only the target value.
# the true number of particles per cell and mode will fluctuate around this
no_spcm = np.array([6, 8])
#no_spcm = np.array([12, 12])
#no_spcm = np.array([26, 38])
#no_spcm = np.array([16, 24])
#no_spcm = np.array([18, 26])
#no_spcm = np.array([52, 76])

# seed of the SIP generation -> needed for the right grid folder
# 3711, 3713, 3715, 3717
# 3719, 3721, 3723, 3725
seed_SIP_gen = 3711

if len(sys.argv) > 2:
    seed_SIP_gen = int(sys.argv[2])

# for collisons
# seed start with 4 for dt_col = dt_adv
#seed_sim = 4711

# seed start with 6 for dt_col = 0.5 dt_adv
seed_sim = 6711
if len(sys.argv) > 3:
    seed_sim = int(sys.argv[3])


grid_folder =\
    f"{solute_type}" \
    + f"/grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm[0]}_{no_spcm[1]}/" \
    + f"{seed_SIP_gen}/wo_spin_up_w_col/{seed_sim}/"



#%%

no_cols = np.load(simdata_path + grid_folder + "no_cols_20.npy") 