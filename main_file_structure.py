#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 15:57:02 2019

@author: jdesk
"""

#%% PARAMETERS
# -> PUT TO PARAMETER FILE LATER
# by par_file_name = sys.argv[1]
# if len(sys.argv[1] > 0):
# read_par_file(par_file_name) ...

#"act" for "activate"
act_spin_up = True
spin_up_time = 7200
act_collisions = True

#AON for Unterstrasser
#AON_Shima for Shima
#wating time for Gillespie
col_method = "AON"
#col_method = "wating_time"

# string or int ?
kernel_name = "Long_Bott"

act_gen_SIP = True

# and dt, dx, dy, dz, ....


# DERIVED PARAMETERS