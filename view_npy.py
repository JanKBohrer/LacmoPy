#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 16:28:58 2020

@author: bohrer
"""



import numpy as np

data_dir = '/Users/bohrer/CloudMP/collision/kernel_data/Hall/'

Ecol_Hall_raw_corr = np.load(data_dir + 'Hall_Bott_E_col_table_raw_corr.npy')
Ecol_Hall_raw = np.load(data_dir + 'Hall_Bott_E_col_table_raw.npy')
Rcol_Hall_raw = np.load(data_dir + 'Hall_Bott_R_col_table_raw.npy')


diff = Ecol_Hall_raw_corr - Ecol_Hall_raw