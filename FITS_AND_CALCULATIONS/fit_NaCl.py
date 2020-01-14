#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 12:17:55 2019

@author: jdesk
"""

import numpy as np
import matplotlib.pyplot as plt

import constants as c

import microphysics as mp

from scipy.optimize import curve_fit

from numba import vectorize, njit

#%%

data = np.loadtxt("NaCl_water_act.txt")

wt_percent = data[:,(0,2,4,6)].T.flatten()
water_act = data[:,(1,3,5,7)].T.flatten()


np.save("wt_percent_NaCl", wt_percent)
np.save("water_act_NaCl_low", water_act)
