#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 11:48:37 2019

@author: jdesk
"""

import numpy as np

# m = 4/3 pi rho R**3 = c R**3
cc = 4/3 * np.pi * 1E3
# R = m**1/3 / c**1/3 = a * m**1/3
ac = 1. / cc**(1/3)

m1 = np.logspace(-15,-6,50)
m2 = 1E-12

dm = 5E-13