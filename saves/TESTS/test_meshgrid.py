#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 18:00:45 2019

@author: jdesk
"""
import numpy as np

i_list = [1,2,3]
j_list = [4,5,6,7]
#i_list = [20,40,20,40,20,40]
#j_list = [20,20,40,40,60,60]

M = np.meshgrid(i_list,j_list, indexing = "xy")

print(M)