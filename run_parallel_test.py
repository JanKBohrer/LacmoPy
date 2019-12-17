#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 16:20:03 2019

@author: bohrer
"""

import os

#from IPython import get_ipython
#ip = get_ipython()
#ip.run_cell("!mpirun -np 2 python3 mpi_test.py")

RunMpiProgram = "mpirun -np 2 python3 mpi_test.py"

os.system(RunMpiProgram)