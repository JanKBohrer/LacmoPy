#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 13:32:24 2019

@author: bohrer
"""

from mpi4py import MPI

mpicomm = MPI.COMM_WORLD

nproc = mpicomm.Get_size()
rank = mpicomm.Get_rank()



if rank == 0:
    print("my rank =", rank, "nproc =", nproc )
elif rank == 1:    
    print("my rank is not zero but =", rank,
          "nproc =", nproc)