#!/bin/bash
# adjust the number of OMP threads for numpy
# and threads available to numba per process
export OMP_NUM_THREADS=8
#export MKL_NUM_THREADS=4
export NUMBA_NUM_THREADS=16
python3 run_box_model.py &
