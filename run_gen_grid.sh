#!/bin/bash
# execute generate grid and particles n times with different seeds

sseed=9719
#system="TROPOS_server"
system="Mac"

for x in {1..4}
do
    export OMP_NUM_THREADS=8
    #export MKL_NUM_THREADS=4
    export NUMBA_NUM_THREADS=16
    #echo $((3709 + 2*x))
    #python3 generate_grid_and_particles.py TROPOS_server $((sseed + 2*x)) &
    python3 generate_grid_and_particles.py $system $((sseed + 2*x)) &
done
