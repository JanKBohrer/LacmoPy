#!/bin/bash
# execute generate grid and particles n times with different seeds

first_seed_gen=1401
no_sims=7

for ((n=0; n<$no_sims; n++))
do
    export OMP_NUM_THREADS=8
    export NUMBA_NUM_THREADS=16
    #export MKL_NUM_THREADS=4
    python3 generate_grid_and_particles_mod.py $((first_seed_gen + 2*n)) &
    echo $((first_seed_gen + 2*n))
    sleep 0.05
done
