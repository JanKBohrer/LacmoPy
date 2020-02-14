#!/bin/bash
# execute generate_grid_and_particles.py no_sim times with different seeds
# the seed list will start with the "first" value declared here
# and increment by 2: [seed0, seed0 + 2, seed0 + 4, ...]
first_seed_gen=1001
no_sims=10

for ((n=0; n<$no_sims; n++))
do
    export OMP_NUM_THREADS=8
    export NUMBA_NUM_THREADS=16
    #export MKL_NUM_THREADS=4
    python3 generate_grid_and_particles.py $((first_seed_gen + 2*n)) &
    echo $((first_seed_gen + 2*n))
    sleep 0.05
done
