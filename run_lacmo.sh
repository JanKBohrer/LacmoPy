#!/bin/bash
# execute lacmo.py "no_sims" times with different seeds
# the seed lists will start with the "first" values declared here
# and increment by 2: [seed0, seed0 + 2, seed0 + 4, ...]
first_seed_gen=1001
first_seed_sim=1001
no_sims=10

for ((n=0; n<$no_sims; n++))
do
    export OMP_NUM_THREADS=8
    export NUMBA_NUM_THREADS=16
    #export MKL_NUM_THREADS=4
    python3 lacmo.py $((first_seed_gen + 2*n)) $((first_seed_sim + 2*n)) &
    echo $((first_seed_gen + 2*n)) $((first_seed_sim + 2*n))
    sleep 0.05
done