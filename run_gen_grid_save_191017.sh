#!/bin/bash
# execute generate grid and particles n times with different seeds

sseed=3709
system="TROPOS_server"
#system="Mac"
solute_type="AS"
no_spcm0=26
no_spcm1=38
dx=20
dz=20

for x in {1..50}
do
    export OMP_NUM_THREADS=8
    #export MKL_NUM_THREADS=4
    export NUMBA_NUM_THREADS=16
    echo $((sseed + 2*x))
    #python3 generate_grid_and_particles.py TROPOS_server $((sseed + 2*x)) &
    python3 generate_grid_and_particles.py $system $((sseed + 2*x)) $solute_type $no_spcm0 $no_spcm1 $dx $dz &
done
