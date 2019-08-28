#!/bin/bash
# execute generate grid and particles n times with different seeds

sseed=9709
#system = "TROPOS_server"
system="Mac"

for x in {1..4}
do
	#echo $((3709 + 2*x))
    #python3 generate_grid_and_particles.py TROPOS_server $((sseed + 2*x)) &
    python3 generate_grid_and_particles.py $system $((sseed + 2*x)) &
done
