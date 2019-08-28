#!/bin/bash
# execute generate grid and particles n times with different seeds

for x in {1..4}
do
	#echo $((3709 + 2*x))
	python3 generate_grid_and_particles.py Mac $((3715 + 2*x)) &
done
