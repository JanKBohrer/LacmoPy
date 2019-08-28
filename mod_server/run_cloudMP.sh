#!/bin/bash
# execute generate grid and particles n times with different seeds

sseed1=3717
sseed2=6717
for x in {1..21}
do
	#echo $((3709 + 2*x))
	python3 cloudMP.py TROPOS_server $((sseed1 + 2*x)) $((sseed2 + 2*x)) with_collision 7200.0 14400.0 &
	echo $((sseed1 + 2*x)) $((sseed2 + 2*x))
	sleep 0.5
done

#python3 cloudMP.py TROPOS_server 3759 6759 with_collision 7200.0 14400.0 &
