#!/bin/bash
# execute generate grid and particles n times with different seeds

sseed1=3717
sseed2=6717
system = "TROPOS_server"
#system = "Mac"
t_start = 0.0
t_end = 7200.0
sim_type = "spin_up"
#sim_type = "with_collision"


for x in {1..21}
do
	#echo $((3709 + 2*x))
	python3 cloudMP.py $system $((sseed1 + 2*x)) $((sseed2 + 2*x)) $sim_type $t_start $t_end &
	echo $((sseed1 + 2*x)) $((sseed2 + 2*x))
	sleep 0.1
done

#python3 cloudMP.py TROPOS_server 3759 6759 with_collision 7200.0 14400.0 &
