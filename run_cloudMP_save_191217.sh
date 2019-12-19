#!/bin/bash
# execute generate grid and particles n times with different seeds
sseed1=3809
sseed2=8809
system="TROPOS_server"
#system="Mac"

#t_start=0.0
t_start=7200.0
#t_end=7200.0
t_end=14400.0

no_cells_x=75
no_cells_z=75
solute_type="AS"
no_spcm0=26
no_spcm1=38
no_col_per_adv=10

kernel_type="Long_Bott"
kernel_method="Ecol_grid_R"
#sim_type="spin_up"
sim_type="with_collision"

for x in {1..50}
do
    #echo $((3709 + 2*x))
    export OMP_NUM_THREADS=8
    #export MKL_NUM_THREADS=4
    export NUMBA_NUM_THREADS=16
    python3 cloudMP.py $system $no_cells_x $no_cells_z $solute_type $no_spcm0 $no_spcm1 $((sseed1 + 2*x)) $((sseed2 + 2*x)) $sim_type $t_start $t_end $no_col_per_adv $kernel_type $kernel_method &
    echo $((sseed1 + 2*x)) $((sseed2 + 2*x))
    sleep 0.05
done
#python3 cloudMP.py TROPOS_server 3759 6759 with_collision 7200.0 14400.0 &
