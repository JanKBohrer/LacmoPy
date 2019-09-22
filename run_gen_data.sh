#!/bin/bash
gseed=3811
sseed=6811
Ns=50
system="TROPOS_server"
#system="Mac"
#system="Linux_desk"

#t_start=0.0
t_start=7200.0
#t_end=7200.0
t_end=14400.0

no_cells_x=75
no_cells_z=75
solute_type="AS"
no_spcm0=26
no_spcm1=38
no_col_per_adv=2
sim_type="with_collision"

export OMP_NUM_THREADS=8
#export MKL_NUM_THREADS=4
export NUMBA_NUM_THREADS=16
python3 gen_plot_data.py $system $no_cells_x $no_cells_z $solute_type $no_spcm0 $no_spcm1 $Ns ${gseed} ${sseed} $sim_type $t_start $t_end $no_col_per_adv
