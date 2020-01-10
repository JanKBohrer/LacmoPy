#!/bin/bash
gseed=(3711 3811 3811 3411 3311 3811 4811 4811 3711 3711 3811)
sseed=(6711 6811 6811 6411 6311 7811 6811 7811 6711 6711 8811)
Ns=50
system="TROPOS_server"
#system="Mac"
#system="Linux_desk"

#t_start=0.0
t_start=7200.0
#t_end=7200.0
t_end=14400.0

no_cells=(75 75 75 75 75 75 75 75 75 150 75)
#no_cells_z=75
solute_type=("AS" "AS" "AS" "AS" "AS" "AS" "AS" "AS" "NaCl" "AS" "AS")
no_spcm0=(13 26 52 26 26 26 26 26 26 26 26)
no_spcm1=(19 38 76 38 38 38 38 38 38 38 38)
no_col_per_adv=(2 2 2 2 2 2 2 2 2 2 10)
sim_type="with_collision"

#array1=(a b cc)
#array2=(1 2 10)

#for i in {0..8}
for i in 9 10
do
    export OMP_NUM_THREADS=8
    #export MKL_NUM_THREADS=4
    export NUMBA_NUM_THREADS=16
    python3 gen_plot_data.py $system ${no_cells[$i]} ${no_cells[$i]} ${solute_type[$i]} ${no_spcm0[$i]} ${no_spcm1[$i]} $Ns ${gseed[$i]} ${sseed[$i]} $sim_type $t_start $t_end ${no_col_per_adv[$i]} >> log_gen_data.log &
done
