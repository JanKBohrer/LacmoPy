#!/bin/bash
gseed=(3711 3811 3811 3411 3311 3811 4811 4811 3711 3711 3811)
sseed=(6711 6811 6811 6411 6311 7811 6811 7811 6711 6711 8811)
Ns=50
#gseed=3711
#sseed=6711
#Ns=50
n1=(13 26 52 26 26 26 26 26 26 26 26)
n2=(19 38 76 38 38 38 38 38 38 38 38)
nc=(75 75 75 75 75 75 75 75 75 150 75)
type=("AS" "AS" "AS" "AS" "AS" "AS" "AS" "AS" "NaCl" "AS" "AS")

#for i in {0..8}
for i in 9 10
do	 
    rsync -av --progress -e "ssh -i /home/jdesk/.ssh/gauss.pub" gauss5:/vols/fs1/work/bohrer/sim_data_cloudMP/${type[$i]}/grid_${nc[$i]}_${nc[$i]}_spcm_${n1[$i]}_${n2[$i]}/eval_data_avg_Ns_${Ns}_sg_${gseed[$i]}_ss_${sseed[$i]}/ /mnt/D/sim_data_cloudMP/${type[$i]}/grid_${nc[$i]}_${nc[$i]}_spcm_${n1[$i]}_${n2[$i]}/eval_data_avg_Ns_${Ns}_sg_${gseed[$i]}_ss_${sseed[$i]}/
done
