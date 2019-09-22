gseed=3711
sseed=6711
Ns=50
n1=13
n2=19
nc=75
type=AS

rsync -av --progress -e "ssh -i /home/jdesk/.ssh/gauss.pub" gauss5:/vols/fs1/work/bohrer/sim_data_cloudMP/${type}/grid_${nc}_${nc}_spcm_${n1}_${n2}/eval_data_avg_Ns_${Ns}_sg_${gseed}_ss_${sseed}/ /mnt/D/sim_data_cloudMP/${type}/grid_${nc}_${nc}_spcm_${n1}_${n2}/eval_data_avg_Ns_${Ns}_sg_${gseed}_ss_${sseed}/
