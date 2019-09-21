gseed=4811
sseed=6811
Ns=50
n1=26
n2=38
nc=75
type=AS

mkdir -p /mnt/D/sim_data_cloudMP/${type}/grid_${nc}_${nc}_spcm_${n1}_${n2}/${gseed}/w_spin_up_w_col/${sseed}/

rsync -av -e "ssh -i /home/jdesk/.ssh/gauss.pub" gauss5:/vols/fs1/work/bohrer/sim_data_cloudMP/${type}/grid_${nc}_${nc}_spcm_${n1}_${n2}/${gseed}/w_spin_up_w_col/${sseed}/ /mnt/D/sim_data_cloudMP/${type}/grid_${nc}_${nc}_spcm_${n1}_${n2}/${gseed}/w_spin_up_w_col/${sseed}/
