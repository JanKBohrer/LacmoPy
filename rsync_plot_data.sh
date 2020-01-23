gseed=2101
sseed=2101
Ns=30
nc=75
n1=16
n2=24
type="AS"

#simdata_path=/mnt/D/sim_data_cloudMP/
source_path="bohrer@gauss5:/vols/fs1/work/bohrer/sim_data_cloudMP_ab_Jan20/"
destination_path="/Users/bohrer/sim_data_cloudMP_ab_Jan20/"
home_path="/Users/bohrer/"

#mkdir -p ${simdata_path}/${type}/grid_${nc}_${nc}_spcm_${n1}_${n2}/${gseed}/w_spin_up_w_col/${sseed}/

#scp -r gauss5:/vols/fs1/work/bohrer/sim_data_cloudMP_ab_Jan20/${type}/grid_${nc}_${nc}_spcm_${n1}_${n2}/eval_data_avg_Ns_${Ns}_sg_${gseed}_ss_${sseed} ${simdata_path}${type}/grid_${nc}_${nc}_spcm_${n1}_${n2}

mkdir -p ${destination_path}${type}/grid_${nc}_${nc}_spcm_${n1}_${n2}

rsync --progress -av -e "ssh -F /dev/null -i /Users/bohrer/.ssh/gauss" ${source_path}${type}/grid_${nc}_${nc}_spcm_${n1}_${n2}/eval_data_avg_Ns_${Ns}_sg_${gseed}_ss_${sseed} ${destination_path}${type}/grid_${nc}_${nc}_spcm_${n1}_${n2}

#scp -r gauss5:/vols/fs1/work/bohrer/sim_data_cloudMP_ab_Jan20/${type}/grid_${nc}_${nc}_spcm_${n1}_${n2}/eval_data_avg_Ns_${Ns}_sg_${gseed}_ss_${sseed} ${simdata_path}${type}/grid_${nc}_${nc}_spcm_${n1}_${n2}/

#rsync -av --progress -e "ssh -i /Users/bohrer/.ssh/gauss.pub" gauss5:/vols/fs1/work/bohrer/sim_data_cloudMP/${type}/grid_${nc}_${nc}_spcm_${n1}_${n2}/${gseed}/w_spin_up_w_col/${sseed}/ ${simdata_path}${type}/grid_${nc}_${nc}_spcm_${n1}_${n2}/${gseed}/w_spin_up_w_col/${sseed}/

#rsync -av --progress -e "ssh -i /Users/bohrer/.ssh/gauss.pub" gauss5:/vols/fs1/work/bohrer/sim_data_cloudMP/${type}/grid_${nc}_${nc}_spcm_${n1}_${n2}/${gseed}/w_spin_up_w_col/${sseed}/ ${simdata_path}${type}/grid_${nc}_${nc}_spcm_${n1}_${n2}/${gseed}/w_spin_up_w_col/${sseed}/
