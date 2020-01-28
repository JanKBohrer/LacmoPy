gseed=1001
sseed=1001
Ns=50
nc=75
n1=26
n2=38
type="AS"

#simdata_path=/mnt/D/sim_data_cloudMP/
source_dir="bohrer@gauss5:/vols/fs1/work/bohrer/sim_data_cloudMP/"
destination_dir="/Users/bohrer/sim_data_cloudMP/"
home_dir="/Users/bohrer/"

#mkdir -p ${simdata_path}/${type}/grid_${nc}_${nc}_spcm_${n1}_${n2}/${gseed}/w_spin_up_w_col/${sseed}/

#scp -r gauss5:/vols/fs1/work/bohrer/sim_data_cloudMP_ab_Jan20/${type}/grid_${nc}_${nc}_spcm_${n1}_${n2}/eval_data_avg_Ns_${Ns}_sg_${gseed}_ss_${sseed} ${simdata_path}${type}/grid_${nc}_${nc}_spcm_${n1}_${n2}

mkdir -p ${destination_dir}${type}/grid_${nc}_${nc}_spcm_${n1}_${n2}

rsync --progress -av -e "ssh -F /dev/null -i /Users/bohrer/.ssh/gauss" ${source_dir}${type}/grid_${nc}_${nc}_spcm_${n1}_${n2}/eval_data_avg_Ns_${Ns}_sg_${gseed}_ss_${sseed} ${destination_dir}${type}/grid_${nc}_${nc}_spcm_${n1}_${n2}

#scp -r gauss5:/vols/fs1/work/bohrer/sim_data_cloudMP_ab_Jan20/${type}/grid_${nc}_${nc}_spcm_${n1}_${n2}/eval_data_avg_Ns_${Ns}_sg_${gseed}_ss_${sseed} ${simdata_path}${type}/grid_${nc}_${nc}_spcm_${n1}_${n2}/

#rsync -av --progress -e "ssh -i /Users/bohrer/.ssh/gauss.pub" gauss5:/vols/fs1/work/bohrer/sim_data_cloudMP/${type}/grid_${nc}_${nc}_spcm_${n1}_${n2}/${gseed}/w_spin_up_w_col/${sseed}/ ${simdata_path}${type}/grid_${nc}_${nc}_spcm_${n1}_${n2}/${gseed}/w_spin_up_w_col/${sseed}/

#rsync -av --progress -e "ssh -i /Users/bohrer/.ssh/gauss.pub" gauss5:/vols/fs1/work/bohrer/sim_data_cloudMP/${type}/grid_${nc}_${nc}_spcm_${n1}_${n2}/${gseed}/w_spin_up_w_col/${sseed}/ ${simdata_path}${type}/grid_${nc}_${nc}_spcm_${n1}_${n2}/${gseed}/w_spin_up_w_col/${sseed}/
