gseed=1001
sseed=1001
Ns=10
n1=6
n2=10
nc=15
type="AS"

#simdata_path=/mnt/D/sim_data_cloudMP/
simdata_path="/Users/bohrer/sim_data_cloudMP_TEST200108/"

home_path="/Users/bohrer/"

#mkdir -p ${simdata_path}/${type}/grid_${nc}_${nc}_spcm_${n1}_${n2}/${gseed}/w_spin_up_w_col/${sseed}/

#scp -r gauss5:/vols/fs1/work/bohrer/sim_data_cloudMP_ab_Jan20/${type}/grid_${nc}_${nc}_spcm_${n1}_${n2}/eval_data_avg_Ns_${Ns}_sg_${gseed}_ss_${sseed} ${simdata_path}${type}/grid_${nc}_${nc}_spcm_${n1}_${n2}

mkdir -p ${simdata_path}${type}/grid_${nc}_${nc}_spcm_${n1}_${n2}/eval_data_avg_Ns_${Ns}_sg_${gseed}_ss_${sseed}

scp -r gauss5:/vols/fs1/work/bohrer/sim_data_cloudMP_ab_Jan20/${type}/grid_${nc}_${nc}_spcm_${n1}_${n2}/eval_data_avg_Ns_${Ns}_sg_${gseed}_ss_${sseed} ${simdata_path}${type}/grid_${nc}_${nc}_spcm_${n1}_${n2}/

#rsync -av --progress -e "ssh -i /Users/bohrer/.ssh/gauss.pub" gauss5:/vols/fs1/work/bohrer/sim_data_cloudMP/${type}/grid_${nc}_${nc}_spcm_${n1}_${n2}/${gseed}/w_spin_up_w_col/${sseed}/ ${simdata_path}${type}/grid_${nc}_${nc}_spcm_${n1}_${n2}/${gseed}/w_spin_up_w_col/${sseed}/

#rsync -av --progress -e "ssh -i /Users/bohrer/.ssh/gauss.pub" gauss5:/vols/fs1/work/bohrer/sim_data_cloudMP/${type}/grid_${nc}_${nc}_spcm_${n1}_${n2}/${gseed}/w_spin_up_w_col/${sseed}/ ${simdata_path}${type}/grid_${nc}_${nc}_spcm_${n1}_${n2}/${gseed}/w_spin_up_w_col/${sseed}/
