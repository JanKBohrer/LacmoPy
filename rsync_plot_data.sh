gseed=6001
sseed=6001
Ns=50
nc=75
n1=26
n2=38
type="AS"

source_dir="bohrer@gauss5:/vols/fs1/work/bohrer/sim_data_cloudMP/"
destination_dir="/Users/bohrer/sim_data_cloudMP/"
home_dir="/Users/bohrer/"

mkdir -p ${destination_dir}${type}/grid_${nc}_${nc}_spcm_${n1}_${n2}

# -F /dev/null is necessary. without it, the config file is read and
# the login-commands are tried to be executed, which is not possible while using rsync
rsync --progress -av -e "ssh -F /dev/null -i /Users/bohrer/.ssh/gauss" ${source_dir}${type}/grid_${nc}_${nc}_spcm_${n1}_${n2}/eval_data_avg_Ns_${Ns}_sg_${gseed}_ss_${sseed} ${destination_dir}${type}/grid_${nc}_${nc}_spcm_${n1}_${n2}

#scp -r gauss5:/vols/fs1/work/bohrer/sim_data_cloudMP_ab_Jan20/${type}/grid_${nc}_${nc}_spcm_${n1}_${n2}/eval_data_avg_Ns_${Ns}_sg_${gseed}_ss_${sseed} ${simdata_path}${type}/grid_${nc}_${nc}_spcm_${n1}_${n2}/
