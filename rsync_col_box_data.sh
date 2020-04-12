# sync data from collision box model simulation
# requires analysis, i.e. args_sim[1], to be finished in source folders
# copies only files of analyzed data such as f_m_num_avg_vs_time_....npy
# and one of the save_times_seed1

dist="expo"
init="SinSIP"
eta="1e-09"
thresh="fix"
kernel="Golovin"
sseed=1001

source_dir="bohrer@gauss5:/vols/fs1/work/bohrer/sim_data_col_box_model2/${dist}/${init}/eta_${eta}_${thresh}/results/"
destination_dir="/home/jdesk/sim_data_col_box_model2/${dist}/${init}/eta_${eta}_${thresh}/results/"
include_f="*/*/*/*save_times_${sseed}*"
#destination_dir="/Users/bohrer/sim_data_cloudMP/"
#home_dir="/Users/bohrer/"

echo ${source_dir}
echo ${destination_dir}

mkdir -p ${destination_dir}/${kernel}

# -F /dev/null is necessary. without it, the config file is read and
# the login-commands are tried to be executed, which is not possible while using rsync
rsync --progress -av --include=${include_f} --exclude='*masses*' --exclude='*xis*' --exclude='*save_times*' -e "ssh -F /dev/null -i /home/jdesk/.ssh/gauss" ${source_dir}/${kernel} ${destination_dir}

#scp -r gauss5:/vols/fs1/work/bohrer/sim_data_cloudMP_ab_Jan20/${type}/grid_${nc}_${nc}_spcm_${n1}_${n2}/eval_data_avg_Ns_${Ns}_sg_${gseed}_ss_${sseed} ${simdata_path}${type}/grid_${nc}_${nc}_spcm_${n1}_${n2}/
