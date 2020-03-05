#!/bin/bash
for x in 5 10 20 40 60 100 200 400 600 800 1000 1500 2000 3000
do
    mkdir kappa_$x/dt_10/
    scp gauss5:/vols/fs1/work/bohrer/sim_data_col_box_mod/expo/SinSIP/eta_1e-09_fix/results/Long_Bott/Ecol_grid_R/kappa_$x/dt_10/{save_times_3711
bins_mass_centers_500_no_bins_50
bins_rad_centers_500_no_bins_50
f_m_num_avg_vs_time_no_sims_500_no_bins_50
g_m_num_avg_vs_time_no_sims_500_no_bins_50
g_ln_r_num_avg_vs_time_no_sims_500_no_bins_50
moments_vs_time_avg_no_sims_500_no_bins_50
moments_vs_time_std_no_sims_500_no_bins_50} kappa_$x/dt_10/
done
