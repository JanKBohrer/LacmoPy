#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 11:33:54 2019

@author: bohrer
"""

import numpy as np

import matplotlib.pyplot as plt

path = "/Users/bohrer/CloudMP/data_Wang_bin_Unt/"
path2 = "/Users/bohrer/CloudMP/collision/ref_data/"

#%% MOMENTS VERSUS TIME ("Unt calls themprofiles" ?)
### HALL
data_mom_Hall = np.loadtxt(path + "Hall_Profiles.txt").T[:,::20]
data_mom_Hall_my = np.reshape( np.loadtxt(path2 + "Wang_2007_Hall.txt"), (4,7))

### LONG

data_mom_Long = np.loadtxt(path + "Long_Profiles.txt").T[:,::20]
data_Wang_2007 = [
                    295.4 ,
                    287.4 ,
                    278.4 ,
                    264.4 ,
                    151.7 ,
                    13.41 ,
                    1.212,
                    0.999989, 
                    0.999989 ,
                    0.999989 ,
                    0.999989 ,
                    0.999989 ,
                    0.999989 ,
                    0.999989 ,
                    6.739E-9 ,
                    7.402E-9 ,
                    8.720E-9 ,
                    3.132E-7 ,
                    3.498E-4 ,
                    1.068E-2 ,
                    3.199E-2 ,
                    6.813e-14 ,
                    9.305e-14 ,
                    5.710e-13 ,
                    3.967e-8 ,
                    1.048e-3 ,
                    2.542e-1 ,
                    1.731    
                    ]

data_mom_Long_my = np.reshape(data_Wang_2007, (4,7))

###

scales_my = np.array((1E6,1E-3,1E-6,1E-12))

dt_mom = 200 # s
dt_mom_my = 600 # s

time_mom = np.arange(0, 3600 + 1, dt_mom)
time_mom_my = np.arange(0, 3600 + 1, dt_mom_my)

no_dpts = len(time_mom)

#%% data OK -> save to txt file
#np.savetxt(path2 + "Long_Bott/" + "Wang_2007_moments.txt",
#           data_mom_Long[:,:no_dpts])
#np.savetxt(path2 + "Hall_Bott/" + "Wang_2007_moments.txt",
#           data_mom_Hall[:,:no_dpts])
#np.savetxt(path2 + "Long_Bott/" + "Wang_2007_times.txt",
#           time_mom)
#np.savetxt(path2 + "Hall_Bott/" + "Wang_2007_times.txt",
#           time_mom)


#DH = np.loadtxt(path2 + "Hall_Bott/" + "Wang_2007_moments.txt")
#DL = np.loadtxt(path2 + "Long_Bott/" + "Wang_2007_moments.txt")
#
#tH = np.loadtxt(path2 + "Hall_Bott/" + "Wang_2007_times.txt")
#tL = np.loadtxt(path2 + "Long_Bott/" + "Wang_2007_times.txt")


#%% plotting

#fig, axes = plt.subplots(nrows=4,ncols=2,figsize=(12,20))
#
#for mom_n in range(4):
#    ax = axes[mom_n,0]
#    ax.plot(time_mom/60, data_mom_Hall[mom_n][:no_dpts], "x-")
#    ax.plot(tH/60, DH[mom_n], "o-")
#    ax.plot(time_mom_my/60, data_mom_Hall_my[mom_n][:]*scales_my[mom_n], "d-")
#    if mom_n != 1:
#        ax.set_yscale("log")
#
#for mom_n in range(4):
#    ax = axes[mom_n,1]
#    ax.plot(time_mom/60, data_mom_Long[mom_n][:no_dpts], "x-")
#    ax.plot(tL/60, DL[mom_n], "o-")
#    ax.plot(time_mom_my/60, data_mom_Long_my[mom_n][:]*scales_my[mom_n], "d-")
#    if mom_n != 1:
#        ax.set_yscale("log")

#%% Spectra: g_ln_R vs R

#times_str = ( "00", "10", "20", "30", "40" )

# in minutes

def load_and_save_data_bin_model(load_path, save_path, kernel_name ):

    times = np.linspace(0,60,7)
    times_str = []
    for t_ in times:
        times_str.append(f"{t_:02.0f}")
    #    print(f"{t_:02.0f}")
    
    
    no_bins_max = 572
    
    no_times = 7
    
    g_ln_R_vs_time = np.zeros((no_times,no_bins_max), dtype=np.float64)
    bin_centers_vs_time = np.zeros((no_times,no_bins_max), dtype=np.float64)
    no_bins_vs_time = np.zeros(7, dtype=np.int32)
    
    for time_n,t_ in enumerate(times_str):
        filename = path + f"GQ_{t_}_{kernel_name}_s16.dat"
        data = np.loadtxt(filename)
        R = data[:,0]
        g_ln_R = data[:,4]
        no_bins = data.shape[0]
        
        no_bins_vs_time[time_n] = no_bins
        bin_centers_vs_time[time_n,:no_bins] = R*1E3
        g_ln_R_vs_time[time_n,:no_bins] = g_ln_R
        
        print(data.shape)
    
    
       
    np.savetxt(save_path + f"{kernel_name}_Bott/" + "Wang_2007_radius_bin_centers.txt",
               bin_centers_vs_time)
    np.savetxt(save_path + f"{kernel_name}_Bott/" + "Wang_2007_g_ln_R.txt",
               g_ln_R_vs_time)
    np.savetxt(save_path + f"{kernel_name}_Bott/" + "Wang_2007_no_bins_vs_time.txt",
               no_bins_vs_time)
    np.savetxt(save_path + f"{kernel_name}_Bott/" + "Wang_2007_times_spectra.txt",
               times*60)
    
    return bin_centers_vs_time, g_ln_R_vs_time, no_bins_vs_time, times

    
    
#%% plotting spectra

kernel_name = "Long"
bin_centers_vs_time, g_ln_R_vs_time, no_bins_vs_time, times = \
    load_and_save_data_bin_model(path, path2, kernel_name)

fig, axes = plt.subplots(figsize=(12,8))    

ax = axes
for time_n, t_ in enumerate(times):
    no_bins = no_bins_vs_time[time_n]
    ax.loglog(bin_centers_vs_time[time_n,:no_bins],
              g_ln_R_vs_time[time_n, :no_bins], linestyle="dotted" )
    
    ax.set_xlim((1,5E3))
    ax.set_ylim((1E-4,1E1))

kernel_name = "Hall"
bin_centers_vs_time, g_ln_R_vs_time, no_bins_vs_time, times = \
    load_and_save_data_bin_model(path, path2, kernel_name)

fig, axes = plt.subplots(figsize=(12,8))    

ax = axes
for time_n, t_ in enumerate(times):
    no_bins = no_bins_vs_time[time_n]
    ax.loglog(bin_centers_vs_time[time_n,:no_bins],
              g_ln_R_vs_time[time_n, :no_bins], linestyle="dotted" )
    
    ax.set_xlim((1,5E3))
    ax.set_ylim((1E-4,1E1))
    
    
    
    
    
    




