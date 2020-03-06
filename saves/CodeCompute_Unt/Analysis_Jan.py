#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 15:23:00 2019

@author: jdesk
"""

#%% IMPORTS AND DEFINITIONS
import os
import math
import numpy as np
from numba import njit

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


#moments_file = sim_data_path + 

# modified moment_1
def defineMoments_0D():
    import numpy as np
    #verwende tabellierte Werte aus Wang-Paper
    nt_ref=7
    WangMomente=np.zeros([nt_ref,4])
    timeWang=np.arange(nt_ref)*10 # in Minuten
    WangMomente[:,0]=np.array([295.4e6     , 287.4e6     , 278.4e6     ,  264.4e6       , 151.7e6, 13.41e6, 1.212e6])  # ; Anzahl in m^-3
    WangMomente[:,1]=1.0E-3*np.array([0.999989e-3 , 0.999989e-3 , 0.999989e-3 ,  0.999989e-3   ,0.999989e-3, 0.999989e-3, 0.999989e-3])/0.999989e-3 #; Masse
    WangMomente[:,2]=np.array([6.739e-15   , 7.402e-15   , 8.72e-15,      3.132e-13      , 3.498e-10, 1.068e-8,3.199e-8])# ; Moment 2 in kg^2/m^3
    WangMomente[:,3]=np.array([6.813e-26   , 9.305e-26   ,5.71e-25,       3.967e-20       , 1.048e-15, 2.542e-13, 1.731e-12] )#; ;Moment 3 in kg^3/m^3
    return timeWang, WangMomente

#%%

t_Wang, mom_Wang = defineMoments_0D()

kappa_list = [5,10,20,40,60,100,200,400]

sim_data_path = "/home/jdesk/CloudMP/CodeCompute_Unt/kappa_10/"
t_mom = np.loadtxt(sim_data_path + "Moments_meta.dat" ,skiprows=2)

moments_vs_time_kappa_var = []

for kappa_n,kappa in enumerate(kappa_list):

    sim_data_path = f"/home/jdesk/CloudMP/CodeCompute_Unt/kappa_{kappa}/"
    
    if kappa == 40:
        t_mom = np.loadtxt(sim_data_path + "Moments_meta.dat" ,skiprows=2)
    elif kappa == 60:
        t_mom = np.loadtxt(sim_data_path + "Moments_meta.dat" ,skiprows=2)
    
#    if kappa == 400:
#        moments_vs_time = np.loadtxt(sim_data_path + "Moments1.dat")
#    else:
    moments_vs_time = np.loadtxt(sim_data_path + "Moments.dat")
    
    no_time_steps = len(t_mom)
    
    
    no_inst = moments_vs_time.shape[0]//no_time_steps
    
    print("kappa = ", kappa, no_time_steps, moments_vs_time.shape[0], no_inst)

    last_ind = moments_vs_time.shape[0]//no_time_steps * no_time_steps
    
    #print(moments_vs_time[0:last_ind].shape)
    moments_vs_time = moments_vs_time[0:last_ind]
    
    moments_vs_time2 = np.zeros( (4,no_time_steps,no_inst), dtype = np.float64 )
    
    for mom_n in range(4):
        print("kappa = ", kappa)
        moments_vs_time2[mom_n] = np.reshape(moments_vs_time[:,mom_n], 
                                             (no_inst, no_time_steps)).T
    
    moments_vs_time_avg = np.average(moments_vs_time2, axis=2)
#    if kappa == 40:
#        moments_vs_time_avg = moments_vs_time_avg[:,::5]
    moments_vs_time_kappa_var.append(moments_vs_time_avg)

#%%

TTFS, LFS, TKFS = 14,14,12

no_rows = 4
#plot_every_R = 10
fig, axes = plt.subplots(no_rows,figsize=(10,6*no_rows))
#ax = axes[0]
#ax = axes
for i,ax in enumerate(axes):
    for kappa_n, kappa in enumerate(kappa_list):
        ax.plot(t_mom/60,  moments_vs_time_kappa_var[kappa_n][i],
                "x-", label=f"{kappa}")
    ax.plot(t_Wang, mom_Wang[:,i], "o", c= "k",
            fillstyle='none', markersize = 10, mew=2.0, label="Wang")
    ax.legend()
    ax.tick_params(which="both", bottom=True, top=True,
                   left=True, right=True
                   )
    ax.tick_params(axis='both', which='major', labelsize=TKFS,
                   width=2, size=10)
    ax.tick_params(axis='both', which='minor', labelsize=TKFS,
                   width=1, size=6)
    ax.set_xlim((0,60))
for i in [0,2,3]:
    axes[i].set_yscale("log")
    axes[i].grid()
fig.tight_layout()
fig.savefig("/home/jdesk/CloudMP/CodeCompute_Unt/moments_vs_time2.pdf")    