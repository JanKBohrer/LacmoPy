#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:24:41 2019

@author: jdesk
"""
#%% IMPORTS AND DEFS
import os
import math
import numpy as np
from numba import njit

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

#import Kernel as K
# from microphysics import compute_mass_from_radius
#from microphysics import compute_radius_from_mass
#from microphysics import compute_radius_from_mass_jit

import sys
# Linux Desk
sys_path1 = "/home/jdesk/CloudMP"
sys.path.append(sys_path1)

import AON_Unt_algo_Jan as AON_my
import constants as c

# R_low, R_high in mu = 1E-6 m
# no_bins_10: number of bins per radius decade
# mass_density in kg/m^3
# R_low and R_high are both included in the radius_grid range interval
# but R_high itself might NOT be a value of the radius_grid
# (which is def by no_bins_10 and R_low)
def generate_kernel_grid_Long_Bott_np(R_low, R_high, no_bins_10,
                                      mass_density):
    
    bin_factor = 10**(1.0/no_bins_10)
    
    no_bins = int(math.ceil( no_bins_10 * math.log10(R_high/R_low) ) ) + 1
    
    radius_grid = np.zeros( no_bins, dtype = np.float64 )
    radius_grid[0] = R_low
    for bin_n in range(1,no_bins):
        radius_grid[bin_n] = radius_grid[bin_n-1] * bin_factor
    
    # generate velocity grid first at pos. of radius_grid
    vel_grid = np.zeros( no_bins, dtype = np.float64 )
    
    for i in range(no_bins):
        vel_grid[i] = AON_my.compute_terminal_velocity_Beard2(radius_grid[i])
    
    kernel_grid = np.zeros( (no_bins, no_bins), dtype = np.float64 )
    
    for j in range(1,no_bins):
        R_j = radius_grid[j]
        v_j = vel_grid[j]
        for i in range(j):
            R_i = radius_grid[i]
            ### Kernel "LONG" provided by Bott (via Unterstrasser)
            if R_j <= 50.0:
                E_col = 4.5E-4 * R_j * R_j \
                        * ( 1.0 - 3.0 / ( max(3.0, R_i) + 1.0E-2 ) )
            else: E_col = 1.0
            
            kernel_grid[j,i] = 1.0E-12 * math.pi * (R_i + R_j) * (R_i + R_j) \
                               * E_col * abs(v_j - vel_grid[i]) 
            kernel_grid[i,j] = kernel_grid[j,i]

    c_radius_to_mass = 4.0E-18 * math.pi * mass_density / 3.0
#    c_mass_to_radius = 1.0 / c_radius_to_mass
    mass_grid = c_radius_to_mass * radius_grid**3
    

    
    return kernel_grid, vel_grid, mass_grid, radius_grid
generate_kernel_grid_Long_Bott = njit()(generate_kernel_grid_Long_Bott_np)

def generate_kernel_grid_Long_Bott_given_R_np(radius_grid, mass_density):
    c_radius_to_mass = 4.0E-18 * math.pi * mass_density / 3.0
#    c_mass_to_radius = 1.0 / c_radius_to_mass
    no_grid_pts = radius_grid.shape[0]
    mass_grid = c_radius_to_mass * radius_grid**3
    
    # generate velocity grid first at pos. of radius_grid
    vel_grid = np.zeros( no_grid_pts, dtype = np.float64 )
    
    for i in range(no_grid_pts):
        vel_grid[i] = AON_my.compute_terminal_velocity_Beard2(radius_grid[i])
    
    kernel_grid = np.zeros( (no_grid_pts, no_grid_pts), dtype = np.float64 )
    
    for j in range(1,no_grid_pts):
        R_j = radius_grid[j]
        v_j = vel_grid[j]
        for i in range(j):
            R_i = radius_grid[i]
            ### Kernel "LONG" provided by Bott (via Unterstrasser)
            if R_j <= 50.0:
                E_col = 4.5E-4 * R_j * R_j \
                        * ( 1.0 - 3.0 / ( max(3.0, R_i) + 1.0E-2 ) )
            else: E_col = 1.0
            
            kernel_grid[j,i] = 1.0E-12 * math.pi * (R_i + R_j) * (R_i + R_j) \
                               * E_col * abs(v_j - vel_grid[i]) 
            kernel_grid[i,j] = kernel_grid[j,i]
    
#    np.save(save_dir + "radius_grid_out.npy", radius_grid)
#    np.save(save_dir + "mass_grid_out.npy", mass_grid)
#    np.save(save_dir + "kernel_grid.npy", kernel_grid)
#    np.save(save_dir + "velocity_grid.npy", vel_grid)
    
    return kernel_grid, vel_grid, mass_grid, radius_grid
generate_kernel_grid_Long_Bott_given_R = \
    njit()(generate_kernel_grid_Long_Bott_given_R_np)

def generate_and_save_kernel_grid_Long_Bott(R_low, R_high, no_bins_10,
                                               mass_density, save_dir):
    kernel_grid, vel_grid, mass_grid, radius_grid = \
        generate_kernel_grid_Long_Bott(R_low, R_high, no_bins_10,
                                       mass_density)
    
    np.save(save_dir + "radius_grid_out.npy", radius_grid)
    np.save(save_dir + "mass_grid_out.npy", mass_grid)
    np.save(save_dir + "kernel_grid.npy", kernel_grid)
    np.save(save_dir + "velocity_grid.npy", vel_grid)        
    
    return kernel_grid, vel_grid, mass_grid, radius_grid

# with my kernel(R_i,R_j) full
#def generate_kernel_grid_Long_Bott_np(radius_grid, mass_density):
#    c_radius_to_mass = 4.0E-18 * math.pi * mass_density / 3.0
##    c_mass_to_radius = 1.0 / c_radius_to_mass
#    
#    no_grid_pts = radius_grid.shape[0]
#    
#    mass_grid = c_radius_to_mass * radius_grid**3
#    
#    kernel_grid = np.zeros( (no_grid_pts, no_grid_pts), dtype = np.float64 )
#    
#    for i in range(1,no_grid_pts):
#        for j in range(i):
#            kernel_grid[i,j] = \
#                AON_my.kernel_Long_Bott_R(radius_grid[i], radius_grid[j])
#            kernel_grid[j,i] = kernel_grid[i,j]
#    
#    return kernel_grid, mass_grid, radius_grid
#generate_kernel_grid_Long_Bott = njit()(generate_kernel_grid_Long_Bott_np)

#%% generate kernel grid

dist = "expo"
kernel = "Long_Bott"

OS = "LinuxDesk"
# OS = "MacOS"
if OS == "MacOS":
    sim_data_path = "/Users/bohrer/sim_data/"
elif OS == "LinuxDesk":
    sim_data_path = "/mnt/D/sim_data_my_kernel_grid/"

save_dir = sim_data_path + f"col_box_mod/results/{dist}/{kernel}/kernel_data/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

plotting = False

mass_density = 1.0E3

#radius_grid = np.loadtxt("data_Long_kernel/Long_radius.txt").flatten()[:-2]


#%%

# create own radius_grid
# need: R_low, R_high (in mu)

# R_(i+1) = R_i * factor ; factor = 10**(1/kappa_R)

no_bins_10 = 200

R_low = 0.6
R_high = 6.0E3 

kernel_grid, vel_grid, mass_grid, radius_grid = \
    generate_and_save_kernel_grid_Long_Bott(R_low, R_high, no_bins_10,
                                            mass_density, save_dir)
#    generate_kernel_grid_Long_Bott(radius_grid, mass_density)

no_kernel_bins = radius_grid.shape[0]
print(radius_grid.shape)

radius_grid_log = np.log(radius_grid)
#print(radius_grid_log[1:] - radius_grid_log[:-1])

bin_factor_m = mass_grid[1] / mass_grid[0]
bin_factor_m_log = math.log(bin_factor_m)

m_kernel_low = mass_grid[0]
m_kernel_low_log = math.log(m_kernel_low)

indi = np.floor( np.log( mass_grid/m_kernel_low ) / np.log( bin_factor_m ) + 0.5)

ind_kernel = np.zeros(no_kernel_bins, dtype = np.int64)

for i in range(no_kernel_bins):
    # auto conversion to int due to dtype of ind_kernel
    ind_kernel[i] = \
        ( math.log(mass_grid[i]) - m_kernel_low_log ) / bin_factor_m_log + 0.5
    if ind_kernel[i] < 0:
        ind_kernel[i] = 0
    elif ind_kernel[i] > (no_kernel_bins - 1):
        ind_kernel[i] = no_kernel_bins - 1

#%%

no_rows = 1
fig, axes = plt.subplots(no_rows,figsize=(10,6*no_rows))
#f = 1E6*(vel_grid-v_Beard ) / v_Beard
axes.plot(radius_grid, np.ones_like(radius_grid), "x")
#axes.vlines([10.0,535.,3500.],f.min(), f.max())
axes.set_xscale("log")

#%%

data_Beard = np.loadtxt("data_Long_kernel/VBeard.dat").T
R_Beard = data_Beard[0]
v_Beard = 1.0E-2 * data_Beard[2]

no_rows = 1
fig, axes = plt.subplots(no_rows,figsize=(10,6*no_rows))
f = vel_grid
axes.plot(radius_grid, f , "o" )
axes.plot(R_Beard, v_Beard )
axes.vlines([10.0,535.,3500.],f.min(), f.max())
axes.set_xscale("log")

axes.set_xlabel(r"radius $(\mathrm{\mu m})$ ")
axes.set_ylabel(r" velocity (m/s) ")

#fig.savefig("my_vel_rel_dev_to_Beard_tabul.pdf")

#%%

#save_dir = sim_data_path + f"col_box_mod/results/{dist}/{kernel}/kernel_data_given_R/"
#if not os.path.exists(save_dir):
#    os.makedirs(save_dir)

#kernel_grid2, vel_grid2, mass_grid2, radius_grid2 = \
#    generate_kernel_grid_Long_Bott_given_R(R_Beard, mass_density)
#
##for i,k in enumerate(kernel_grid):
##    for j in range(400):
##        if ( abs(kernel_grid[i,j]-kernel_grid[j,i]) > 1E-18 ):
##            print(i,j,abs(kernel_grid[i,j]-kernel_grid[j,i]))
#
##print(kernel_grid - kernel_grid)
#
#no_grid_pts_m = mass_grid2.shape[0]
#cck=np.loadtxt("data_Long_kernel/Values_Longkernel.txt") # given in cm^3/s
#nxn=cck.size-2
#cck=cck.flatten()[0:nxn]
#cck=np.reshape(cck, (no_grid_pts_m, no_grid_pts_m))
#cck *= 1.0E-6
#
#for i,k in enumerate(kernel_grid2):
#    for j in range(400):
#        if ( abs(cck[i,j] - cck[j,i]) > 1E-16 ):
#            print(i,j,abs(cck[i,j] - cck[j,i]))


#%% VELOCITY FUNCTION COMPARE

#no_rows = 1
#fig, axes = plt.subplots(no_rows,figsize=(10,6*no_rows))
#f = 1E6*(vel_grid2-v_Beard ) / v_Beard
#axes.plot(radius_grid2, f )
#axes.vlines([10.0,535.,3500.],f.min(), f.max())
#axes.set_xscale("log")
#
#axes.set_xlabel(r"radius $(\mathrm{\mu m})$ ")
#axes.set_ylabel(r"rel. dev. $(v - v_\mathrm{Beard})/v_\mathrm{Beard}$ (1E-6)")

#fig.savefig("my_vel_rel_dev_to_Beard_tabul.pdf")

#%% PLOTTING AND CHECKING

#print( math.log(radius_grid[0]) - math.log(0.6) )
#print( math.log(radius_grid[1]) - math.log(radius_grid[0]) )
#print( radius_grid[1]/radius_grid[0] )
#
#fac_R = radius_grid[1] / radius_grid[0]
#
#radius_grid_ext = np.zeros(403)
#radius_grid_ext[3:] = radius_grid
#radius_grid_ext[2] = radius_grid_ext[3] /fac_R
#radius_grid_ext[1] = radius_grid_ext[2] /fac_R
#radius_grid_ext[0] = radius_grid_ext[1] /fac_R
#
#radius_grid_ext_center = 0.5 * (radius_grid_ext[:-1] + radius_grid_ext[1:])
#
#radius_grid_ext_center_log = np.exp( 0.5 * (np.log(radius_grid_ext[:-1])
#                                    + np.log(radius_grid_ext[1:]) ))
#
####
#
#bins_rad = np.zeros(402)
#bins_rad[0] = 0.6
#for i in range(401):
#    bins_rad[i+1] = bins_rad[i] * 1.02337389
#
#bins_rad_log = np.log(bins_rad)
#
#bins_rad_center_log_log = 0.5 * (bins_rad_log[:-1] + bins_rad_log[1:])
#
#bins_rad_center_log = np.exp(bins_rad_center_log_log)
#
#bins_rad_center = 0.5 * (bins_rad[:-1] + bins_rad[1:])
#
####
#
#radius_grid_log = np.log(radius_grid)
#
#radius_grid_center = 0.5 * (radius_grid[:-1] + radius_grid[1:])
#
#no_rows = 1
#no_cols = 1
##plot_every_R = 10
#fig, axes = plt.subplots(no_rows, no_cols, figsize=(12*no_cols,4*no_rows))
#
##axes.plot(bins_rad_center_log, np.ones_like(radius_grid), "x")
##axes.plot(radius_grid, bins_rad_log[2:] - radius_grid_log, "x")
##axes.plot(radius_grid_ext, bins_rad, "x")
#axes.plot(radius_grid_ext[1:] / radius_grid_ext[:-1] )
##axes.plot(radius_grid_ext, bins_rad, "x")
##axes.plot(radius_grid_ext, radius_grid_ext)
##axes.plot(radius_grid_ext, bins_rad - radius_grid_ext, "x")
##axes.vlines(radius_grid, 0.999,1.001)
##axes.plot(bins_rad, np.ones_like(bins_rad), "x")
##axes.vlines(radius_grid, 0.999,1.001)
#
##axes.set_xscale("log")
##axes.set_yscale("log")
#
#%%

if plotting:
    cck_limit_low = 1.0E-16
    cck_value_low = 1.0E-8
    cck_no_zeros = np.where(cck < cck_limit_low, cck_value_low, cck)
    
    #%% PLOTTING
    
    plt.ioff()
    
    split = np.linspace(0,400,21).astype(int)
    
    print(split)
    print(split.shape)
    
    
    no_rows = 10
    no_cols = 2
    #plot_every_R = 10
    fig, axes = plt.subplots(no_rows, no_cols, figsize=(5*no_cols,8*no_rows))
    #ax = axes[0]
    axes = axes.flatten()
    for i,ax in enumerate(axes):
        for j in range( split[i], split[i+1] ):
    #        print(i,j)
            f = 1E5*(kernel_grid[j]-cck[j]) / cck_no_zeros[j]
            ax.plot( radius_grid, f,
                    label = f"{radius_grid[j]:.1e}")
            ax.vlines( (10.,50.,535.,3500.), f.min(), f.max(), linestyle="dashed")
            ax.legend()
    #    R_0 = 10.0
    #    R_1 = 535.0
    #    R_max = 3500.0
    
    #for i in range( cck.shape[0] ):
    #    ax.plot( radius_grid, np.abs(cck[i]-kernel_grid[i]) / cck_no_zeros[i])
    #    ax.plot( radius_grid, np.abs(cck[i]-kernel_grid[i]) / (cck[i] + 1E-6) )
    #    ax.plot( radius_grid, kernel_grid[i] )
    
        ax.set_xscale("log")
    #ax.set_yscale("log")
    #ax.set_ylim([1E-5,1E-3])
    fig.suptitle("vlines = 10,50,535,3500 mu, yscale = 1E-5", y = 0.999)
    fig.tight_layout()
    fig.savefig("kernel_compare.png")
    #fig.savefig("kernel_compare.pdf")
    
    #%%
    no_rows = 10
    no_cols = 2
    #plot_every_R = 10
    fig, axes = plt.subplots(no_rows, no_cols, figsize=(5*no_cols,8*no_rows))
    #ax = axes[0]
    axes = axes.flatten()
    for i,ax in enumerate(axes):
        for j in range( split[i], split[i+1] ):
    #        print(i,j)
            f = cck_no_zeros[j]
    #        f = 1E5*(kernel_grid[j]-cck[j]) / cck_no_zeros[j]
            ax.plot( radius_grid, f,
                    label = f"{radius_grid[j]:.1e}")
            ax.vlines( (10.,50.,535.,3500.), f.min(), f.max(), linestyle="dashed")
            ax.legend()
    #    R_0 = 10.0
    #    R_1 = 535.0
    #    R_max = 3500.0
    
    #for i in range( cck.shape[0] ):
    #    ax.plot( radius_grid, np.abs(cck[i]-kernel_grid[i]) / cck_no_zeros[i])
    #    ax.plot( radius_grid, np.abs(cck[i]-kernel_grid[i]) / (cck[i] + 1E-6) )
    #    ax.plot( radius_grid, kernel_grid[i] )
    
        ax.set_xscale("log")
        ax.set_yscale("log")
    #ax.set_yscale("log")
    #ax.set_ylim([1E-5,1E-3])
    fig.suptitle("vlines = 10,50,535,3500 mu, yscale = 1", y = 0.999)
    fig.tight_layout()
    fig.savefig("cck_no_zeros.png")
    #fig.savefig("kernel_compare.pdf")
    
    #%%
    no_rows = 10
    no_cols = 2
    #plot_every_R = 10
    fig, axes = plt.subplots(no_rows, no_cols, figsize=(5*no_cols,8*no_rows))
    #ax = axes[0]
    axes = axes.flatten()
    for i,ax in enumerate(axes):
        for j in range( split[i], split[i+1] ):
    #        print(i,j)
            f = np.abs(kernel_grid[j]-cck[j])
            ax.plot( radius_grid, f,
                    label = f"{radius_grid[j]:.1e}")
            ax.vlines( (10.,50.,535.,3500.), f.min(), f.max(), linestyle="dashed")
            ax.legend()
    #    R_0 = 10.0
    #    R_1 = 535.0
    #    R_max = 3500.0
    
    #for i in range( cck.shape[0] ):
    #    ax.plot( radius_grid, np.abs(cck[i]-kernel_grid[i]) / cck_no_zeros[i])
    #    ax.plot( radius_grid, np.abs(cck[i]-kernel_grid[i]) / (cck[i] + 1E-6) )
    #    ax.plot( radius_grid, kernel_grid[i] )
    
        ax.set_xscale("log")
        ax.set_yscale("log")
    #ax.set_yscale("log")
    #ax.set_ylim([1E-5,1E-3])
    fig.suptitle("vlines = 10,50,535,3500 mu, yscale = 1", y = 0.999)
    fig.tight_layout()
    fig.savefig("kernel_compare_abs_error.png")
    #fig.savefig("kernel_compare_abs_error.pdf")
    
    #%%
    
    no_rows = 10
    no_cols = 2
    #plot_every_R = 10
    fig, axes = plt.subplots(no_rows, no_cols, figsize=(5*no_cols,8*no_rows))
    #ax = axes[0]
    axes = axes.flatten()
    for i,ax in enumerate(axes):
        for j in range( split[i], split[i+1] ):
    #        print(i,j)
    #        f = 1E5*(kernel_grid[j]-cck[j]) / cck_no_zeros[j]
            f = kernel_grid[j]
            ax.plot( radius_grid, f,
                    label = f"{radius_grid[j]:.1e}")
            ax.vlines( (10.,50.,535.,3500.), f.min(), f.max(), linestyle="dashed")
            ax.legend()
    #    R_0 = 10.0
    #    R_1 = 535.0
    #    R_max = 3500.0
    
    #for i in range( cck.shape[0] ):
    #    ax.plot( radius_grid, np.abs(cck[i]-kernel_grid[i]) / cck_no_zeros[i])
    #    ax.plot( radius_grid, np.abs(cck[i]-kernel_grid[i]) / (cck[i] + 1E-6) )
    #    ax.plot( radius_grid, kernel_grid[i] )
    
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_yticks( np.logspace(-24,-3,22) )
        ax.set_ylim( (1.0E-24,1.0E-3) )
    #ax.set_yscale("log")
    #ax.set_ylim([1E-5,1E-3])
    fig.suptitle("vlines = 10,50,535,3500 mu, yscale = 1", y = 0.999)
    fig.tight_layout()
    fig.savefig("my_kernel_grid.png")
    #fig.savefig("my_kernel_grid.pdf")
    
    #%% Beard kernel
    
    no_rows = 10
    no_cols = 2
    #plot_every_R = 10
    fig, axes = plt.subplots(no_rows, no_cols, figsize=(5*no_cols,8*no_rows))
    #ax = axes[0]
    axes = axes.flatten()
    for i,ax in enumerate(axes):
        for j in range( split[i], split[i+1] ):
    #        print(i,j)
    #        f = 1E5*(kernel_grid[j]-cck[j]) / cck_no_zeros[j]
            f = cck[j]
            ax.plot( radius_grid, f,
                    label = f"{radius_grid[j]:.1e}")
            ax.vlines( (10.,50.,535.,3500.), f.min(), f.max(), linestyle="dashed")
            ax.legend()
    #    R_0 = 10.0
    #    R_1 = 535.0
    #    R_max = 3500.0
    
    #for i in range( cck.shape[0] ):
    #    ax.plot( radius_grid, np.abs(cck[i]-kernel_grid[i]) / cck_no_zeros[i])
    #    ax.plot( radius_grid, np.abs(cck[i]-kernel_grid[i]) / (cck[i] + 1E-6) )
    #    ax.plot( radius_grid, kernel_grid[i] )
    
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_yticks( np.logspace(-24,-3,22) )
        ax.set_ylim( (1.0E-24,1.0E-3) )
    #ax.set_yscale("log")
    #ax.set_ylim([1E-5,1E-3])
    fig.suptitle("vlines = 10,50,535,3500 mu, yscale = 1", y = 0.999)
    fig.tight_layout()
    fig.savefig("Bott_kernel_grid.png")
    #fig.savefig("my_kernel_grid.pdf")
