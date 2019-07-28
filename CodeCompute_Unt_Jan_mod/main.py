#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 13:38:44 2019

@author: jdesk
"""
#Python modules
import os, sys
import numpy as np
import math
import matplotlib.pyplot as plt
import warnings
import random
import time

#Aggregation modules
import SIPinit as SI
import AON_Alg as AON
import Misc as FK

import Kernel as K

import AON_Unt_Jan as my_AON

#import PlotSim as PS

warnings.filterwarnings("ignore",category =RuntimeWarning)

#-----------Parameterdefinitionen---------------------------


    #steuert Art und Anzahl der Ausgaben waehrend des Programmdurchlaufs
iPrintMessage = 2

    #Zeitschritt
dt= 10.

    #Zeitdauer
Tsim= 3600.

    #Volumen einer Gitterbox
dV=1. #in m^-3
dVi=1./dV

    #Anzahl der Instanzen
nr_inst=10
nr_ins_start=0   # bei verteilter Berechnung der Instanzen kann Seed-Parameter angepasst werden

    #Ausgabe und Speicherung der Simulationsergebnisse
        #Zeitpunkte, an denen Groessenverteilung geplottet werden soll
t_start_GVplot = 0   # in s
t_intervall_GVplot = 600   # in s

        #Zeitpunkte, an denen Momente gespeichert und geplottet werden soll
t_start_MOMsave = 0  # in s
t_intervall_MOMsave = 60 # in s


    #Density of water in kg/m**3
rho_w=1e3
    #Konversionskonstante
const_mass2rad=1./((4./3.)*math.pi*rho_w)

    #Festlegung Anfangsverteilung (SingleSIP of ExpVerteilung)
        #physikalische Parameter
            #Mean Droplet radius in m
r0=9.3*1e-6  #9.3*1e-6
            #Massenkonzentration in kg/m^3
LWC=1e-3  #1e-3
        #numerische Parameter
            #Anzahl der Bins pro Massen-Groessenordnung
n10=60
            #Anzahl an Massen-Groessenordnungen
r10=18
            #log10 von unterer Massen-Grenze
min10=-18
            #legt minimale SIP-Troepfchenmasse fest
            # = 0 verwende 0 bzw min10 als untere Grenze, =1 verwende 1.5625um als untere Grenze, =2 verwende 0.6um als untere Grenze
imlow=2
            #minimal zulaessiges SIP-Gewicht relativ zu vorhandenem Maximalgewicht
eta_nu=1e-9
            #maximale SIP-Anzahl
nr_sip_max=500

        #normalisation constants for concentration and mass, usually relevant for the discrete case
skal_n= 1.  # 1 represents a concentration of 1/cm^3 in SI units
skal_m= 1.  # 1 represents an elemental mass, i.e mass of a 17um particle in SI units


    #Definition Bin-Gitter fuer GV-Plot
n10_plot=4
r10_plot=17
nplot=n10_plot*r10_plot
min10_plot=-18

#-----------Parameterdefinitionen ENDE---------------------------

#%%

cck, mass_grid, radius_grid = K.LongKernel(rho_w)

#mass_grid_my = 1.3333E-18*math.pi * rho_w * radius_grid**3
#mass_grid_my = 4.0E-18*math.pi/3.0 * rho_w * radius_grid**3

no_grid_pts = len(mass_grid)

cck_my = np.zeros((400,400),dtype=np.float64)
for i in range(no_grid_pts):
    for j in range(i):
        cck_my[i,j] = my_AON.kernel_Long_Bott(mass_grid[i],mass_grid[j])
        cck_my[j,i] = cck[i,j]

kernel_max_dev = np.zeros(no_grid_pts)
kernel_max_dev_ind = np.zeros(no_grid_pts, dtype = int)
for i,R1 in enumerate(radius_grid):
    kernel_max_dev[i] = np.amax((cck_my[i] - cck[i]) / (cck[i] +1E-6))
    kernel_max_dev_ind[i] = np.argmax((cck_my[i] - cck[i]) / (cck[i] +1E-6))
    print(i, R1, kernel_max_dev[i], kernel_max_dev_ind[i])

Beard_data = np.loadtxt("/home/jdesk/CloudMP/CodeCompute_Unt/data/" + "VBeard.dat")

R_Beard = Beard_data[:,0]
m_Beard = Beard_data[:,1]
v_Beard = Beard_data[:,2]

#%% VELOCITY FUNCTION COMPARE

v_my = np.zeros(no_grid_pts)
for i in range(no_grid_pts):
    v_my[i] = my_AON.compute_terminal_velocity_Beard2(R_Beard[i])

v_my *= 100.

no_rows = 1
fig, axes = plt.subplots(no_rows,figsize=(10,6*no_rows))

axes.plot(R_Beard, 100*(v_my-v_Beard ) / v_Beard )

#%% FULL KERNEL COMPARE

no_rows = 4
plot_every_R = 10
fig, axes = plt.subplots(no_rows,figsize=(10,6*no_rows))
ax = axes[0]
ax.plot(R_Beard, 100*(v_my-v_Beard ) / v_Beard )
#ax.plot(radius_grid, (mass_grid-mass_grid_my)/mass_grid)
ax = axes[1]
for i,R1 in enumerate(radius_grid[::plot_every_R]):
#    ax.plot(radius_grid, cck[i*plot_every_R], label=f"{R1:.3}")
    ax.plot(radius_grid,
            (cck_my[i*plot_every_R] - cck[i*plot_every_R]) / (cck[i*plot_every_R] +1E-6) ,
                          label=f"{R1:.2}")
ax.grid()
ax.legend()

ax = axes[2]
ax.plot(radius_grid, kernel_max_dev)

ax = axes[3]
ax.plot(radius_grid, radius_grid[kernel_max_dev_ind])

