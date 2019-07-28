#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 14:53:32 2019

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

#import PlotSim as PS

warnings.filterwarnings("ignore",category =RuntimeWarning)

#the following statement includes a parameter file via a preprocessor directive


















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
n10=40
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

##-----------abgeleitete Parameter---------------------------

    #Anfangs-GV
        #Mean mass in kg
xf0=r0**(3.0)/const_mass2rad
        #Anfangskonzentration in 1/m**3
        #N0=239*(10**6)
N0=LWC/xf0
        #Binanzahl
n=n10*r10

    #Gesamtanzahl der Zeitschritte
iend=int(Tsim/dt) + 1
#------------------------------------------------------------------

#%%
nr_SIPs,nEK_sip_ins,mEK_sip_ins,EK_krit,mdelta,mfix = \
SI.InitSIP_ExpVert_singleSIP_WS(imlow,n10,r10,min10,eta_nu,xf0,N0,nr_sip_max)

REK = (mEK_sip_ins*const_mass2rad)**(0.33333333) * 1.0E6
no_rows = 1
#plot_every_R = 10
fig, axes = plt.subplots(no_rows,figsize=(10,6*no_rows))
#ax = axes[0]
ax = axes
#ax.plot(mEK_sip_ins, nEK_sip_ins )
ax.plot(REK, nEK_sip_ins )

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_yticks(np.logspace(-3,8,12))
ax.grid()

#%%
m_maxx = 9.169582e-05
print( (m_maxx*const_mass2rad)**(0.33333333) * 1.0E6)

