#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:25:25 2019

@author: jdesk
"""

import numpy as np
import random
import math

import matplotlib.pyplot as plt

#%%

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
nr_inst=500
nr_ins_start=1   # bei verteilter Berechnung der Instanzen kann Seed-Parameter angepasst werden

    #Ausgabe und Speicherung der Simulationsergebnisse
        #Zeitpunkte, an denen Groessenverteilung geplottet werden soll
t_start_GVplot = 0   # in s
t_intervall_GVplot = 600   # in s

        #Zeitpunkte, an denen Momente gespeichert und geplottet werden soll
t_start_MOMsave = 0  # in s
t_intervall_MOMsave = 300 # in s

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
            #! = KAPPA !
n10=10
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
#nr_sip_max=500
nr_sip_max=3000

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


#%%

print('Initialize Exponential Distribution, kappa = ', n10)
nr_SIPs = 0

#Eigenschaften Bingitter
nr_bins=n10*r10
mfix=np.zeros(nr_bins)
mdelta=np.zeros(nr_bins)
m=np.zeros(nr_bins-1)


nEK_sip=np.zeros(nr_sip_max)
mEK_sip=np.zeros(nr_sip_max)
nEK_sip_krit_low=0

m_low = 0
if(imlow == 1):
    m_low=(4/3)*math.pi*1e3*(1.5625e-6)**3.0
if(imlow == 2):
    m_low=(4/3)*math.pi*1e3*(0.6e-6)**3.0
    
nEK_sip_tmp=np.zeros(nr_bins)
mfix[0]=10**min10
for i in range(1,nr_bins):
    mfix[i]=10**((i-1)/n10+min10)
    mdelta[i-1]=mfix[i]-mfix[i-1]
    m[i-1]=mfix[i-1]+random.random()*mdelta[i-1]
    
s=m/xf0
### array operation here, loop not nec.
for i in range(0,nr_bins-1-1):
    nEK_sip_tmp[i]=N0/xf0*math.exp(-s[i])*mdelta[i]

nEK_sip_krit_low=max(nEK_sip_tmp)*eta_nu
iSIP=0
for i in range(0,nr_bins):
    if(nEK_sip_tmp[i]>nEK_sip_krit_low and m[i]>m_low):
        nEK_sip[iSIP]=nEK_sip_tmp[i]
        mEK_sip[iSIP]=m[i]
        iSIP=iSIP+1
#        if(iSIP>nr_sip_max):
#            sys.exit("nr_sip_ins(k) ist groesser als nr_sip_max",iSIP,i)
nr_SIPs=iSIP

#%%
imax = 90

fig, ax = plt.subplots(1)
ax.plot( m[:imax], nEK_sip_tmp[0:imax] )
ax.vlines(mfix[0:imax], 1E-200,1E-5)
    
ax.set_xscale("log")
ax.set_yscale("log")


imax = 90

fig, ax = plt.subplots(1)
ax.plot( mEK_sip, nEK_sip, "x" )
ax.vlines(mfix[0:imax], 1E-200,1E-5)
    
ax.set_xscale("log")
ax.set_yscale("log")


#%%
#print('Initialize Exponential Distribution, kappa = ', n10)
#nr_SIPs = 0
#
##Eigenschaften Bingitter
#nr_bins=n10*r10
#mfix=np.zeros(nr_bins)
#mdelta=np.zeros(nr_bins)
#m=np.zeros(nr_bins-1)
#
#
#nEK_sip=np.zeros(nr_sip_max)
#mEK_sip=np.zeros(nr_sip_max)
#nEK_sip_krit_low=0
#
#m_low = 0
#if(imlow == 1):
#    m_low=(4/3)*math.pi*1e3*(1.5625e-6)**3.0
#if(imlow == 2):
#    m_low=(4/3)*math.pi*1e3*(0.6e-6)**3.0
#    
#nEK_sip_tmp=np.zeros(nr_bins)
#mfix[0]=10**min10
#for i in range(1,nr_bins):
#    mfix[i]=10**((i-1)/n10+min10)
#    mdelta[i-1]=mfix[i]-mfix[i-1]
#    m[i-1]=mfix[i-1]+random.random()*mdelta[i-1]
#    
#s=m/xf0
#### array operation here, loop not nec.
#for i in range(0,nr_bins-1-1):
#    nEK_sip_tmp[i]=N0/xf0*math.exp(-s[i])*mdelta[i]
#
#nEK_sip_krit_low=max(nEK_sip_tmp)*eta_nu
#iSIP=0
#for i in range(0,nr_bins):
#    if(nEK_sip_tmp[i]>nEK_sip_krit_low and m[i]>m_low):
#        nEK_sip[iSIP]=nEK_sip_tmp[i]
#        mEK_sip[iSIP]=m[i]
#        iSIP=iSIP+1
#        if(iSIP>nr_sip_max):
#            sys.exit("nr_sip_ins(k) ist groesser als nr_sip_max",iSIP,i)
#nr_SIPs=iSIP