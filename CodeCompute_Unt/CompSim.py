

















#!/usr/bin/env python3
# -*- coding: utf-8 -*-
########################################################################
#
# einfache Implementierung des AON-Algorithmus'
# speichere SIP-Daten und Momente in Matrizen. Zusaetzliche Ausgabe in Dateien
# nach Beendigung der Berechnungen werden Plots von Mom(t) und von GVs erstellt
#
########################################################################

#Python modules
import os, sys
import numpy as np
import math
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
nr_inst=50
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
n10=5
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

path_cr = f'kappa_{n10}/'
if not os.path.exists(path_cr):
    os.makedirs(path_cr)
#------------------------------------------------------------------

#Definition des Bin-Gitters fuer GV-Plots
mfix_plot=np.zeros(nplot)
mdelta_plot=np.zeros(nplot)
for i in range(0,nplot-1):
    mfix_plot[i+1]=10**(i/n10_plot+min10_plot)
    mdelta_plot[i]=mfix_plot[i+1]-mfix_plot[i]

imod_GVplot=int(t_intervall_GVplot/dt)
nr_GVplot = int(1 + (Tsim-t_start_GVplot)/t_intervall_GVplot)
t_end_GVplot = t_start_GVplot + (nr_GVplot-1)*t_intervall_GVplot
t_vec_GVplot=np.arange(t_start_GVplot,t_end_GVplot+1,t_intervall_GVplot) #Zeitpunkte an denen geplottet wird
    #Matrix: speichert SIP-Daten aller Instanzen zu den GVPlot-Zeitpunkten
nEK_sip_plot=np.zeros([nr_inst,nr_GVplot,nr_sip_max])
mEK_sip_plot=np.zeros([nr_inst,nr_GVplot,nr_sip_max])
nr_SIPs_plot=np.zeros([nr_inst,nr_GVplot],dtype='int')

imod_MOMsave=int(t_intervall_MOMsave/dt)
nr_MOMsave = int(1 + (Tsim-t_start_MOMsave)/t_intervall_MOMsave)
t_end_MOMsave = t_start_MOMsave + (nr_MOMsave-1)*t_intervall_MOMsave
t_vec_MOMsave=np.arange(t_start_MOMsave,t_end_MOMsave+1,t_intervall_MOMsave) 

print(t_vec_MOMsave)
print(nr_MOMsave)

nr_MOMs=4  # notiere Momente 1 bis 3
MOMsave=np.zeros([nr_inst,nr_MOMsave,nr_MOMs])


#Aufruf Long Kernel
[cck,m_kernel,R_kernel]=K.LongKernel(rho_w)
eta_indize=m_kernel[1]/m_kernel[0]
m_low=m_kernel[0]/eta_indize


fMom = open(f'kappa_{n10}/Moments.dat', 'wb')
fGV  = open(f'kappa_{n10}/SIP.dat', 'wb')
fLog= open(f'kappa_{n10}/log.txt','a')
starttime = time.time()
localtime = time.asctime( time.localtime(starttime) )
print("Start time computation:", localtime)

fLog.write(os.getcwd()+ '\n')
fLog.write("Start time computation: "+ localtime+ '\n')
fLog.close()

#Instanzschleife
for k in range(0,nr_inst):
    print(os.getcwd())

    print('-------------------------------------- neue Instanz ',k,'-------')
    fLog= open(f'kappa_{n10}/log.txt','a')
    fLog.write('Instanz: ' + str(k)+ '\n')
    fLog.close()

    i_MOMsave=1
    i_GVplot=1
    random.seed(32+(k+nr_ins_start)*123433)
    count_colls=np.zeros(20,dtype=np.uint64)
    ###########################################################################
    #
    # Anfangsverteilung
    #
    ###########################################################################

    # Initialisiere Anfangsverteilung
    [nr_SIPs,nEK_sip_ins,mEK_sip_ins,EK_krit,mdelta,mfix]=SI.InitSIP_ExpVert_singleSIP_WS(imlow,n10,r10,min10,eta_nu,xf0,N0,nr_sip_max)
    MOMsave[k,0,:] = FK.Moments_k0_3(nEK_sip_ins,mEK_sip_ins) 
    if (iPrintMessage > 0): print('initial moments: ', MOMsave[k,0,:])
    if (iPrintMessage > 0): print('nr_SIPs: ', nr_SIPs)
    np.savetxt(fMom, MOMsave[k,0,:].reshape(1,4), fmt='%1.6e') # reshape um alle 4 Werte in eine Zeile zu schreiben
    nr_SIPs_plot[k,0 ]= nr_SIPs
    nEK_sip_plot[k,0,0:nr_SIPs]=nEK_sip_ins
    mEK_sip_plot[k,0,0:nr_SIPs]=mEK_sip_ins
    np.savetxt(fGV, [nr_SIPs], fmt='%5i')          
    np.savetxt(fGV, nEK_sip_ins[0:nr_SIPs].reshape(1,nr_SIPs), fmt='%1.6e')          
    np.savetxt(fGV, mEK_sip_ins[0:nr_SIPs].reshape(1,nr_SIPs), fmt='%1.6e')  
    
    ibreak = 0
    iibreak = 0
    # Zeitschleife
    for it in range(0,iend):
        t=it*dt
        ibreak = AON.Aggregation(dt,nr_SIPs,nEK_sip_ins,mEK_sip_ins,dV,count_colls,cck,m_low,eta_indize,m_kernel)
        if (it%imod_MOMsave ==0) & (it != 0): 
            MOMsave[k,i_MOMsave,:]=FK.Moments_k0_3(nEK_sip_ins,mEK_sip_ins)
            if (iPrintMessage > 0): print(it,i_MOMsave,MOMsave[k,i_MOMsave,:])
            np.savetxt(fMom, MOMsave[k,i_MOMsave,:].reshape(1,4), fmt='%1.6e')          
            i_MOMsave = i_MOMsave+1

        #entferne SIPs mit Gewicht 0, wenn ueberhaupt vorhanden
        if (min(nEK_sip_ins) == 0):
            index_list=nEK_sip_ins.nonzero()
            print('nr_SIPs alt: ',nr_SIPs)
            nEK_sip_ins=nEK_sip_ins[index_list]
            mEK_sip_ins=mEK_sip_ins[index_list]
            nr_SIPs = nEK_sip_ins.size
            print('nr_SIPs neu: ',nr_SIPs)
            if (nr_SIPs == 1):
                print('only one SIP remains, interrupt computation of the current instance, proceed with next instance' )
                print('nu: ', nEK_sip_ins, 'm: ', mEK_sip_ins)
                ibreak = 1

        #Einspeichern der Werte zum Plotten nach imod_GVplot Iterationen
        if (it%imod_GVplot == 0) & (it != 0):
            nr_SIPs_plot[k,i_GVplot ]= nr_SIPs
            nEK_sip_plot[k,i_GVplot,0:nr_SIPs]=nEK_sip_ins
            mEK_sip_plot[k,i_GVplot,0:nr_SIPs]=mEK_sip_ins
            if (iPrintMessage > 0): print('GV rausschreiben, #,Iter-schritt, Zeit:', i_GVplot, it, it*dt)
            np.savetxt(fGV, [nr_SIPs], fmt='%5i')          
            np.savetxt(fGV, nEK_sip_plot[k,i_GVplot,0:nr_SIPs].reshape(1,nr_SIPs), fmt='%1.6e')          
            np.savetxt(fGV, mEK_sip_plot[k,i_GVplot,0:nr_SIPs].reshape(1,nr_SIPs), fmt='%1.6e')      
            i_GVplot = i_GVplot+1

        if (ibreak == 1):
            iibreak +=1
            print('break condition met at iteration it = ', it)
            if (iibreak ==2): break

    #Ende der Zeititeration einer Instanz
    
    #print(count_colls/count_colls.sum())
    
    #Rechenzeitanalyse
    currenttime = time.time()
    currenttime_str = time.asctime( time.localtime(currenttime))
    endtime_expected=starttime+ (nr_inst/(k+1)) * (currenttime-starttime)
    endtime_expected_str = time.asctime( time.localtime(endtime_expected))

    print("Instance {} of {} finished".format(k+1,nr_inst))
    print("Start time/Current time/Expected end time: ")
    print(localtime, ' --- ', currenttime_str, ' --- ', endtime_expected_str)
    fLog= open(f'kappa_{n10}/log.txt','a')
    fLog.write('total computing time in sec: '+ str(int(currenttime-starttime)) + '\n')
    fLog.write(currenttime_str+ ' --- '+ endtime_expected_str+ '\n')
    fLog.close()
#Ende der Schleife ueber Instanzen  
#---------------------------------------------------------------------------------------------

fMom.close()
fGV.close()
localtime = time.asctime( time.localtime(time.time()) )
print("End time computation:", localtime)

#Erstelle Metadaten der ausgegebenen Dateien
fMom = open(f'kappa_{n10}/Moments_meta.dat', 'wb')
np.savetxt(fMom,np.array([nr_inst,nr_MOMsave]),fmt = '%4d')
np.savetxt(fMom, t_vec_MOMsave.reshape(1,nr_MOMsave),fmt = '%6d')
fMom.close()
fGV  = open(f'kappa_{n10}/SIP_meta.dat', 'wb')
np.savetxt(fGV,np.array([nr_inst,nr_GVplot]),fmt = '%4d')
np.savetxt(fGV, t_vec_GVplot.reshape(1,nr_GVplot),fmt = '%6d')
fGV.close()
#Erstelle Dateien mit gemittelten Werte
#FK.CIO_MOMmean(data_in=MOMsave,fp_out='')

fLog= open(f'kappa_{n10}/log.txt','a')
currenttime = time.time()
fLog.write('total computing time in sec: '+ str(int(currenttime-starttime)) + '\n')
fLog.write('finalised')
fLog.close()

#---------------------------------------------------------------------------------------------
#Plotten der Simulation

#PS.PlotMoments(MOMsave,t_vec_MOMsave)
#PS.PlotGV(nEK_sip_plot,mEK_sip_plot,nr_SIPs_plot,t_vec_GVplot)

