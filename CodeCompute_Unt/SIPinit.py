

















'''
Anfangsverteilung (Exponentialverteilung) mit Single-SIP Ansatz
mit Sedimentation
'''


import numpy as np
import math


import random
import sys

def InitSIP_ExpVert_singleSIP_WS(imlow,n10,r10,min10,eta_nu,xf0,N0,nr_sip_max):
#imlow      IN :
#n10        IN : Anzahl an Bins pro Massendekade = kappa ?
#r10        IN : Anzahl an Massendekaden
#min10      IN : log von unterster Bingrenze
#xf0        IN : Mean mass in kg
#N0         IN : Anfangskonzentration in 1/m**3
#nr_sip_max IN: maximale Anzahl an SIPs

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
            if(iSIP>nr_sip_max):
                sys.exit("nr_sip_ins(k) ist groesser als nr_sip_max",iSIP,i)
    nr_SIPs=iSIP

    return  nr_SIPs,nEK_sip[0:nr_SIPs],mEK_sip[0:nr_SIPs],nEK_sip_krit_low,mdelta,mfix



