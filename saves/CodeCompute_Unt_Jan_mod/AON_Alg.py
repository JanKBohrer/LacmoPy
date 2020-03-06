


















import math
import random
import numpy as np


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





def Aggregation(dt,nr_SIPs,nEK_sip_tmp,mEK_sip_tmp,dV,count_colls,cck,m_low,eta_indize,m_kernel):
#dt             : Zeitschritt in s
#cck            : Kernel-Matrixwerte
#nr_SIPs        :
#nEK_sip_tmp    : Vektor mit SIP-Gewichtsfaktor (will be multiplied by skal_n in KCE)
#mEK_sip_tmp    : Vektor mit SIP-Troepfchenmassen (will be multiplied by skal_m in KCE)
#m_low          : kleinste Masse fuer die Kernelwert in Matrix gegeben ist
#eta_indize     : Anzahl der Kernelwerte in Matrix pro Massendekade
#m_kernel       : mass grid on which kernel values are given
#dV             : Volumen der Gitterbox
#count_colls    : 10-array: counts the number of collections
   
    ncoll=len(count_colls)
    dVi=1./dV
    ibreak = 0

    indize=np.zeros(mEK_sip_tmp.shape)
    for i in range(0,max(mEK_sip_tmp.shape)):
        if(math.log(mEK_sip_tmp[i]*skal_m)>math.log(m_low)):
            indize[i]=(math.log(mEK_sip_tmp[i]*skal_m)-math.log(m_low))/math.log(eta_indize)
    #print(min(indize),max(indize),nr_SIPs)
    np.clip(indize,0,398,out=indize)

    indize_floored=np.floor(indize+0.5).astype(int)  # floor liefert float values, in int umwandeln, damit die Werte als Indizes verwendet werden koennen
    #nun zentriert

    const=dt*dVi

    for i in range(0,nr_SIPs):
        for j in range(i+1,nr_SIPs):
            iiu=indize_floored[i]
            iju=indize_floored[j]
            cck_value=cck[iiu,iju]*const
            nEK_agg_tmp=cck_value*nEK_sip_tmp[i]*nEK_sip_tmp[j]*skal_n   # eigentlich linke mit skal_n und rechte Seite mit skal_n^2 multiplizieren
            mEK_agg_tmp=mEK_sip_tmp[i]+mEK_sip_tmp[j]

            tmp_min=min(nEK_sip_tmp[i],nEK_sip_tmp[j])
            if(tmp_min!=0):
                c_mc = nEK_agg_tmp/tmp_min
                c_mcCEIL = math.ceil(c_mc)
                p_col=c_mc-math.floor(c_mc)
            else:
                c_mcCEIL=0
                p_col=0

            #Erzeugung Zufallszahl (auf Grundlage des Mersenne Twisters)
            pkrit=random.random()
            if (c_mcCEIL == 1):
                if(p_col>pkrit):
                    count_colls[1] += 1
                    if(nEK_sip_tmp[i]<nEK_sip_tmp[j]):
                        i1=i; i2=j
                    else:
                        i1=j; i2=i
                    nEK_sip_tmp[i2]=nEK_sip_tmp[i2]-nEK_sip_tmp[i1]
                    mEK_sip_tmp[i1]=mEK_agg_tmp
                else:
                    count_colls[0] += 1
            elif (c_mcCEIL > 1):
                # Multiple Collection case
                if(nEK_sip_tmp[i]<nEK_sip_tmp[j]):
                    i1=i; i2=j
                else:
                    i1=j; i2=i
                #i1 ist das SIP mit kleinerem Gewichtsfaktor
            #Floating Point Multiple Collection, das berechnete c_mc wird verwendet, diese Operation generiert nicht-ganzzahlige Vielfache der Ausgangsmassen
                mEK_sip_tmp[i1]=(mEK_sip_tmp[i1]+c_mc*mEK_sip_tmp[i2])
                nEK_sip_tmp[i2]=nEK_sip_tmp[i2]-nEK_sip_tmp[i1]*c_mc
                index=min([math.ceil(c_mc -0.5),ncoll-1])  # 1 < c_mc < 1.5 -> Index 1; 1.5 < c_mc < 2.5 -< Index 2
                count_colls[index] += 1

    #print('mEK_sip_tmp', mEK_sip_tmp)
    #print('nEK_sip_tmp', nEK_sip_tmp)
    return ibreak
#end subroutine Aggregation(dt,nr_SIPs,nEK_sip_tmp,mEK_sip_tmp,dV,count_colls,cck,m_low,eta_indize)



