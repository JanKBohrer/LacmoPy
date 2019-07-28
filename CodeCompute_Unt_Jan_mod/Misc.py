""" %    This function generates uniformly distributed pseudorandom numbers
    %    between 0 and 1, using the 32-bit generator from figure 3 of
    %    the article by L'Ecuyer.
    %
    %    The cycle length is claimed to be 2.30584E+18.
    %
    %  Licensing:
    %
    %    This code is distributed under the GNU LGPL license.
    %
    %  Modified:
    %
    %    08 July 2008
    %
    %  Author:
    %
    %    Original PASCAL version by Pierre L'Ecuyer.
    %    Modifications by John Burkardt.
    %
    %  Reference:
    %
    %    Pierre LEcuyer,
    %    Efficient and Portable Combined Random Number Generators,
    %    Communications of the ACM,
    %    Volume 31, Number 6, June 1988, pages 742-751.
    %
    %  Parameters:
    %
    %    Input, integer S1, S2, two values used as the
    %    seed for the sequence.  On first call, the user should initialize
    %    S1 to a value between 1 and 2147483562;  S2 should be initialized
    %    to a value between 1 and 2147483398.
    %
    %    Output, real R, the next value in the sequence.
    %
    %    Output, integer S1, S2, updated seed values.
"""
def rng(s1,s2):
    import math
    o=math.floor(s1/53668)
    s1 = 40014 * ( s1 - o * 53668 ) - o * 12211
    if(s1<0):
        s1=s1+2147483563
    o=math.floor(s2/52774)
    s2 = 40692 * ( s2 - o * 52774 ) - o * 3791
    if(s2<0):
        s2=s2+2147483399
    z=s1-s2
    if(z<1):
        z=z+2147483562
    r=z/2147483563.0
    return r,s1,s2

##############################################################################
"""
Mapping SIP to Bin
"""

#Mapping des SIP-Ensembles auf ein Bingitter
#SIP-Ensemble (nur eine Instanz) mit nr_sip Teilchen gegeben durch nEK_sip, mEK_sip
#Eigenschaften des Bingitters gegeben durch n10_plot, r10_plot, min10_plot




def MapSIPBin(nEK_sip,mEK_sip,nr_SIPs,n10,r10,min10):
    import numpy as np
    n= n10 * r10
    
    mfix=np.zeros(n)
    mdelta=np.zeros(n)
    mfix[0]=10**(min10)
    for i in range(1,n):
        mfix[i]=10**((i-1)/n10+min10)
        mdelta[i-1]=mfix[i]-mfix[i-1]
    if (mfix[-1] < max(mEK_sip[0:nr_SIPs])):
        print("Achtung. Gitter zu klein ",mfix[-1] , max(mEK_sip[0:nr_SIPs]))

    #Einordnen in das Bingitter
    z=1
    mEK_bin_tot=np.zeros(n)
    nEK_bin_ord=np.zeros(n)
    nEK_sip_tmp=nEK_sip[0:nr_SIPs]  #int(nr_SIPs) entfernt
    mEK_sip_tmp=mEK_sip[0:nr_SIPs]

    per=np.argsort(mEK_sip_tmp)
    #print(mfix[-1],max(mEK_sip_tmp))
    for i in range(0,nr_SIPs):
        while(mfix[z]<mEK_sip_tmp[per[i]]):
            z=z+1

        nEK_bin_ord[z-1]=nEK_bin_ord[z-1]+nEK_sip_tmp[per[i]]/mdelta[z-1]
        #Hier entweder Binmittelpunkte (bei log mus noch mfix[i]=10^((i-1+0.5)/n10+min10)) oder so aehnlich abgeaendert werden
        if(nEK_bin_ord[z-1]!=0):
            mEK_bin_tot[z-1]=mEK_bin_tot[z-1]+nEK_sip_tmp[per[i]]*mEK_sip_tmp[per[i]]
        else:
            mEK_bin_tot[z-1]=(mfix[z]-mfix[z-1]/2)+mfix[z-1]
            
    mEK_bin_ord=np.zeros(n)
    for i in range(0,n):
        if(nEK_bin_ord[i]!=0):
            mEK_bin_ord[i]=mEK_bin_tot[i]/(nEK_bin_ord[i]*mdelta[i])
    return nEK_bin_ord,mEK_bin_ord,mEK_bin_tot,mdelta,z

#end subroutine MapSIPBin(nEK_sip,mEK_sip,nr_SIPs,n10,r10,min10):

def CountSIPs(nEK_sip,mEK_sip,nr_SIPs,nplot):
    nEK_bin_plot=np.zeros(nplot)
    if (max(mEK_sip) > nplot):
        print('---------- Achtung: ', max(mEK_sip), nplot)
    for i in range(nr_SIPs):
        nEK_bin_plot[mEK_sip] += nEK_sip[mEK_sip]

    return nEK_bin_plot
#end subroutine CountSIPs(nEK_sip,mEK_sip,nr_SIPs,nplot)

def CIO_MOMmean(data_in=None,fp_in=None,fp_out=None):
    #CIO = "Compute or Input/Read mean Moments" and Output the data 
    import numpy as np
    #Berechne gemittelte Momente
    #Input: Momente aller Instanzen (entweder gegeben durch Array data_in oder Einlesen aus Datei in Ordner fp_in)
    #Output: gebe Array zurueck und schreibe zusätzlich in Datei wenn fp_out gegeben ist.
    
    if (fp_in is not None):
        fu = open(fp_in + 'Moments_meta.dat','r')
        nr_inst = int(fu.readline())
        nr_MOMsave = int(fu.readline())
        #t_vec_MOMsave= np.array(fu.readline().split(), dtype='float' )
        fu.close()
        fu = open(fp_in + 'Moments.dat','rb')
        data_in=np.loadtxt(fu).reshape((nr_inst,nr_MOMsave,4))
        fu.close()
        #print(data_in.shape)
    else:
        if (data_in is not None):
            [nr_inst,nt_MOMsave,nr_Mom]=data_in.shape
        else: 
            print('supply either data or fp_in')

    data_out=np.mean(data_in,axis=0)

    if (fp_out is not None):
        fu = open(fp_out + 'MomentsMean.dat','wb')
        data_in=np.savetxt(fu,data_out, fmt='%1.6e')

    return data_out

#def get_MOMmean(fp_in):
    
    

'''
Funktion zur Berechung der Momente einer Anzahlverteilung
'''

def Moments(nr_inst,nEK_sip_ins,mEK_sip_ins,nr_moment):
    import numpy as np
    moment_inst=np.array(np.zeros(nr_inst))

    for k in range(0,nr_inst):
        moment_inst[k]=sum(nEK_sip_ins[k,:]*(mEK_sip_ins[k,:]**nr_moment))
    moment_all=sum(moment_inst[:])/nr_inst

    return moment_all, moment_inst       

##############################################################################

def Moments_k0_3(nEK_sip_ins,mEK_sip_ins):
    import numpy as np
    moments_k=np.zeros(4)
    for k in range(0,4):
        moments_k[k]=sum(nEK_sip_ins*(mEK_sip_ins**k))

    return moments_k
##############################################################################
