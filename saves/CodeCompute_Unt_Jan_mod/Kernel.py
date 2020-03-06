

















'''
Long Kernel
'''
import numpy as np
import math
import sys
import Misc as FK



'''
Berechnung der Fallgeschwindigkeiten nach Beard fuer Long- bzw Hall-Kernel
'''

def Fallg(r,n):
    b=np.array([-0.318657,0.992696,-0.153193*1e-2,-0.987059*1e-3,-0.578878*1e-3,0.855176*1e-4,-0.327815*1e-5])
    c=np.array([-0.500015,0.523778,-0.204914,0.475294,-0.542819*1e-1,0.238449*1e-2])
    eta=1.818*1e-4
    xlamb=6.62*1e-6
    # mod
    rhow=1.
    rhoa=1.223*1e-3
    grav=980.665
    cunh=1.257*xlamb
    t0=273.15
    sigma=76.1-0.155*(293.15-t0)
    stok=2*grav*(rhow-rhoa)/(9*eta)
    stb=32*rhoa*(rhow-rhoa)*grav/(3*eta*eta)
    phy=sigma*sigma*sigma*rhoa*rhoa/((eta**4)*grav*(rhow-rhoa))
    py=phy**(1/6)

    rr=np.zeros(n)
    winf=np.zeros(n)

    for j in range(0,n):
        rr[j]=r[j]*1e-4
    for j in range(0,n):
        if(rr[j]<=1e-3):
            winf[j]=stok*(rr[j]*rr[j]+cunh*rr[j])
        elif( rr[j] > 1.0e-3 and rr[j] <= 5.35e-2):
            x=math.log(stb*rr[j]*rr[j]*rr[j])
            y=0.0
            # mod
            for i in range(7):
                # mod: ERROR: x**() instead of y**()
                y=y+b[i]*(x**i)
            xrey=(1.+cunh/rr[j])*math.exp(y)
            winf[j]=xrey*eta/(2.*rhoa*rr[j])
        elif(rr[j] > 5.35e-2):
            bond=grav*(rhow-rhoa)*rr[j]*rr[j]/sigma
            if (rr[j]>0.35):
                bond=grav*(rhow-rhoa)*0.35*0.35/sigma
            x=math.log(16.*bond*py/3.)
            y=0.0
            for i in range(6):
                # mod
                y=y+c[i]*(x**i)
            xrey=py*math.exp(y)
            winf[j]=xrey*eta/(2.*rhoa*rr[j])
            if(rr[j]>0.35):
                winf[j]=xrey*eta/(2.*rhoa*0.35)
    return winf,rr

# mod Jan:
def Fallg2(r):
    b = np.array([-0.318657e1, 0.992696, -0.153193e-2,
         -0.987059e-3, -0.578878e-3, 0.855176e-4, -0.327815e-5])
#    b = b[::-1]
    c = np.array([-0.500015e1, 0.523778e1, -0.204914e1,
         0.475294, -0.542819e-1 ,0.238449e-2])
#    c = c[::-1]
#    pi = 3.141592654
    eta=1.818e-4
    # = l0
    xlamb=6.62e-6
    rhow=1.
    rhoa=1.225e-3
    grav=980.665
    # cc = 1.257 or 1.255
    # Bott Fortran is 1.257, but 1.255 is from paper...
    cunh=1.257*xlamb
    t0=273.15
#    sigma=10.0*(76.1-0.155*(293.15-t0))
    sigma=76.1-0.155*(293.15-t0)
    stok=2.*grav*(rhow-rhoa)/(9.*eta)
    stb=32.*rhoa*(rhow-rhoa)*grav/(3.*eta*eta)
    phy=sigma*sigma*sigma*rhoa*rhoa/((eta**4)*grav*(rhow-rhoa))
    py=phy**(1./6.)
    
    rr = r*1.0E-4
    n = len(r)
    
    winf = np.zeros_like(r)
    for j in range(n):
        if rr[j] <= 1.0E-3:
            winf[j] = stok * (rr[j] * rr[j] + cunh * rr[j])
        elif rr[j] <= 5.35E-2:
            x = math.log(stb *rr[j]*rr[j]*rr[j])
#            print(x)
            y = b[0]
#            do i=1,7
            for i in range(1,7):
               y = y + b[i]*(x**i)
            xrey = ( 1. + cunh / rr[j] ) * math.exp(y)
            winf[j] = xrey * eta / ( 2. * rhoa * rr[j] )
        else:
#        elif rr[j] > 5.35E-2:
            bond = grav * (rhow - rhoa) * rr[j]*rr[j] / sigma
            if rr[j] > 0.35:
                bond = grav * (rhow - rhoa) * 0.35 * 0.35 / sigma
            x = math.log(16. * bond * py / 3.)
            y = c[0]
#            do i=1,6
            for i in range(1,6):
               y = y + c[i]*(x**i)
            xrey = py * math.exp(y)
            winf[j] = xrey * eta / (2. * rhoa * rr[j])
            if rr[j] > 0.35:
                winf[j] = xrey * eta / (2. * rhoa * 0.35)
    
    return winf
    
    

##############################################################################




def LongKernel(rho_w):
    #rho_w      IN: density of water in kg/m^3
    #cck       OUT: kernel values in m^3/s
    #mass_grid OUT: mass values in kg

    #fp="/athome/lerc_mi/Documents/Werte_Long_Kernel/"
#    fp="/athome/unte_si/_data/Themen/Aggregation/Algorithmen 0D Box Model/Bott-Algorithm/Data Hall Kernel Long Kernel VBeard/Long_Kernel_fein/"
    #fp="/athome/unte_si/_data/Themen/Aggregation/Algorithmen 0D Box Model/Bott-Algorithm/Data Hall Kernel Long Kernel VBeard/Long_Kernel/"
    # Linux Desk
#    fp = "/home/jdesk/CloudMP/CodeCompute_Unt/data/"
    fp = "data_Long_kernel/"
    
    #Einlesen des Radius in Einheit um
    file=np.loadtxt(fp+"Long_radius.txt")
    #print(file.shape)
    n=400
    radius_grid=np.zeros(400)
    radius_grid[0:400:3]=file[0:134,0]
    radius_grid[1:400:3]=file[0:133,1]
    radius_grid[2:400:3]=file[0:133,2]
    print('kernel values given for radius range: min ', radius_grid[0], ' max ', radius_grid[399])
    #Berechnung der Masse
    mass_grid=radius_grid**3*1.3333*math.pi*rho_w*1e-18


    #Einlesen der Werte
    file_cck=np.loadtxt(fp+"Values_Longkernel.txt") # given in cm^3/s
    nxn=file_cck.size-2
    file_cck=file_cck.flatten()[0:nxn]
    cck=np.reshape(file_cck,[n,n])

    return cck*1e-6, mass_grid, radius_grid



