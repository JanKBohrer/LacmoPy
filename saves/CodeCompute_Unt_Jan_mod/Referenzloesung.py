

















#Beginn Golovin-Kernel
#Ende Golovin-Kernel



'''
Referenzloesung Wang fuer Long-Kernel
'''

def read_GVdata_Wang(kn_name):
# kn_name entweder 'Hall' oder 'Long'
    import numpy as np
    # 572 ist die maximale Binanzahl der eingelesenen Daten
    g_wang=np.zeros([7,572])
    r_wang=np.zeros([7,572])
    nr_wang=np.zeros(7,dtype='int')

    # Spalte 1: r ist gegeben in mm
    # Spalte 5: g(ln(r)) ist gegeben in g/m^3

    fn_part_vec = ("00","10","20","30","40","50","60")
    nr_time=7
    fp = "/athome/unte_si/_data/Themen/Aggregation/Algorithmen 0D Box Model/Wang-Algorithm/Reference_OutputData/"
    for i_time,fn_part in enumerate(fn_part_vec):
        fn="GQ_" +fn_part + "_"+ kn_name +"_s16.dat"
       # print('read file: ', fn)
        read_tmp=np.genfromtxt(fp+fn)
        nr_bins, nr_col = read_tmp.shape
        print('Wang ', kn_name, ', nr_bins, nr_col: ', nr_bins, nr_col, fn)
        g_wang[i_time,:nr_bins]=read_tmp[:,4]
        r_wang[i_time,:nr_bins]=read_tmp[:,0]
        nr_wang[i_time] = nr_bins

    print('nr_wang: ', nr_wang)
    return r_wang,g_wang,nr_wang

#--------------------------------------------------------------------------------------------

#Beginn Long-Kernel


def PlotGV_0D(ZPit_vec,plot_Anfangsverteilung):
    import numpy as np
    import matplotlib.pyplot as plt
    #Einlesen der Referenzloesung
    r_wang,g_wang,nr_wang = read_GVdata_Wang('Long')
    #Plotten der Referenzloesung
    if(plot_Anfangsverteilung==1):
        plt.plot(r_wang[0,0:222]*1e3,g_wang[0,0:222],"k--")
    for i in range(0,len(ZPit_vec)):
        #Zum Plotten wird der Radius in um umgerechnet
        if abs(int(ZPit_vec[i]/600) - ZPit_vec[i]/600) < 0.001:
            iii=int(ZPit_vec[i]/600)
            #print('plotte Zeitschritt',ZPit_vec[i],iii)
            plt.plot(r_wang[iii,0:nr_wang[iii]]*1e3,g_wang[iii,0:nr_wang[iii]],"--", color='k', label='Wang')

#def PlotMoments_OD(ZPit_vec):
#    #Plotten der Momente
#    
#    masse_wang=np.zeros((r_wang.shape))
#    n_wang=np.zeros((r_wang.shape))
#    for i in range(0,7):
#        for j in range(0,nr_wang[i]):
#            masse_wang[i,j]=(4/3)*(1e3)*np.pi*(r_wang[i,j])**3 #Umrechung der Radien in Masse
#            n_wang[i,j]=(g_wang[i,j]*1e-3)/(3*(masse_wang[i,j]**2))
#
#    #Plotten des 0ten Momentes der Referenzloesung von Wang
#    moment_wang=np.zeros(len(ZPit_vec))
#    for i in range(0,len(ZPit_vec)):
#        moment_wang[i]=sum(n_wang[int(ZPit_vec[i]/600),:]*masse_wang[int(ZPit_vec[i]/600),:])
#    plt.plot([0,600,1200,1800,2400,3000,3600],moment_wang,'o', color='black')
#

def defineMoments_0D():
    import numpy as np
    #verwende tabellierte Werte aus Wang-Paper
    nt_ref=7
    WangMomente=np.zeros([nt_ref,4])
    timeWang=np.arange(nt_ref)*10 # in Minuten
    WangMomente[:,0]=np.array([295.4e6     , 287.4e6     , 278.4e6     ,  264.4e6       , 151.7e6, 13.41e6, 1.212e6])  # ; Anzahl in m^-3
    WangMomente[:,1]=np.array([0.999989e-3 , 0.999989e-3 , 0.999989e-3 ,  0.999989e-3   ,0.999989e-3, 0.999989e-3, 0.999989e-3])/0.999989e-3 #; Masse
    WangMomente[:,2]=np.array([6.739e-15   , 7.402e-15   , 8.72e-15,      3.132e-13      , 3.498e-10, 1.068e-8,3.199e-8])# ; Moment 2 in kg^2/m^3
    WangMomente[:,3]=np.array([6.813e-26   , 9.305e-26   ,5.71e-25,       3.967e-20       , 1.048e-15, 2.542e-13, 1.731e-12] )#; ;Moment 3 in kg^3/m^3
    return timeWang, WangMomente

#Ende Long-Kernel

#--------------------------------------------------------------------------------------------

#Beginn Hall-Kernel
#Ende Hall-Kernel

#------------------------------------------------------------------------------------------
