import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
mpl.rcParams.update(plt.rcParamsDefault)
#mpl.use("pdf")
mpl.use("pgf")

import numpy as np

from plotting import cm2inch
from plotting import generate_rcParams_dict
from plotting import pgf_dict, pdf_dict
plt.rcParams.update(pgf_dict)
#plt.rcParams.update(pdf_dict)

from file_handling import load_grid_and_particles_full,\
                          load_grid_scalar_fields\
#import numpy as np

#%% SET DEFAULT PLOT PARAMETERS
# (can be changed lateron for specific elements directly)
# TITLE, LABEL (and legend), TICKLABEL FONTSIZES
TTFS = 10
LFS = 10
TKFS = 8

#TTFS = 16
#LFS = 12
#TKFS = 12

# LINEWIDTH, MARKERSIZE
LW = 1.5
MS = 2

# raster resolution for e.g. .png
DPI = 600

mpl.rcParams.update(generate_rcParams_dict(LW, MS, TTFS, LFS, TKFS, DPI))

#%%
def plot_phase_diagram(T_vl, e_vl, T_vi, e_vi, T_sl, e_sl, figname, figsize):
    
    fig, ax = plt.subplots(figsize=( cm2inch(figsize) ), tight_layout=True)
    
    T_shift = -273.15
    ax.plot(T_vl + T_shift, e_vl, c = "k")
    ax.plot(T_vi + T_shift, e_vi, "--", c = "k")
    ax.plot(T_sl + T_shift, e_sl, "-.", c = "k")
#    ax.annotate("triple point", (T_vi[-1] + T_shift, e_vi[-1]-0.5),
##                ha='left',
#                ha='center',
#                va='top',
#                xycoords='data',
#                arrowprops=dict(facecolor='black',
#                                width = 0.5,
#                                headwidth = 2,
#                                shrink=0.05
#                                ),
#                textcoords='data',
#                xytext=(T_vi[-1] + T_shift, 2)
#                )
    ax.annotate("vapor", (20,6), ha='center')
    ax.annotate("solid", (-5,15), ha='center')
    ax.annotate("liquid", (12,30), ha='center')
    ax.annotate(r"$e_s(T)$", (22,16), ha='center')
#    ax.annotate("vapor")
    ax.set_yscale("log")
    ax.set_xticks(np.arange(-10,41,10))
    ax.set_xlim((-10,40))
    ax.set_ylim((3,1E2))
#    ax.set_xlabel("$T$ (K)")
    ax.set_xlabel("$T$ (°C)")
    ax.set_ylabel("$p$ (hPa)")
#    ax.grid(which = "both", linestyle="--", linewidth=0.5, c = "0.5", zorder = 0)
    fig.savefig(figname,
#                bbox_inches = 0,
                bbox_inches = 'tight',
                pad_inches = 0.065
                )
# formula from
# http://www1.lsbu.ac.uk/water/water_phase_diagram.html
# T in K
# p in MPa
def p_melt(T):
    return -395.2 * ( ( T/273.16 )**9. - 1. )

#%%

#saturation pressure gas - water (vapor-liquid)
# values from saturation toolbox -> reset with Lohmann values
path = "/home/jdesk/Masterthesis/saturation_pressure_values.txt"
data = np.loadtxt(path)
cutoff = 12
T = data[:cutoff,0] + 273.15
# in hPa
e_vl = data[:cutoff,1] * 10

# vapor - ice
path = "/home/jdesk/Masterthesis/saturation_pressure_values_ice_vapor.txt"
data = np.loadtxt(path)
cutoff = 3
T_vi = data[:cutoff][::-1] + 273.15
# in hPa
e_vi = data[12:12+cutoff][::-1] / 100

# solid-liquid
#e_sl = np.array((e_vi[-1], e_vi[-1]))
T_sl = np.linspace(0.0098,0.009999,10) + 273.15
e_sl = p_melt(T_sl) * 1E6

T_sl = np.append(T_sl, (273.16))
e_sl = np.append(e_sl, e_vi[-1])

print(T)
print(e_vl)
print(T_vi)
print(e_vi)

figname = "/home/jdesk/Masterthesis/Figures/02Theory/phaseDiagram.pdf"
figsize = (7.6,6)
plot_phase_diagram(T, e_vl, T_vi, e_vi, T_sl, e_sl, figname, figsize)





