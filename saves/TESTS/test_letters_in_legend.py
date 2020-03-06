import os
import numpy as np
# import os
# from datetime import datetime
# import timeit

import constants as c
from microphysics import compute_R_p_w_s_rho_p_AS
from microphysics import compute_R_p_w_s_rho_p_NaCl
from microphysics import compute_radius_from_mass_vec
from file_handling import load_grid_and_particles_full,\
                          load_grid_scalar_fields\
                          

#                     plot_particle_size_spectra
# from integration import compute_dt_max_from_CFL
#from grid import compute_no_grid_cells_from_step_sizes

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
mpl.rcParams.update(plt.rcParamsDefault)
#mpl.use("pdf")
mpl.use("pgf")

import matplotlib.ticker as mticker


#import numpy as np

from microphysics import compute_mass_from_radius_vec
import constants as c

#import numba as 

from plotting import cm2inch
from plotting import generate_rcParams_dict
from plotting import pgf_dict, pdf_dict
plt.rcParams.update(pgf_dict)
#plt.rcParams.update(pdf_dict)

from file_handling import load_grid_and_particles_full,\
                          load_grid_scalar_fields\

from analysis import sample_masses, sample_radii
from analysis import sample_masses_per_m_dry , sample_radii_per_m_dry
from analysis import plot_size_spectra_R_Arabas, generate_size_spectra_R_Arabas

from plotting_fcts_MA import plot_size_spectra_R_Arabas_MA

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
LW = 1.2
MS = 2

# raster resolution for e.g. .png
DPI = 600

mpl.rcParams.update(generate_rcParams_dict(LW, MS, TTFS, LFS, TKFS, DPI))



#%%
import matplotlib.patches as mpatches

def cpatch(x0,y0,width,height):
    return mpatches.Rectangle([x0, y0], width, height, facecolor=None,
                                   edgecolor=None, hatch='xx', lw=3,
                                   )
annotations = ["A", "B", "C", "D", "E", "F",
           "G", "H", "I", "J", "K", "L",
           "M", "N", "O", "P", "Q", "R"]
x = np.arange(5)
t = np.linspace(0,120,5)
fig, ax = plt.subplots()

patches = []
labels = []

textbox = []

ax.plot(x,x,"o")
for i,x_ in enumerate(x):
    ax.annotate(annotations[i],(x_,x_))
    patches.append(cpatch(x_,x_,1,1))
    labels.append(r"{}".format(t[i]))
    
    textbox.append(f"\\textbf{{{annotations[i]}}}: {t[i]}")
#    textbox += f"\\textbf{{{annotations[i]}}}: {t[i]} "
textbox = ", ".join(textbox)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.5,0, textbox, bbox=props)
#ax.legend(patches, labels)    
fig.savefig("test_letters.pdf")
#plt.show()




