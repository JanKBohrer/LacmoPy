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

def compute_f_m0(m, DNC, LWC):
    k = DNC/LWC
    return DNC * k * np.exp(-k * m)

def plot_f_m_g_m_g_ln_R(DNC, LWC, m, R, xlim_m, xlim_R, figsize, figname):
    
    f_m0 = compute_f_m0(m, DNC, LWC)
    g_m0 = f_m0 * m
    g_ln_R = 3. * f_m0 * m * m
    
    f_m_max = f_m0.max()
    g_m_max = g_m0.max()
    g_R_max = g_ln_R.max()
    
    scale_f = 1E-9
    scale_gR = 1E9
    
    fig, ax = plt.subplots(figsize=( cm2inch(figsize) ), tight_layout=True)
    
    
    
#    ax.plot(m, f_m0 * scale_f)    
    ax.plot(m, f_m0 / f_m_max, c = "k")    
    ax.plot(m, g_m0 / g_m_max, "--", c = "0.3")    
#    ax.plot(m, g_ln_R)    
#    ax.set_xticks(np.arange(-10,41,10))
    ax.set_xlim(xlim_m)
#    ax.set_ylim((1E-6,1E12))
#    ax.set_ylim((3,1E2))
#    ax.set_xlabel("$T$ (K)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Droplet mass (kg)")
#    ax.set_ylabel("$f_m$ ($\mathrm{kg^{-1}\,m^{-3}}$)")
    ax.set_ylabel("Distributions (relative units)")
    
    ax_ = ax.twiny()
#    ax2 = ax_.twiny()
    ax_.plot(R, g_ln_R / g_R_max, ":", c = "0.3")
#    ax_.plot(R, g_ln_R*scale_gR)
    ax_.set_xscale("log")
    ax_.set_xlim(xlim_R)
    ax_.set_xlabel(r"Droplet radius ($\si{\micro\meter}$)")
#    ax_ = ax.twinx()
#    ax2 = ax_.twiny()
#    ax2.plot(R, g_ln_R*scale_gR)
#    ax2.set_xscale("log")
#    ax2.set_yscale("log")
#    ax.grid(which = "both", linestyle="--", linewidth=0.5, c = "0.5", zorder = 0)
    fig.savefig(figname,
#                bbox_inches = 0,
                bbox_inches = 'tight',
                pad_inches = 0.065
                )
    
    
#%%
import math
LWC0 = 1.0E-3 # kg/m^3
R_mean = 9.3 # in mu
mass_density = 1E3 # approx for water
#mass_density = c.mass_density_water_liquid_NTP
#mass_density = c.mass_density_NaCl_dry
c_radius_to_mass = 4.0E-18 * math.pi * mass_density / 3.0
c_mass_to_radius = 1.0 / c_radius_to_mass
m_mean = c_radius_to_mass * R_mean**3 # in kg
DNC0 = LWC0 / m_mean # in 1/m^3

figname = "/home/jdesk/Masterthesis/Figures/02Theory/initDistExpo.pdf"
figsize = (7.6,6)

min_log = -1
max_log = np.log10(30)
#min_log = -15
#max_log = -10
R = np.logspace(min_log,max_log)
#R = (m*c_mass_to_radius)**(1./3.)
m = c_radius_to_mass * R**3
#m = np.logspace(min_log,max_log)
#R = (m*c_mass_to_radius)**(1./3.)

xlim_m = (m[0], m[-1])
xlim_R = (R[0], R[-1])
plot_f_m_g_m_g_ln_R(DNC0, LWC0, m, R, xlim_m, xlim_R, figsize, figname)
