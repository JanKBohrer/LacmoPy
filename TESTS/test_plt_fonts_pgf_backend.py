#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 15:21:48 2019

@author: bohrer
"""

import numpy as np


import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)
# need to set mpl.use() explicitely (not via rcParams)
mpl.use("pgf")
# this does not work:...:
#plt.rcParams['backend'] = 'pgf'

from plotting import cm2inch
from plotting import generate_rcParams_dict
from plotting import pgf_dict
plt.rcParams.update(pgf_dict)

#%% SET DEFAULT PLOT PARAMETERS (can be changed for specific elements directly)
# TITLE, LABEL (and legend), TICKLABEL FONTSIZES
TTFS = 10
LFS = 10
TKFS = 8

#TTFS = 16
#LFS = 12
#TKFS = 12

# LINEWIDTH, MARKERSIZE
LW = 2
MS = 2

# raster resolution for e.g. .png
DPI = 600

plt.rcParams.update(generate_rcParams_dict(LW, MS, TTFS, LFS, TKFS, DPI))

#%%

par1 = 50.9876543
par2 = 0.123456789

# subs = "\\alpha, 7" # OR USE raw string (r)
subs = r"\alpha,7"

x = np.linspace(0,10.,1000)
y = np.sin(x)

i = 6

# figsize in cm (x,y)
fig_size = (7.5,5.0)
fig, ax = plt.subplots(figsize=cm2inch(fig_size))

ax.plot(x,y, label = f"$v_{i}$")
ax.set_xlabel(f"$x_{{{subs}}}$ [$p_1={par1:06.2f}$, $p_2={par2:6.5g}$]")
ax.set_ylabel(f"$y_{{{subs}}}$")
ax.set_title("abcdefghijklmnopqrstuvwxyz 01234567890 \n"
             + "$abcdefghijklmnopqrstuvwxyz$ $01234567890$")

ax.set_xticks(np.arange(1,11))

ax.grid()
ax.legend()

fig.tight_layout()


### REMOVE WHITE MARGINS
#gca().set_axis_off()
#subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
#            hspace = 0, wspace = 0)
#margins(0,0)
#gca().xaxis.set_major_locator(NullLocator())
#gca().yaxis.set_major_locator(NullLocator())
#savefig("filename.pdf", bbox_inches = 'tight',
#    pad_inches = 0)

fig.savefig("test_figs/plt_latex_pgf9.pgf")

### REMOVE WHITE MARGINS -> makes figure smaller...
#fig.savefig("test_figs/plt_latex_pgf.pdf",bbox_inches = 'tight')
fig.savefig("test_figs/plt_latex_pgf9.pdf", 
            bbox_inches = 'tight', pad_inches = 0)

fig.savefig("test_figs/plt_latex_pgf9.png")
