#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 11:09:37 2019

@author: jdesk
"""

import matplotlib as mpl
mpl.use('svg')
import matplotlib.pyplot as plt
import numpy as np

from plotting import cm2inch
from plotting import generate_rcParams_dict

new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)

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
LW = 2
MS = 2

# raster resolution for e.g. .png
DPI = 600

mpl.rcParams.update(generate_rcParams_dict(LW, MS, TTFS, LFS, TKFS, DPI))

#%%

par1 = 50.9876543
par2 = 0.123456789

# subs = "\\alpha, 7" # OR USE raw string (r)
subs = r"\alpha,7"

x = np.linspace(0,6.3,1000)
y = np.sin(x)

i = 6

# figsize in cm (x,y)
fig_size = (7.5,5.0)
fig, ax = plt.subplots(figsize=cm2inch(fig_size))

ax.plot(x,y, label = f"$v_{{i}}$")
#ax.set_xlabel(f"$x$")
ax.set_xlabel(f"$x_{{{subs}}}$ [$p_1={par1:06.2f}$, $p_2={par2:6.5g}$]")
ax.set_ylabel(f"$y$")
ax.set_title("abcdefghijklmnopqrstuvwxyz 01234567890 \n"
             + "\$abcdefghijklmnopqrstuvwxyz\$ \$01234567890\$")

ax.grid()
ax.legend()

#fig.tight_layout()


### REMOVE WHITE MARGINS
#gca().set_axis_off()
#subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
#            hspace = 0, wspace = 0)
#margins(0,0)
#gca().xaxis.set_major_locator(NullLocator())
#gca().yaxis.set_major_locator(NullLocator())
#savefig("filename.pdf", bbox_inches = 'tight',
#    pad_inches = 0)

#fig.savefig("test_figs/plt_to_inkscape_01.svg", format = 'svg')
fig_name = "pltToInkscape03.svg"
fig.savefig("/home/jdesk/Masterthesis/Figures/test/" + fig_name,
            format = 'svg')

#### REMOVE WHITE MARGINS -> makes figure smaller...
##fig.savefig("test_figs/plt_latex_pgf.pdf",bbox_inches = 'tight')
#fig.savefig("test_figs/plt_latex_pgf.pdf", 
#            bbox_inches = 'tight', pad_inches = 0)
#
#fig.savefig("test_figs/plt_latex_pgf.png")