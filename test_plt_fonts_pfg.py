#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 15:21:48 2019

@author: bohrer
"""

import numpy as np

#plt.rcParams['backend'] = 'pgf'

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)
mpl.use("pgf")

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
LW = 1
MS = 2

# raster resolution for e.g. .png
DPI = 600

plt.rcParams.update(generate_rcParams_dict(LW, MS, TTFS, LFS, TKFS, DPI))

#%%

par1 = 50.9876543
par2 = 0.123456789

# subs = "\\alpha, 7" # OR USE raw string (r)
subs = r"\alpha,7"

x = np.linspace(0,6.3,1000)
y = np.sin(x)

i = 6

fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

ax.plot(x,y, label = f"$v_{i}$")
ax.set_xlabel(f"$x_{{{subs}}}$ [$p_1={par1:06.2f}$, $p_2={par2:6.5g}$]")
ax.set_ylabel(f"$y_{{{subs}}}$")
# ax.set_title("this is normal text 01234567890  $this is normal text 01234567890$")
ax.set_title("abcdefghijklmnopqrstuvwxyz 01234567890 \n"
             + "$abcdefghijklmnopqrstuvwxyz$ $01234567890$")

#{'center', 'top', 'bottom', 'baseline', 'center_baseline'}
#center_baseline seems to be def, center is OK, little bit too far down but OK
#for label in ax.yaxis.get_ticklabels():
##    label.set_verticalalignment('baseline')
#    label.set_verticalalignment('center')


ax.grid()
ax.legend()

#ax.plot(t, s)
#
#
#ax.set_xlabel(r'\textbf{time (s)}')
#ax.set_ylabel('\\textit{Velocity (\N{DEGREE SIGN}/sec)}', fontsize=16)
#ax.set_title(r'\TeX\ is Number $\displaystyle\sum_{n=1}^\infty'
#             r'\frac{-e^{i\pi}}{2^n}$!', fontsize=16, color='r')
#plt.show()

fig.tight_layout()


fig.savefig("test_figs/plt_latex_pgf5.pdf")
fig.savefig("test_figs/plt_latex_pgf5.png")
