#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 12:05:53 2019

@author: bohrer
"""

import matplotlib as mpl

#matplotlib.font_manager._rebuild()

import numpy as np
#import matplotlib
#matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
mpl.rcParams.update(plt.rcParamsDefault)
#from matplotlib import rc

### rc params for this script only
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)

#rc('text.latex.preamble', '\usepackage[ttscale=.9]{libertine}')
#   \usepackage[T1]{fontenc}
#   \usepackage[libertine]{newtxmath})

#t = np.linspace(0.0, 1.0, 100)
#s = np.cos(4 * np.pi * t) + 2
#import matplotlib
#sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])

#%%

from os.path import expanduser
fontpath = expanduser("~/Library/Fonts/LinLibertine_R.otf")

#print(fontpath)

import matplotlib.font_manager as font_manager

fontprop = font_manager.FontProperties(fname=fontpath)

#for font in font_manager.findSystemFonts():
#    print(font)

#font_manager._rebuild()

#%%


mpl.use("pgf")

pgf_with_lualatex = {
    "text.usetex": True,
    "pgf.rcfonts": False,   # Do not set up fonts from rc parameters.
    "pgf.texsystem": "lualatex",
    "pgf.preamble": [
        r'\usepackage{libertine}',
        r'\usepackage[libertine]{newtxmath}',
        r'\usepackage[no-math]{fontspec}',
        ]
}
mpl.rcParams.update(pgf_with_lualatex)

#%%

#plt.rcParams['backend'] = 'wxAgg'
#
#
####
#
#font_serif = 'Linux Libertine O'
##font_serif = 'DejaVu Serif'
##font_serif = "cmr10"
##plt.rcParams['font.serif'] = [font_serif]
#
#
##plt.rcParams['font.family'] = fontprop.get_name()
##plt.rcParams['font.family'] = 'serif'
#plt.rcParams['font.family'] = font_serif
#
####
##plt.rcParams['font.style'] = 'normal'
#
#plt.rcParams['text.usetex'] = True
#plt.rcParams['text.latex.preamble'] = \
#    r'\usepackage{lmodern} \usepackage[T1]{fontenc}' 
#    r'\usepackage{libertine} \usepackage[libertine]{newtxmath} \usepackage[T1]{fontenc}' 
#    r'\usepackage{libertine} \usepackage[libertine]{newtxmath} \usepackage[T1]{fontenc}' 
#    r'\usepackage{lmodern}' 


TTFS = 10
LFS = 10
TKFS = 8

LW = 1
MS = 2


#plt.rcParams['lines.linewidth'] = LW
#plt.rcParams['lines.markersize'] = MS

#TTFS = 16
#LFS = 12
#TKFS = 12

#plt.rcParams['axes.titlesize'] = TTFS
#plt.rcParams['axes.labelsize'] = LFS
#plt.rcParams['legend.fontsize'] = LFS
#plt.rcParams['xtick.labelsize'] = TKFS
#plt.rcParams['ytick.labelsize'] = TKFS

## alignemnt of ticks
#plt.rcParams['xtick.alignment'] = "center"
#plt.rcParams['ytick.alignment'] = "center"


#plt.rcParams['savefig.dpi'] = 600

par1 = 50.9876543
par2 = 0.123456789

# subs = "\\alpha, 7" # OR USE raw string (r)
subs = r"\alpha,7"

x = np.linspace(0,6.3,1000)
y = np.sin(x)

i = 6




fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

ax.plot(x,y, label = f"$v_{i}$")
ax.set_xlabel(f"$x_{{{subs}}}$ [$p_1={par1:06.2f}$, $p_2={par2:06.2f}$]")
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


fig.savefig("test_figs/plt_latex_pgf.pdf")
fig.savefig("test_figs/plt_latex_pgf.png")

#fig.savefig("test_figs/plt_latex_" + font_serif + ".png")
#fig.savefig("test_figs/plt_latex_" + font_serif + ".png")
#fig.savefig("test_figs/plt_latex_lmodern.png")
#plt.show()