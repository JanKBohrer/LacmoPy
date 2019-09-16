#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 12:05:53 2019

@author: bohrer
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
mpl.rcParams.update(plt.rcParamsDefault)

import numpy as np

from plotting import cm2inch
from plotting import generate_rcParams_dict

#matplotlib.font_manager._rebuild()
#import matplotlib
#matplotlib.rcParams['text.usetex'] = True
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
#from os.path import expanduser
# for Mac OS:
#fontpath = expanduser("~/Library/Fonts/LinLibertine_R.otf")
# for LinuxDesk
#fontpath = "/usr/share/fonts/opentype/linux-libertine/LinLibertine_R.otf"
#fontprop = font_manager.FontProperties(fname=fontpath)

#print(fontpath)
#for font in font_manager.findSystemFonts():
#    print(font)

#font_manager._rebuild()

#%%
mpl.use("pgf")

pgf_with_lualatex = {
    "text.usetex": True,
    "pgf.rcfonts": False,   # Do not set up fonts from rc parameters.
    "pgf.texsystem": "lualatex",
#    "pgf.texsystem": "pdflatex",
#    "pgf.texsystem": "xelatex",
    "pgf.preamble": [
            r'\PassOptionsToPackage{no-math}{fontspec}',
            r'\usepackage[ttscale=.9]{libertine}',
#            r'\usepackage[T1]{fontenc}',
            r'\usepackage[libertine]{newtxmath}',
#            r'\usepackage{unicode-math}',
#            r'\usepackage[]{mathspec}',
            r'\setmainfont{LinLibertine_R}',
            r'\setromanfont[]{LinLibertine_R}',
            r'\setsansfont[]{LinLibertine_R}',
            r'\DeclareSymbolFont{digits}{TU}{\sfdefault}{m}{n}',
            r'\DeclareMathSymbol{0}{\mathalpha}{digits}{`0}',
            r'\DeclareMathSymbol{1}{\mathalpha}{digits}{`1}',
            r'\DeclareMathSymbol{2}{\mathalpha}{digits}{`2}',
            r'\DeclareMathSymbol{3}{\mathalpha}{digits}{`3}',
            r'\DeclareMathSymbol{4}{\mathalpha}{digits}{`4}',
            r'\DeclareMathSymbol{5}{\mathalpha}{digits}{`5}',
            r'\DeclareMathSymbol{6}{\mathalpha}{digits}{`6}',
            r'\DeclareMathSymbol{7}{\mathalpha}{digits}{`7}',
            r'\DeclareMathSymbol{8}{\mathalpha}{digits}{`8}',
            r'\DeclareMathSymbol{9}{\mathalpha}{digits}{`9}'           
#            r'\setromanfont[Mapping=tex-text]{LinLibertine_R}',
#            r'\setsansfont[Mapping=tex-text]{LinLibertine_R}',
#            r'\setmathfont{LinLibertine_R}'
#            r'\setmathfont[range={"0031-"0040}]{LinLibertine_R}'
#            r'\setmathsfont(Digits){LinLibertine_R}'
#        r'\usepackage{libertine}',
#        r'\usepackage[libertine]{newtxmath}',
#        r'\usepackage[]{fontspec}',
#        r'\usepackage[no-math]{fontspec}',
        ]
}
mpl.rcParams.update(pgf_with_lualatex)

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

#%%

figname = "plt_latex_pgf2"

figsize=(10,10)
fig, ax = plt.subplots(figsize=( cm2inch(figsize) ), tight_layout=True)

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

fig.tight_layout()


fig.savefig(f"test_figs/{figname}.pdf")
fig.savefig(f"test_figs/{figname}.png")

#fig.savefig("test_figs/plt_latex_" + font_serif + ".png")
#fig.savefig("test_figs/plt_latex_" + font_serif + ".png")
#fig.savefig("test_figs/plt_latex_lmodern.png")
#plt.show()