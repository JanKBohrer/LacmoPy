#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 11:21:06 2019

@author: jdesk
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
plt.rcParams.update(plt.rcParamsDefault)


# plt.style.use('ggplot')
# plt.style.use('seaborn')
# plt.style.use('seaborn-paper')
# plt.style.use('seaborn-dark')
# plt.style.use('Solarize_Light2')

# print(plt.style.available)

par1 = 50.9876543
par2 = 0.123456789

# subs = "\\alpha, 7" # OR USE raw string (r)
subs = r"\alpha,7"

x = np.linspace(0,6.3,1000)
y = np.sin(x)

i = 6

fig, axes = plt.subplots(2, figsize=(6,6))
ax = axes[0]
ax.plot(x,y, label = f"$v_{i}$")
ax.legend()
# with raw string (r) and .format()
ax.set_xlabel(r"$x_{{{}}}$ [p1={:06.2f}, p2={:06.2f}]".format(subs, par1, par2))

ax = axes[1]
ax.plot(x,y)
# with f-string
ax.set_xlabel(f"$x_{{{subs}}}$ [p1={par1:06.2f}, p2={par2:06.2f}]")
# ax.set_xlabel(r"$x_{{m,2}}$ [p1={}, p2={}]".format(par1, par2))

fig.tight_layout()

###

font = FontProperties()
font.set_family('sans-serif')
# font.set_name('Times New Roman')
font.set_name('cmss10')
# font.set_style('italic')
font.set_style('normal')

params = {
    'backend': 'wxAgg',
    # 'font.sans-serif' :  "Comic Sans MS",
    # 'font.sans-serif' :  "Latin Modern Roman",
    'font.sans-serif' :  'CMU Sans Serif',
    # 'font.sans-serif' :  "cm",
    # 'font.sans-serif' :  "computer modern roman",
    # 'font.sans-serif'  :  'Latin Modern Roman',
    # 'font.serif' :  'CMU Serif',
    # 'font.serif' :  'cmr10',
    # 'font.sans-serif' :  'Latin Modern Math',
    # 'font.serif' :  'Latin Modern Math',
    'font.serif'  :  'CMU Serif',
    # 'font.serif'  :  'CMU Serif Upright Italic',
    # 'font.serif'  :  'CMU Serif Extra',
    # 'font.serif'  :  'CMU Classical Serif',
    # 'font.serif'  :  'Latin Modern Roman',
    'font.family'  :  "serif",
    # 'font.family'  :  'Latin Modern Roman',
    # 'font.family'  :  "CMU Serif",
    # 'font.family'  :  "CMU Concrete",
    # 'font.family'  :  "CMU Bright",
    # 'font.family'  :  'CMU Classical Serif',
    # 'font.family'  :  'cmr10',
    # 'font.family'  :  "sans-serif",
    # "font.style"  :  "italic",
    "font.style"  :  "normal",
    'mathtext.fontset'  :  "cm"
    # 'mathtext.default'  :  "sans:italic"
    # 'mathtext.default'  :  "sf"
}
    # 'lines.markersize' : 2,
    # 'axes.labelsize': fontlabel_size,
    # 'text.fontsize': fontlabel_size,
    # 'legend.fontsize': fontlabel_size,
    # 'xtick.labelsize': tick_size,
    # 'ytick.labelsize': tick_size,
    # 'text.usetex': True,
    # 'figure.figsize': fig_size
plt.rcParams.update(params)

#mathtext.fontset : dejavusans # Should be 'dejavusans' (default),
                               # 'dejavuserif', 'cm' (Computer Modern), 'stix',
                               # 'stixsans' or 'custom'

no_rows = 2
fig, axes = plt.subplots(no_rows, figsize=(8,6))
ax = axes[0]
ax.plot(x,y, label = f"$v_{i}$")
ax.set_xlabel(f"$x_{{{subs}}}$ [$p_1={par1:06.2f}$, $p_2=${par2:06.2f}]")
# ax.set_title("this is normal text 01234567890  $this is normal text 01234567890$")
ax.set_title("abcdefghijklmnopqrstuvwxyz 01234567890 \n"
             + "$abcdefghijklmnopqrstuvwxyz$ $01234567890$")
ax = axes[1]
ax.plot(x,y, label = f"$v_{i}$")
# ax.set_xlabel(f"$x_{{{subs}}}$ [p1={par1:06.2f}, p2={par2:06.2f}]",
#               fontproperties=font)
ax.set_xlabel(f"$x_{{{subs}}}$ [p1={par1:06.2f}, p2={par2:06.2f}]")
ax.set_title(f"this is normal text 01234567890  $this is normal text 01234567890$ (f-string)")

fig.tight_layout()
fig.savefig("./test_figs/cmu_font1.png")
fig.savefig("./test_figs/cmu_font1.pdf")

# import matplotlib.font_manager
# flist = matplotlib.font_manager.get_fontconfig_fonts()
# names = [matplotlib.font_manager.FontProperties(fname=fname).get_name() for fname in flist]
# print (names)