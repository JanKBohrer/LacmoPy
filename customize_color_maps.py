#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 19:06:42 2019

@author: jdesk
"""

import numpy as np
#import palettable
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
#from matplotlib import cm
#cmap = palettable.colorbrewer.sequential.YlGn_9.mpl_colormap
cmap = plt.cm.rainbow
#cmap = cm.Reds

#colors1 = cm.Reds

#z = (np.random.random((10,10))*0.35+0.735)*1.e-7
#
#fig, ax = plt.subplots()
#plot = ax.contourf(z, levels=np.linspace(0.735e-7,1.145e-7,10), cmap=cmap)
#
##fmt = FormatScalarFormatter("%.2f")
#
#cbar = fig.colorbar(plot)
#
#colors1 = plt.cm.Reds(np.linspace(0, 0.1, 16))
##colors1 = plt.cm.Reds
#colors2 = plt.cm.rainbow(np.linspace(0, 1, 256))
## stacking the 2 arrays row-wise
#colors = np.vstack((colors1, colors2))
#cmap = mcolors.ListedColormap('colormap', colors)
##cmap = mcolors.LinearSegmentedColormap.from_list('colormap', colors)
#
#data = np.random.rand(10,10) * 100 
#
#
#plt.pcolor(data, cmap=cmap)
#plt.colorbar()

#cmap_dict = cmap._segmentdata

#newcmap = cmap.from_list('newcmap',list(map(cmap,range(50))), N=50)
#for x in range(80):
#    plt.bar(x,1, width=1, edgecolor='none',facecolor=newcmap(x))
#plt.show()

#%%

import matplotlib as mpl

# Setting the figure size 
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8, 5), gridspec_kw = {'width_ratios':[3, 0.1]})

# Initializing the data
num = 1000
x = np.linspace(-0.5,1,num) + (0.5 - np.random.rand(num))
y = np.linspace(-5,5,num) + (0.5 - np.random.rand(num))

# Concatenating colormaps
#bottom = plt.cm.get_cmap('Purples', 256)
colors1 = plt.cm.get_cmap('gist_ncar_r', 256)
#top = plt.cm.get_cmap('gist_rainbow_r', 256)
colors2 = plt.cm.get_cmap('rainbow', 256)
#top = plt.cm.get_cmap('Greys', 128)
#bottom = plt.cm.get_cmap('Blues', 128)

newcolors = np.vstack((colors1(np.linspace(0, 0.16, 16)),
                       colors2(np.linspace(0, 1, 256))))
#newcolors = np.vstack((top(np.linspace(0, 1, 128)),
#                       bottom(np.linspace(0, 1, 128))))
cmap = mpl.colors.ListedColormap(newcolors, name='my_rainbow')

# Colormap
ax1.scatter(x, y, c=x, cmap=cmap)
ax1.set_title('Concatenated colormaps', fontsize=16, weight='bold')

# Normalized Colorbar
norm = mpl.colors.Normalize(vmin=5, vmax=10)
mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm) 

# Displaying the figure
plt.show()