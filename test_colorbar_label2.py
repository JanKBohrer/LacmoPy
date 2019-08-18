#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 18:12:43 2019

@author: jdesk
"""

#import numpy as np; np.random.seed(0)
#import matplotlib.pyplot as plt
#import matplotlib.ticker
#
#class OOMFormatter(matplotlib.ticker.ScalarFormatter):
#    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
#        self.oom = order
#        self.fformat = fformat
#        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
#    def _set_orderOfMagnitude(self, nothing):
#        self.orderOfMagnitude = self.oom
#    def _set_format(self):
##    def _set_format(self, vmin, vmax):
#        self.format = self.fformat
#        if self._useMathText:
#            self.format = '$%s$' % matplotlib.ticker._mathdefault(self.format)
#
#
#z = (np.random.random((10,10)) - 0.5) * 0.2
#
#fig, ax = plt.subplots()
#plot = ax.contourf(z)
#cbar = fig.colorbar(plot, format=OOMFormatter(-2, mathText=False))
#
#plt.show()


#from matplotlib import colors
#import matplotlib.pyplot as plt
#import numpy as np
#
#np.random.seed(19680801)
#Nr = 3
#Nc = 2
#cmap = "cool"
#
#fig, axs = plt.subplots(Nr, Nc)
#fig.suptitle('Multiple images')
#
#images = []
#for i in range(Nr):
#    for j in range(Nc):
#        # Generate data with a range that varies from one plot to the next.
#        data = ((1 + i + j) / 10) * np.random.rand(10, 20) * 1e-6
#        images.append(axs[i, j].imshow(data, cmap=cmap))
#        axs[i, j].label_outer()
#
## Find the min and max of all colors for use in setting the color scale.
#vmin = min(image.get_array().min() for image in images)
#vmax = max(image.get_array().max() for image in images)
#norm = colors.Normalize(vmin=vmin, vmax=vmax)
#for im in images:
#    im.set_norm(norm)
#
#fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1)
#
#
## Make images respond to changes in the norm of other images (e.g. via the
## "edit axis, curves and images parameters" GUI on Qt), but be careful not to
## recurse infinitely!
#def update(changed_image):
#    for im in images:
#        if (changed_image.get_cmap() != im.get_cmap()
#                or changed_image.get_clim() != im.get_clim()):
#            im.set_cmap(changed_image.get_cmap())
#            im.set_clim(changed_image.get_clim())
#
#
#for im in images:
#    im.callbacksSM.connect('changed', update)


import numpy as np; np.random.seed(0)
import matplotlib.pyplot as plt
import matplotlib.ticker

class FormatScalarFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, fformat="%1.1f", offset=True, mathText=True):
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,
                                                        useMathText=mathText)
#    def _set_format(self, vmin, vmax):
    def _set_format(self):
        self.format = self.fformat
        if self._useMathText:
            self.format = '$%s$' % matplotlib.ticker._mathdefault(self.format)

z = (np.random.random((10,10))*0.35+0.735)*1.e-7

fig, ax = plt.subplots()
plot = ax.contourf(z, levels=np.linspace(0.735e-7,1.145e-7,10))

#fmt = FormatScalarFormatter("%.2f")

cbar = fig.colorbar(plot,
                    format=FormatScalarFormatter(fformat="%.2f",
                                                 offset=True,
                                                 mathText=True))

plt.show()
