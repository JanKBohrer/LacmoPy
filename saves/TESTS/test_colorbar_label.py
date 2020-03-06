#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 18:12:43 2019

@author: jdesk
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker

class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_orderOfMagnitude(self, nothing):
        self.orderOfMagnitude = self.oom
#    def _set_format(self, vmin, vmax):
    def _set_format(self):
        self.format = self.fformat
        if self._useMathText:
            self.format = '$%s$' % matplotlib.ticker._mathdefault(self.format)



x = np.linspace(1,9,9)
y1 = x*10**(-4)
y2 = x*10**(-3)

fig, ax = plt.subplots(2,1,sharex=True)

ax[0].plot(x,y1)
ax[1].plot(x,y2)

for axe in ax:
    axe.yaxis.set_major_formatter(OOMFormatter(-4, "%1.1f"))
    axe.ticklabel_format(axis='y', style='sci', scilimits=(-4,-4))

plt.show()