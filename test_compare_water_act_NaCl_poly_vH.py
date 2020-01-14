#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 12:43:26 2020

@author: bohrer
"""

import numpy as np
from matplotlib import pyplot as plt


from microphysics import compute_water_activity_NaCl_vH
from microphysics import compute_water_activity_NaCl
from microphysics import compute_mass_from_radius

import constants as c

def compute_m_w_from_m_s_and_w_s(m_s, w_s):
    return m_s * (1./w_s - 1.)

R_s = 10 # mu
m_s = compute_mass_from_radius(R_s, c.mass_density_NaCl_dry) # fg

w_s = np.logspace(-5,-0.3,100)
m_w = compute_m_w_from_m_s_and_w_s(m_s, w_s)

print(R_s)
print(m_s)
print(w_s)
print(m_w)


aw1 = compute_water_activity_NaCl_vH(m_w, m_s, w_s)
aw2 = compute_water_activity_NaCl(w_s)

#%% plotting


fig, ax = plt.subplots(figsize=(8,8))


ax.plot(w_s, aw1, label="vantHoffArcher92")
ax.plot(w_s, aw2, label="polyTang1997")

ax.grid()
ax.legend()
ax.set_xlabel("solute mass fraction")
ax.set_ylabel("water activity")

#fig.savefig("compare_wat_act_NaCl_poly_and_vantHoff.png")
