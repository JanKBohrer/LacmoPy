#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 11:31:49 2019

@author: jdesk
"""

import numpy as np
import math.pi
import matplotlib.pyplot as plt
from grid import Grid
# IN WORK: what do you need from grid.py?
# from grid import *
# from physical_relations_and_constants import *

# dry mass flux  j_d = rho_d * vel
# Z = domain height z
# X = domain width x
# X_over_Z = X/Z
# k_z = pi/Z
# k_x = 2 * pi / X
# j_max = Amplitude
def compute_mass_flux_air_dry_Arabas( x_, z_, j_max_, k_x_, k_z_, X_over_Z_ ):
    j_x = j_max_ * X_over_Z_ * np.cos(k_x_ * x_) * np.cos(k_z_ * z_ )
    j_z = 2 * j_max_ * np.sin(k_x_ * x_) * np.sin(k_z_ * z_)
    return j_x, j_z

#def compute_mass_flux_air_dry_x( x_, z_, j_max_, k_x_, k_z_, X_over_Z_ ):
#    return j_max_ * X_over_Z_ * np.cos(k_x_ * x_) * np.cos(k_z_ * z_ )
#def compute_mass_flux_air_dry_z( x_, z_, j_max_, k_x_, k_z_, X_over_Z_ ):
#    return 2 * j_max_ * np.sin(k_x_ * x_) * np.sin(k_z_ * z_)
#def mass_flux_air_dry(x_, z_):
#    return compute_mass_flux_air_dry_Arabas(x_, z_, j_max, k_x, k_z, X_over_Z)
#
#def mass_flux_air_dry_x(x_, z_):
#    return compute_mass_flux_air_dry_x(x_, z_, j_max, k_x, k_z, X_over_Z)
#def mass_flux_air_dry_z(x_, z_):
#    return compute_mass_flux_air_dry_z(x_, z_, j_max, k_x, k_z, X_over_Z)

def compute_initial_mass_flux_air_dry_kinematic_2D_ICMW_2012_case1( grid_,
                                                                    j_max_ ):
    X = grid_.sizes[0]
    Z = grid_.sizes[1]
    k_x = 2.0 * np.pi / X
    k_z = np.pi / Z
    X_over_Z = X / Z
#    j_max = 0.6 # (m/s)*(m^3/kg)
    # grid only has corners as positions...
    vel_pos_u = [grid_.corners[0], grid_.corners[1] + 0.5 * grid_.steps[1]]
    vel_pos_w = [grid_.corners[0] + 0.5 * grid_.steps[0], grid_.corners[1]]
    j_x = compute_mass_flux_air_dry_Arabas( *vel_pos_u, j_max_,
                                            k_x, k_z, X_over_Z )[0]
    j_z = compute_mass_flux_air_dry_Arabas( *vel_pos_w, j_max_,
                                            k_x, k_z, X_over_Z )[1]
    return j_x, j_z

# j_max_base = the appropriate value for a grid of 1500 x 1500 m^2
def compute_j_max(j_max_base, grid):
    return np.sqrt( grid.sizes[0] * grid.sizes[0]
                  + grid.sizes[1] * grid.sizes[1] ) \
           / np.sqrt(2.0 * 1500.0 * 1500.0)

####
# par[0] = mu
# par[1] = sigma
two_pi_sqrt = math.sqrt(2.0 * math.pi)
def dst_normal(x, par):
    return np.exp( -0.5 * ( ( x - par[0] ) / par[1] )**2 ) \
           / (two_pi_sqrt * par[1])

# par[0] = mu^* = geometric mean of the log-normal dist
# par[1] = ln(sigma^*) = lognat of geometric std dev of log-normal dist
def dst_log_normal(x,par):
    # sig = math.log(par[1])
    f = np.exp( -0.5 * ( np.log( x / par[0] ) / par[1] )**2 ) \
        / ( x * math.sqrt(2 * math.pi) * par[1] )
    return f

# input: P =[p1, p2, p3, ...] = quantile probabilites -> need to set Nq = None
# OR (if P = None)
# input: Nq = number of quantiles (including P = 1.0)
# returns N-1 quantiles q with given (P) or equal probability distances (Nq):
# N = 4 -> P(X < q[0]) = 1/4, P(X < q[1]) = 2/4, P(X < q[2]) = 3/4
# this function was checked with the std normal distribution
def compute_quantiles(func, par, x0, x1, dx, Nq = None, P = None):
    # probabilities of the quantiles
    if par is not None:
        f = lambda x : func(x, par)
    else: f = func
    
    if P is None:
        P = np.linspace(1.0/Nq,1.0,Nq)[:-1]
    print ("quantile probabilities = ", P)
    
    intl = 0.0
    x = x0
    q = []
    
    cnt = 0
    for i,p in enumerate(P):
        while (intl < p and p < 1.0 and x <= x1 and cnt < 1E8):
            intl += dx * f(x)
            x += dx
            cnt += 1
        # the quantile value is somewhere between x - dx and x -> choose middle
        # q.append(x)    
        # q.append(x - 0.5 * dx)    
        q.append(x - dx)    
    
    print ("quantile values = ", q)
    return q, P
          
# for a given grid with grid center positions grid_centers[i,j]:
# create random radii rad[i,j] = array of shape (Nx, Ny, N_spc)
# and the corresp. values of the distribution p[i,j]
# input:
# p_min, p_max: quantile probs to cut off e.g. (0.001,0.999)
# N_spc: number of super-droplets per cell
# dst: distribution to use
# par: params of the distribution par -> "None" possible if dst = dst(x)
# r0, r1, dr: parameters for the numerical integration to find the cutoffs
def generate_random_radii(grid, p_min, p_max, N_spc,
                          dst, par, r0, r1, dr, seed):
    np.random.seed(seed)
    if par is not None:
        func = lambda x : dst(x, par)
    else: func = dst
    
    qs, Ps = compute_quantiles(func, None, r0, r1, dr, None, [p_min, p_max])
    
    r_min = qs[0]
    r_max = qs[1]
    
    bins = np.linspace(r_min, r_max, N_spc+1)
    
    rnd = []
    for i,b in enumerate(bins[0:-1]):
        # we need 1 radius value for each bin for each cell
        rnd.append( np.random.uniform(b, bins[i+1], grid.no_cells_tot) )
    
    rnd = np.array(rnd)
    
    rnd = np.transpose(rnd)    
    shape = np.hstack( [np.array(np.shape(grid.centers)[1:]), [N_spc]] )
    rnd = np.reshape(rnd, shape)
    
    weights = func(rnd)
    
    for p_row in weights:
        for p_row_col in p_row:
            p_row_col /= np.sum(p_row_col)
    
    return rnd, weights #, bins

# N_spc = # super-part/cell
# N_spt = # SP total
# returns positions of particles of shape (2, N_spt)
def generate_random_positions(grid, N_spc, seed):
    N_tot = grid.no_cells_tot * N_spc
    rnd_x = np.random.rand(N_tot)
    rnd_z = np.random.rand(N_tot)
    dx = grid.steps[0]
    dz = grid.steps[1]
    x = []
    z = []
    n = 0
    for j in range(grid.no_cells[1]):
        z0 = grid.corners[1][0,j]
        for i in range(grid.no_cells[0]):
            x0 = grid.corners[0][i,j]
            for k in range(N_spc):
                x.append(x0 + dx * rnd_x[n])
                z.append(z0 + dz * rnd_z[n])
                n += 1
    pos = np.array([x,z])
    return pos

# # c "per cell", t "tot"
# # rad and weights are (Nx,Ny,N_spc) array
# # pos is (N_spt,) array
# def generate_particle_positions_and_dry_radii(grid, p_min, p_max, N_spc,
#                                               dst, par, r0, r1, dr, seed):
#     rad, weights = generate_random_radii(grid, p_min, p_max, N_spc,
#                                          dst, par, r0, r1, dr, seed)
    
#     return pos, rad, weights

def initialize_grid_and_particles():
    pass


# particles: pos, vel, masses, multi,

# base grid without THD or flow values
    
# domain size
x_min = 0.0
x_max = 100.0
z_min = 0.0
z_max = 200.0

# grid steps
dx = 20.0
dy = 1.0
dz = 20.0

grid_ranges = [ [x_min, x_max], [z_min, z_max] ]
grid_steps = [dx, dz]
grid = Grid( grid_ranges, grid_steps, dy )

# set mass flux, note that the velocity requires the dry density field first
j_max = 0.6
j_max = compute_j_max(j_max, grid)
grid.mass_flux_air_dry = \
    compute_initial_mass_flux_air_dry_kinematic_2D_ICMW_2012_case1(grid, j_max)

# gen. positions of N_spc Sp per cell
N_spc = 8
seed = 4713
pos = generate_random_positions(grid, N_spc, seed)
    
# visualize particles in grid
# %matplotlib inline
def plot_pos_pt(pos, grid,figsize=(6,6), MS = 1.0):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(grid.corners[0], grid.corners[1], "x", color="red", markersize=MS)
    ax.plot(pos[0],pos[1], "o", color="k", markersize=2*MS)
    ax.set_xticks(grid.corners[0][:,0])
    ax.set_yticks(grid.corners[1][0,:])
    # ax.set_xticks(grid.corners[0][:,0], minor = True)
    # ax.set_yticks(grid.corners[1][0,:], minor = True)
    plt.minorticks_off()
    # plt.minorticks_on()
    ax.grid()
    # ax.grid(which="minor")
    plt.show()

print(grid.corners[0][:,0])
plot_pos_pt(pos[:,0:8*8], grid, MS = 2.0)
    
