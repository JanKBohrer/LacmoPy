#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 15:45:49 2019

@author: jdesk
"""

import math
import numpy as np
import matplotlib.pyplot as plt

two_pi_sqrt = math.sqrt(2.0 * math.pi)

# par[0] = mu
# par[1] = sigma
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

def convert_sigma_star_to_sigma(mu_star, sigma_star):
    mu = mu_star * sigma_star**(0.5 * np.log(sigma_star))
    sigma = mu * np.sqrt( sigma_star**( np.log(sigma_star) ) - 1.0)
    return mu, sigma

def num_int(func, par, x0, x1, dx):
    res = 0.0
    if par is not None:
        f = lambda x : func(x, par)
    else: f = func
    for x in np.arange(x0, x1, dx):
        res += dx * f(x)
    # for x in np.arange(x0, x1, dx):
    #     if par is None:
    #         res += dx * func(x)
    #     else:
    #         res += dx * func(x, par)
    return res

# print(np.linspace(1.0/4,1.0,4))

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
# create random radii rad[i,j] 
# and the corresp. values of the distribution p[i,j]
# input: quantile probs to cut off P_min, P_max, e.g. (0.001,0.999)
# number of super-droplets per cell N_c
# distribution to use dst
# params of the distribution par -> None possible if dst = dst(x)
def generate_random_radii(grid_centers, P_min, P_max, N_c,
                          dst, par, x0, x1, dx, seed):
    np.random.seed(seed)
    if par is not None:
        func = lambda x : dst(x, par)
    else: func = dst
    
    qs, Ps = compute_quantiles(func, None, x0, x1, dx, None, [P_min, P_max])
    
    x_min = qs[0]
    x_max = qs[1]
    
    bins = np.linspace(x_min, x_max, N_c+1)
    
    
    no_cells = 1
    for s in np.shape(grid_centers)[1:]:
        no_cells *= s
    
    rnd = []
    for i,b in enumerate(bins[0:-1]):
        # we need 1 radius value for each bin for each cell
        rnd.append( np.random.uniform(b, bins[i+1], no_cells) )
    
    rnd = np.array(rnd)
    
    rnd = np.transpose(rnd)    
    shape = np.hstack( [np.array(np.shape(grid_centers)[1:]), [N_c]] )
    rnd = np.reshape(rnd, shape)
    
    weights = func(rnd)
    
    for p_row in weights:
        for p_row_col in p_row:
            p_row_col /= np.sum(p_row_col)
    
    return rnd, weights, bins

# N_c = # super-part/cell
# no_cells = [Nx, Nz]
def generate_random_positions(corners, no_cells, steps, N_c, seed):
    N_tot = no_cells[0] * no_cells[1] * N_c
    rnd_x = np.random.rand(N_tot)
    rnd_z = np.random.rand(N_tot)
    dx = steps[0]
    dz = steps[1]
    x = []
    z = []
    n = 0
    for j in range(no_cells[1]):
        z0 = corners[1][0,j]
        for i in range(no_cells[0]):
            x0 = corners[0][i,j]
            for k in range(N_c):
                x.append(x0 + dx * rnd_x[n])
                z.append(z0 + dz * rnd_z[n])
                n += 1
    pos = np.array([x,z])
    return pos
    

        

# # rad = array of R_s[i,j]
# # weights = array of wt[i,j]
# # N_tot = total number of particles per cell in cell N_tot[i,j]
# def compute_multiplicities(rad, weights, ):
#     return 

xs = 0.0
xe = 100.0
zs = 0.0
ze = 100.0

delta_x = 20.0
delta_y = 1.0
delta_z = 10.0

steps = [delta_x, delta_z]

X = np.arange(xs+0.5*delta_x, xe, delta_x)
Z = np.arange(zs+0.5*delta_z, ze, delta_z)
centers = np.meshgrid(X,Z , indexing = "ij")
no_cells = np.array( centers[0].shape )
print("no_cells = ", no_cells)
X = np.arange(xs, xe+delta_x, delta_x)
Z = np.arange(zs, ze+delta_z, delta_z)
corners = np.meshgrid(X,Z , indexing = "ij")

p_min = 0.001
p_max = 0.999
N_c = 8

# geometric mean and geometric std dev of the log-normal dist
mus = [0.02, 0.075]
sigmas = [1.4, 1.6]
mus_true, sigmas_true = convert_sigma_star_to_sigma(mus, sigmas)
par_dist = []
for i,mu in enumerate(mus):
    par_dist.append([mu, math.log(sigmas[i])])

cnt = 0
dx = 1E-5
x0 = dx
x1 = mus_true[cnt] + 20*sigmas_true[cnt]

seed = 4711

rad, weights, rad_bins = generate_random_radii(centers, p_min, p_max, N_c,
                             dst_log_normal, par_dist[0], x0, x1, dx, seed)

# print(bins)
# print(noc)
# print(shp)
print(rad)
print(rad.shape)
print(rad[1,1])
print(weights)
for p_row in weights:
    for p_row_col in p_row:
        print(np.sum(p_row_col))

# concentration in part/cm^3
n1 = 100
# no part per cell
V_cell = delta_x * delta_y * delta_z
N_base = V_cell * 1.0E6 * n1

# assume profile only dependent on z:
N_tot = np.ones_like(centers[0][0])
# fit in density function here!
N_tot *= N_base

xi = np.zeros_like( weights )

print(N_tot)

for j in range(len(N_tot)):
    xi[:,j] = weights[:,j] * N_tot[j]

xi = xi.astype(int)

print(xi)

R_s = rad.flatten()

# -> convert to m_s

pos = generate_random_positions(corners, no_cells, steps, N_c, seed)

print(pos)


def plot_particle_positions(corners, pos):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot()
# xi[:,] = 

# rad2 = np.reshape(rad, (5,5,N_c))

# print(rad2)

# k = 0
# for i in range(5):
#     for j in range(5):
#         print(rad[k] - rad2[i,j]  )
#         k += 1
    
# print( generate_random_radii(centers, p_min, p_max, N_c,
#                              dst_log_normal, par_dist[0], x0, x1, dx, seed) )




#%%
# mu_star = 0.02
# sigma_star = 1.4
# paras = [mu_star, math.log(sigma_star)]
# mu_t, sigma_t = convert_sigma_star_to_sigma(mu_star, sigma_star)
# print("mu_true, sigma_true = ", mu_t, sigma_t)
# dx = 1.0E-5
# x0 = dx
# # x0 = paras[0] - 10 * paras[1]
# x1 = paras[0] + 20 * sigma_t
# Nq = 8

# # P = [0.9, 0.95, 0.975, 0.99, 0.995, 0.999]

# # intl = num_int(dst_normal, paras, x0, x1, dx)
# # qs, Ps = compute_quantiles(dst_normal, paras, x0, x1, dx, Nq = None, P=P)
# intl = num_int(dst_log_normal, paras, x0, x1, dx)
# qs, Ps = compute_quantiles(dst_log_normal, paras, x0, x1, dx, Nq)

# print("integral = ", intl)
# for i,q in enumerate(qs):
#     print(Ps[i], q)

    
#%%
        
# geometric mean and geometric std dev of the log-normal dist
mus = [0.02, 0.075]
sigmas = [1.4, 1.6]
mus_true, sigmas_true = convert_sigma_star_to_sigma(mus, sigmas)

Nq = 8

Pq = np.hstack( ([0.001], np.linspace(1.0/Nq, 1.0, Nq)[:-1], [0.999]  ) )

par_dist = []
qs = []
Ps = []
for i,mu in enumerate(mus):
    par_dist.append([mu, math.log(sigmas[i])])

for i,paras in enumerate( par_dist ):
    dx = 1.0E-5
    x0 = dx
    x1 = mus_true[i] + 20*sigmas_true[i]
    intl = num_int(dst_log_normal, paras, x0, x1, dx)
    q, P = compute_quantiles(dst_log_normal, paras, x0, x1, dx, None, Pq)
    qs.append(q)
    Ps.append(P)
    
    print("paras = ", paras)
    print("true paras = ", [mus_true[i], sigmas_true[i]])
    print("integral = ", intl)
    print()
    for i,q_ in enumerate(q):
        print(P[i], q_)
    print()

#%%
### random numbers
# let z = ln (x) -> z is normal distributed with
mu_z = np.log(mus)
sigma_z = np.log(sigmas)

# draw random number from normal distribution
seed = 4711
np.random.seed(seed)

randoms = []
for i,mu_ in enumerate(mu_z):
    N = 10000
    rnd = np.random.normal(mu_z[i], sigma_z[i], N)
    rnd = np.exp(rnd)
    randoms.append(rnd)

#%%
### plotting
fig, axes = plt.subplots(2, figsize=(10,10))

for cnt in [0,1]:
    bins = 500
    ax = axes[cnt]
    mu_t = mus_true[cnt]
    sig_t = sigmas_true[cnt]
    x = np.linspace(0.000001, mu_t + 10 * sig_t, 10000)
    ax.plot(x, dst_log_normal(x, par_dist[cnt]))
    ax.vlines(mu_t, 0.0, dst_log_normal(mu_t,par_dist[cnt]), color = "red")
    ax.vlines(mu_t-sig_t, 0.0, dst_log_normal(mu_t-sig_t,par_dist[cnt]),
              color = "red")
    ax.vlines(mu_t+sig_t, 0.0, dst_log_normal(mu_t+sig_t,par_dist[cnt]),
              color = "red")
    
    # for i,q in enumerate(qs[cnt]):
    #     # ax.axvline(q, color = "green")
    #     ax.vlines(q, 0.0, dst_log_normal(q,par_dist[cnt]), color = "green")
    
    #     # ax.annotate(f"{q:.3f},{Ps[cnt][i]}", (q, -1.0))
    #     ax.plot(q, dst_log_normal(q,par_dist[cnt]), "o" )
    #     i_mid = int(len(qs[cnt])//2)-1
    #     f_mid = dst_log_normal(qs[cnt][i_mid],par_dist[cnt])
    #     print("i_mid = ", i_mid)
    #     q_an = q - sig_t * 0.6 if q < qs[cnt][i_mid] \
    #                            else q + sig_t * 0.6
    #     ax.annotate( f"{q:.3f},{Ps[cnt][i]}",
    #                   (q_an, dst_log_normal(q,par_dist[cnt]) + 0.02*f_mid),
    #                   ha = "center"
    #                   # xytext = (q + (q-mu_t)* , 0.0),
    #                    textcoords = "offset points"
    #                   )
        # ax.annotate( f"{q:.3f},{Ps[cnt][i]}",
        #               (lambda q : q + 0.0002*(q-qs[cnt][3])\
        #                               / (0.01 + q-qs[cnt][3])**2,
        #               dst_log_normal(q,par_dist[cnt])),
        #               ha = "center"
        #               # xytext = (q + (q-mu_t)* , 0.0),
        #               textcoords = "offset points"
        #               )
    
    # ax.hist(randoms[cnt], bins, density=True)
    if cnt == 0:
        ax.hist(rad[0,0], rad_bins, weights = xi[0,0],
                density=True,
                # histtype="stepfilled"
                histtype="bar"
                )
        ax.plot(rad[0,0], dst_log_normal(rad[0,0],par_dist[cnt]), "o")
    ax.set_xlabel("dry radius (nm)")
    ax.set_ylabel("PDF")
    tab = "\t"
    ax.set_title(
            f"geom. paras = {par_dist[cnt][0]:.3f}, {par_dist[cnt][1]:.3f},\
    arith. paras = {mu_t:4.3f}, {sig_t:.4f}    {N} random #, {bins} bins")
    ax.grid()
    cnt += 1

fig.tight_layout()
# fig.savefig("distributions.png")
# plt.show()