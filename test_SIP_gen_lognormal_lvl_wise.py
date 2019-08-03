#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 18:00:26 2019

@author: jdesk
"""
import numpy as np

from microphysics import compute_mass_from_radius_vec, compute_mass_from_radius_jit

import constants as c

from generate_SIP_ensemble_dst import\
    gen_mass_ensemble_weights_SinSIP_lognormal_z_lvl

#def moments_analytical_lognormal_m(n, DNC, mu_m_log, sigma_m_log):
#    if n == 0:
#        return DNC
#    else:
#        return DNC * np.exp(n * mu_m_log + 0.5 * n*n * sigma_m_log*sigma_m_log

def moments_analytical_lognormal_m(n, DNC, mu_m_log, sigma_m_log):
    if n == 0:
        return DNC
    else:
        return DNC * np.exp(n * mu_m_log + 0.5 * n*n * sigma_m_log*sigma_m_log)

#%% GRID PARAMETERS
# domain size
#x_min = 0.0
#x_max = 1500.0
#z_min = 0.0
#z_max = 1500.0
#
## grid steps
##dx = 150.0
##dy = 1.0
##dz = 150.0
#dx = 20.0
#dy = 1.0
#dz = 20.0
##dx = 500.0
##dy = 1.0
##dz = 500.0
#
#dV = dx*dy*dz
#
#p_0 = 1015E2 # surface pressure in Pa
#p_ref = 1.0E5 # ref pressure for potential temperature in Pa
#r_tot_0 = 7.5E-3 # kg water / kg dry air (constant over whole domain in setup)
## r_tot_0 = 22.5E-3 # kg water / kg dry air
## r_tot_0 = 7.5E-3 # kg water / kg dry air
#Theta_l = 289.0 # K
#
##%% PARTICLE PARAMETERS
#
## no_super_particles_cell_mode = [N1,N2] is a list with
## N1 = no super part. per cell in mode 1 etc.
## with init method = SingleSIP, this is only the target value.
## the true number of particles per cell and mode will fluctuate around this
##no_spcm = np.array([0, 4])
#
#no_cells = compute_no_grid_cells_from_step_sizes(
#               ((x_min, x_max),(z_min, z_max)), (dx, dz) ) 



no_spcm = np.array([20, 20])
#no_spcm = np.array([16, 24])

no_cells = np.array((3,3))

dV = 1500/no_cells[0] * 1500/no_cells[1]


reseed = False
seed_SIP_gen = 3713
#rnd_seed = seed

#grid_folder =\
#    f"grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm[0]}_{no_spcm[1]}/" \
#    + f"{seed_SIP_gen}/"

idx_mode_nonzero = np.nonzero(no_spcm)[0]

no_modes = len(idx_mode_nonzero)

if no_modes == 1:
    no_spcm = no_spcm[idx_mode_nonzero][0]
else:    
    no_spcm = no_spcm[idx_mode_nonzero]

print("no_modes, idx_mode_nonzero:", no_modes, ",", idx_mode_nonzero)
#print()

### creating random particles
# parameters of log-normal distribution:
#dst = dst_log_normal

dist = "lognormal"
#dist = "expo"

if dist == "lognormal":
    r_critmin = np.array([0.001, 0.003]) # mu, # mode 1, mode 2, ..
    m_high_over_m_low = 1.0E8
    
    # droplet number concentration 
    # set DNC0 manually only for lognormal distr.
    # number density of particles mode 1 and mode 2:
    DNC0 = np.array([60.0E6, 40.0E6]) # 1/m^3
    # n_p = np.array([60.0E6, 40.0E6]) # 1/m^3
#    DNC0 = 60.0E6 # 1/m^3
    #DNC0 = 2.97E8 # 1/m^3
    #DNC0 = 3.0E8 # 1/m^3
    # parameters of radial lognormal distribution -> converted to mass
    # parameters below
#    mu_R = 0.02 # in mu
#    sigma_R = 1.4 #  no unit
    mu_R = 0.5 * np.array( [0.04, 0.15] )
    sigma_R = np.array( [1.4,1.6] )
    
    if no_modes == 1:
        sigma_R = sigma_R[idx_mode_nonzero][0]
        mu_R = mu_R[idx_mode_nonzero][0]
#        no_rpcm = no_rpcm[idx_mode_nonzero][0]
        r_critmin = r_critmin[idx_mode_nonzero][0] # mu
        DNC0 = DNC0[idx_mode_nonzero][0] # mu
        
    else:    
        sigma_R = sigma_R[idx_mode_nonzero]
        mu_R = mu_R[idx_mode_nonzero]
#        no_rpcm = no_rpcm[idx_mode_nonzero]
        r_critmin = r_critmin[idx_mode_nonzero] # mu
        DNC0 = DNC0[idx_mode_nonzero] # mu

    mu_R_log = np.log( mu_R )
    sigma_R_log = np.log( sigma_R )    
    # derive parameters of lognormal distribution of mass f_m(m)
    # assuming mu_R in mu and density in kg/m^3
    # mu_m in 1E-18 kg
    mu_m_log = np.log(compute_mass_from_radius_vec(mu_R,
                                                   c.mass_density_NaCl_dry))
    sigma_m_log = 3.0 * sigma_R_log
    dst_par = (mu_m_log, sigma_m_log)

elif dist == "expo":
    LWC0 = 1.0E-3 # kg/m^3
    R_mean = 9.3 # in mu
    r_critmin = 0.6 # mu
    m_high_over_m_low = 1.0E6    

    rho_w = 1E3
    m_mean = compute_mass_from_radius_jit(R_mean, rho_w) # in 1E-18 kg
    DNC0 = 1E18 * LWC0 / m_mean # in 1/m^3
    # we need to hack here a little because of the units of m_mean and LWC0
    LWC0_over_DNC0 = m_mean
    DNC0_over_LWC0 = 1.0/m_mean
    print("dist = expo", f"DNC0 = {DNC0:.3e}", "LWC0 =", LWC0,
          "m_mean = ", m_mean)
    dst_par = (DNC0, DNC0_over_LWC0)


#%% SINGLE SIP INITIALIZATION PARAMETERS

# derive scaling parameter kappa from no_spcm -> shifted to
# initialize_grid_and_particles_SinSIP (init.py)
#kappa = np.rint( no_spcm / 20 * 35) * 0.1
#kappa = np.maximum(kappa, 0.1)
#print("kappa =", kappa)

eta = 1E-10
#eta_threshold = "weak"
eta_threshold = "fix"
if eta_threshold == "weak":
    weak_threshold = True
else: weak_threshold = False

m_high_over_m_low = 1.0E8    

#%%

#%%
kappa_dst = np.rint( no_spcm / 20 * 28) * 0.1

no_rpcm =  (np.ceil( dV*DNC0 )).astype(int)

no_moments = 4

m_s_lvl, xi_lvl, cells_x_lvl, modes_lvl, no_spc_lvl = \
    gen_mass_ensemble_weights_SinSIP_lognormal_z_lvl(no_modes,
            mu_m_log, sigma_m_log, c.mass_density_NaCl_dry,
            dV, kappa_dst, eta, weak_threshold, r_critmin,
            m_high_over_m_low, seed_SIP_gen, no_cells[0], no_rpcm,
            setseed=False)



#moments_an = []
#for j in range(no_cells[1]):
moments_an_lvl = []
for mom_n in range(no_moments):
    moments_an_lvl.append(
        moments_analytical_lognormal_m(
            mom_n, DNC0,
            mu_m_log, sigma_m_log))
#    moments_an.append(moments_an_lvl)
        
moments_an_lvl = np.array(moments_an_lvl)   


mask = np.logical_and(cells_x_lvl == 0, modes_lvl == 0)
xi_0 = xi_lvl[mask]
m_s_0 = m_s_lvl[mask]

mom_0_0_0 = xi_0.sum() / dV

mom_0 = np.zeros(4)

mom_0[0] = mom_0_0_0

for mom_n in range(1,no_moments):
    mom_0[mom_n] = \
        np.sum( xi_0 * m_s_0**mom_n ) / dV

print((moments_an_lvl[:,0] - mom_0) / moments_an_lvl[:,0])
    