#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 14:05:10 2020

@author: jdesk
"""

#%% GENERATE SIP ENSEMBLES
### OLD APPROACH

# input: prob = [p1, p2, p3, ...] = quantile probab. -> need to set no_q = None
# OR (if prob = None)
# input: no_q = number of quantiles (including prob = 1.0)
# returns N-1 quantiles q with given (prob)
# or equal probability distances (no_q):
# N = 4 -> P(X < q[0]) = 1/4, P(X < q[1]) = 2/4, P(X < q[2]) = 3/4
# this function was checked with the std normal distribution
def compute_quantiles(func, par, x0, x1, dx, prob, no_q=None):
    # probabilities of the quantiles
    if par is not None:
        f = lambda x : func(x, par)
    else: f = func
    
    if prob is None:
        prob = np.linspace(1.0/no_q,1.0,no_q)[:-1]
    print ('quantile probabilities = ', prob)
    
    intl = 0.0
    x = x0
    q = []
    
    cnt = 0
    for i,p in enumerate(prob):
        while (intl < p and p < 1.0 and x <= x1 and cnt < 1E8):
            intl += dx * f(x)
            x += dx
            cnt += 1
        # the quantile value is somewhere between x - dx and x
        # q.append(x)    
        # q.append(x - 0.5 * dx)    
        q.append(max(x - dx,x0))    
    
    print ('quantile values = ', q)
    return q, prob

# for a given grid with grid center positions grid_centers[i,j]:
# create random radii rad[i,j] = array of shape (Nx, Ny, no_spc)
# and the corresp. values of the distribution p[i,j]
# input:
# p_min, p_max: quantile probs to cut off e.g. (0.001,0.999)
# no_spc: number of super-droplets per cell (scalar!)
# dst: distribution to use
# par: params of the distribution par -> 'None' possible if dst = dst(x)
# r0, r1, dr: parameters for the numerical integration to find the cutoffs
def generate_random_radii_monomodal(grid, dst, par, no_spc, p_min, p_max,
                                    r0, r1, dr, seed, setseed = True):
    if setseed: np.random.seed(seed)
    
    if par is not None:
        func = lambda x : dst(x, par)
    else: func = dst
    
    qs, Ps = compute_quantiles(func, None, r0, r1, dr, [p_min, p_max], None)
    
    r_min = qs[0]
    r_max = qs[1]
    
    bins = np.linspace(r_min, r_max, no_spc+1)
    
    rnd = []
    for i,b in enumerate(bins[0:-1]):
        # we need 1 radius value for each bin for each cell
        rnd.append( np.random.uniform(b, bins[i+1], grid.no_cells_tot) )
    
    rnd = np.array(rnd)
    
    rnd = np.transpose(rnd)    
    shape = np.hstack( [np.array(np.shape(grid.centers)[1:]), [no_spc]] )
    rnd = np.reshape(rnd, shape)
    
    weights = func(rnd)
    
    for p_row in weights:
        for p_cell in p_row:
            p_cell /= np.sum(p_cell)
    
    return rnd, weights

# no_spc = super particles per cell
# creates no_spc random radii per cell and the no_spc weights per cell
# where sum(weight_i) = 1.0
# the weights are drawn from a normal distribution with mu = 1.0, sigma = 0.2
# and rejected if weight < 0. The weights are then normalized such that sum = 1
# par = (mu*, ln(sigma*)), where mu* and sigma* are the
# GEOMETRIC expectation value and standard dev. of the lognormal distr. resp.
def generate_random_radii_monomodal_lognorm(grid, par, no_spc,
                                            seed, setseed = True):
    if setseed: np.random.seed(seed)
    
    no_spt = grid.no_cells_tot * no_spc
    
    # draw random numbers from log normal distr. by this procedure
    Rs = np.random.normal(0.0, par[1], no_spt)
    Rs = np.exp(Rs)
    Rs *= par[0]
    # draw random weights
    weights = np.abs(np.random.normal(1.0, 0.2, no_spt))
    
    # bring to shape Rs[i,j,k], where Rs[i,j] = arr[k] k = 0...no_spc-1
    shape = np.hstack( [np.array(np.shape(grid.centers)[1:]), [no_spc]] )
    Rs = np.reshape(Rs, shape)
    weights = np.reshape(weights, shape)
    
    for p_row in weights:
        for p_cell in p_row:
            p_cell /= np.sum(p_cell)
    
    return Rs, weights

# no_spcm must be a list/array with [no_1, no_2, .., no_N], N = number of modes
# no_spcm[k] = 0 is possible for some mode k. Then no particles are generated
# for this mode...
# par MUST be a list with [par1, par2, .., parN]
# where par1 = [p11, p12, ...] , par2 = [...] etc
# everything else CAN be a list or a scalar single float/int
# if given as scalar, all modes will use the same value
# the seed is set at least once
# reseed = True will reset the seed everytime for every new mode with the value
# given in the seed list, it can also be used with seed being a single scalar
# about the seeds:
# using np.random.seed(seed) in a function sets the seed globaly
# after leaving the function, the seed remains the set seed
def generate_random_radii_multimodal(grid, dst, par, no_spcm, p_min, p_max, 
                                     r0, r1, dr, seed, reseed = False):
    no_modes = len(no_spcm)
    if not isinstance(p_min, (list, tuple, np.ndarray)):
        p_min = [p_min] * no_modes
    if not isinstance(p_max, (list, tuple, np.ndarray)):
        p_max = [p_max] * no_modes
    if not isinstance(dst, (list, tuple, np.ndarray)):
        dst = [dst] * no_modes
    if not isinstance(r0, (list, tuple, np.ndarray)):
        r0 = [r0] * no_modes
    if not isinstance(r1, (list, tuple, np.ndarray)):
        r1 = [r1] * no_modes
    if not isinstance(dr, (list, tuple, np.ndarray)):
        dr = [dr] * no_modes
    if not isinstance(seed, (list, tuple, np.ndarray)):
        seed = [seed] * no_modes        

    rad = []
    weights = []
    print(p_min)
    # set seed always once
    setseed = True
    for k in range(no_modes):
        if no_spcm[k] > 0:
            r, w = generate_random_radii_monomodal(
                       grid, dst[k], par[k], no_spcm[k], p_min[k], p_max[k],
                       r0[k], r1[k], dr[k], seed[k], setseed)
            rad.append( r )
            weights.append( w )
            setseed = reseed
            
    return np.array(rad), np.array(weights)

def generate_random_radii_multimodal_lognorm(grid, par, no_spcm,
                                             seed, reseed = False):
    no_modes = len(no_spcm)
    if not isinstance(seed, (list, tuple, np.ndarray)):
        seed = [seed] * no_modes       
        
    rad = []
    weights = []
    # set seed always once
    setseed = True
    for k in range(no_modes):
        if no_spcm[k] > 0:
            r, w = generate_random_radii_monomodal_lognorm(
                       grid, par[k], no_spcm[k], seed[k], setseed)
            rad.append( r )
            weights.append( w )
            setseed = reseed
            
    return np.array(rad), np.array(weights)


# par = 'rate' parameter 'k' of the expo distr: k*exp(-k*m) (in 10^18 kg)
# no_rpc = number of real particles in cell
def generate_SIP_ensemble_expo_SingleSIP_weak_threshold(
        par, no_rpc, r_critmin, m_high_by_m_low, kappa,
        eta, seed, setseed):
    if setseed: np.random.seed(seed)
    m_low = mp.compute_mass_from_radius(r_critmin,
                                     c.mass_density_water_liquid_NTP)
    m_high = m_low * m_high_by_m_low
    # since we consider only particles with m > m_low, the total number of
    # placed particles and the total placed mass will be underestimated
    # to fix this, we could adjust the PDF
    # by e.g. one of the two possibilities:
    # 1. decrease the total number of particles and thereby the total pt conc.
    # 2. keep the total number of particles, and thus increase the PDF
    # in the interval [m_low, m_high]
    # # For 1.: decrease the total number of particles by multiplication 
    # # with factor num_int_expo_np(m_low, m_high, par, steps=1.0E6)
    # print(no_rpc)
    # no_rpc *= num_int_expo(m_low, m_high, par, steps=1.0E6)
    
    # for 2.:
    # increase the total number of
    # real particle 'no_rpc' by 1.0 / int_(m_low)^(m_high)
    # no_rpc *= 1.0/num_int_expo(m_low, m_high, par, steps=1.0E6)
    
    kappa_inv = 1.0/kappa
    bin_fac = 10**kappa_inv
    
    masses = []
    xis = []
    bins = [m_low]
    no_rp_set = 0
    no_sp_set = 0
    bin_n = 0
    
    m = np.random.uniform(m_low/2, m_low)
    xi = dist.dst_expo(m,par) * m_low * no_rpc
    masses.append(m)
    xis.append(xi)
    no_rp_set += xi
    no_sp_set += 1    
    
    m_left = m_low

    while(m < m_high):
        m_right = m_left * bin_fac
        m = np.random.uniform(m_left, m_right)
        # we do not round to integer here, because of the weak threshold below
        # the rounding is done afterwards
        xi = dist.dst_expo(m,par) * (m_right - m_left) * no_rpc
        masses.append(m)
        xis.append(xi)
        no_rp_set += xi
        no_sp_set += 1
        m_left = m_right
        bins.append(m_right)
        bin_n += 1
    
    xis = np.array(xis)
    masses = np.array(masses)
    
    xi_max = xis.max()
    
    xi_critmin = int(xi_max * eta)
    if xi_critmin < 1: xi_critmin = 1
    
    for bin_n,xi in enumerate(xis):
        if xi < xi_critmin:
            p = xi / xi_critmin
            if np.random.rand() >= p:
                xis[bin_n] = 0
            else: xis[bin_n] = xi_critmin
    
    ind = np.nonzero(xis)
    xis = xis[ind].astype(np.int64)
    masses = masses[ind]
    
    return masses, xis, m_low, m_high, bins

# par = 'rate' parameter 'k' of the expo distr: k*exp(-k*m) (in 10^18 kg)
# no_rpc = number of real particles in cell
# NON INTERGER WEIGHTS ARE POSSIBLE AS IN UNTERSTRASSER 2017
def generate_SIP_ensemble_expo_SingleSIP_weak_threshold_nonint(
        par, no_rpc, r_critmin=0.6, m_high_by_m_low=1.0E6, kappa=40,
        eta=1.0E-9, seed=4711, setseed = True):
    
    if setseed: np.random.seed(seed)
    m_low = mp.compute_mass_from_radius(r_critmin,
                                     c.mass_density_water_liquid_NTP)
    m_high = m_low * m_high_by_m_low
    # since we consider only particles with m > m_low, the total number of
    # placed particles and the total placed mass will be underestimated
    # to fix this, we could adjust the PDF
    # by e.g. one of the two possibilities:
    # 1. decrease the total number of particles and thereby the total pt conc.
    # 2. keep the total number of particles, and thus increase the PDF
    # in the interval [m_low, m_high]
    # # For 1.: decrease the total number of particles by multiplication 
    # # with factor num_int_expo_np(m_low, m_high, par, steps=1.0E6)
    # print(no_rpc)
    # no_rpc *= num_int_expo(m_low, m_high, par, steps=1.0E6)
    
    # for 2.:
    # increase the total number of
    # real particle 'no_rpc' by 1.0 / int_(m_low)^(m_high)
    # no_rpc *= 1.0/num_int_expo(m_low, m_high, par, steps=1.0E6)
    
    kappa_inv = 1.0/kappa
    bin_fac = 10**kappa_inv
    
    masses = []
    xis = []
    bins = [m_low]
    no_rp_set = 0
    no_sp_set = 0
    bin_n = 0
    
    m = np.random.uniform(m_low/2, m_low)
    xi = dist.dst_expo(m,par) * m_low * no_rpc
    masses.append(m)
    xis.append(xi)
    no_rp_set += xi
    no_sp_set += 1    
    
    m_left = m_low
    # m = 0.0
    
    # print('no_rpc =', no_rpc)
    # while(no_rp_set < no_rpc and m < m_high):
    while(m < m_high):
        # m_right = m_left * 10**kappa_inv
        m_right = m_left * bin_fac
        # print('missing particles =', no_rpc - no_rp_set)
        # print('m_left, m_right, m_high')
        # print(bin_n, m_left, m_right, m_high)
        m = np.random.uniform(m_left, m_right)
        # print('m =', m)
        # we do not round to integer here, because of the weak threshold below
        # the rounding is done afterwards
        xi = dist.dst_expo(m,par) * (m_right - m_left) * no_rpc
        # xi = int((dst_expo(m,par) * (m_right - m_left) * no_rpc))
        # if xi < 1 : xi = 1
        # if no_rp_set + xi > no_rpc:
        #     xi = no_rpc - no_rp_set
            # print('no_rpc reached')
        # print('xi =', xi)
        masses.append(m)
        xis.append(xi)
        no_rp_set += xi
        no_sp_set += 1
        m_left = m_right
        bins.append(m_right)
        bin_n += 1
    
    xis = np.array(xis)
    masses = np.array(masses)
    
    xi_max = xis.max()
    # print('xi_max =', f'{xi_max:.2e}')
    
    # xi_critmin = int(xi_max * eta)
    xi_critmin = xi_max * eta
    # if xi_critmin < 1: xi_critmin = 1
    # print('xi_critmin =', xi_critmin)
    
    valid_ind = []
    
    for bin_n,xi in enumerate(xis):
        if xi < xi_critmin:
            # print('')
            p = xi / xi_critmin
            if np.random.rand() >= p:
                xis[bin_n] = 0
            else:
                xis[bin_n] = xi_critmin
                valid_ind.append(bin_n)
        else: valid_ind.append(bin_n)
    
    # ind = np.nonzero(xis)
    # xis = xis[ind].astype(np.int64)
    valid_ind = np.array(valid_ind)
    xis = xis[valid_ind]
    masses = masses[valid_ind]
    
    # now, some bins are empty, thus the total number sum(xis) is not right
    # moreover, the total mass sum(masses*xis) is not right ...
    # FIRST: test, how badly the total number and total mass of the cell
    # is violated, if this is bad:
    # -> create a number of particles, such that the total number is right
    # and assign the average mass to it
    # then reweight all of the masses such that the total mass is right.
    
    return masses, xis, m_low, m_high, bins

@njit()
def generate_SIP_ensemble_expo_SingleSIP_weak_threshold_nonint2(
        par, no_rpc, r_critmin=0.6, m_high_by_m_low=1.0E6, kappa=40,
        eta=1.0E-9, seed=4711, setseed = True):
    bin_factor = 10**(1.0/kappa)
    m_low = mp.compute_mass_from_radius(r_critmin,
                                        c.mass_density_water_liquid_NTP)
    m_left = m_low
    l_max = int(kappa * np.log10(m_high_by_m_low)) + 1
    rnd = np.random.rand( l_max )
    
    masses = np.zeros(l_max, dtype = np.float64)
    xis = np.zeros(l_max, dtype = np.float64)
    bins = np.zeros(l_max+1, dtype = np.float64)
    bins[0] = m_left
    
    for l in range(l_max):
        m_right = m_left * bin_factor
        bins[l+1] = m_right
        dm = m_right - m_left
        
        m = m_left + dm * rnd[l]
        masses[l] = m
        xis[l] = no_rpc * dm * dist.dst_expo(m, par)
        
        m_left = m_right
    
    xi_max = xis.max()
    xi_critmin = xi_max * eta
    
    switch = np.ones(l_max, dtype=np.int64)
    
    for l in range(l_max):
        if xis[l] < xi_critmin:
            if np.random.rand() < xis[l] / xi_critmin:
                xis[l] = xi_critmin
            else: switch[l] = 0
    
    ind = np.nonzero(switch)[0]
    
    xis = xis[ind]
    masses = masses[ind]
    
    
    return masses, xis, m_low, bins


# no_spc is the intended number of super particles per cell,
# this will right on average, but will vary due to the random assigning 
# process of the xi_i
# m0, m1, dm: parameters for the numerical integration to find the cutoffs
# and for the numerical integration of integrals
# dV = volume size of the cell (in m^3)
# n0: initial particle number distribution (in 1/m^3)
# IN WORK: make the bin size smaller for low masses to get
# finer resolution here...
# -> 'steer against' the expo PDF for low m -> something in between
# we need higher resolution in very small radii (Unterstrasser has
# R_min = 0.8 mu
# at least need to go down to 1 mu (resolution there...)
# eps = parameter for the bin linear spreading:
# bin_size(m = m_high) = eps * bin_size(m = m_low)
# @njit()
def generate_SIP_ensemble_expo_my_xi_rnd(par, no_spc, no_rpc,
                                         total_mass_in_cell,
                                         p_min, p_max, eps,
                                         m0, m1, dm, seed, setseed = True):
    if setseed: np.random.seed(seed)
    
    m_low = 0.0
    m_high = dist.num_int_expo_impl_right_border(m_low,p_max, dm, par, 1.0E8)
    
    # the bin mean size must be adjusted by ~np.log10(eps)
    # to get to a SIP number in the cell of about the intended number
    # if the reweighting is not done, the SIP number will be much larger
    # due to small bins for low masses
    bin_size_mean = (m_high - m_low) / no_spc * np.log10(eps)
    
    a = bin_size_mean * (eps - 1) / (m_high * 0.5 * (eps + 1))
    b = bin_size_mean / (0.5 * (eps + 1))
    
    # the current position of the left bin border
    m_left = m_low
    
    # generate list of masses and multiplicities
    masses = []
    xis = []
    
    # no of particles placed:
    no_pt = 0
    
    # let f(m) be the PDF!! (m can be the mass or the radius, depending,
    # which distr. is given), NOTE that f(m) is not the number concentration..
    
    pt_n = 0
    while no_pt < no_rpc:
        ### i) determine the next xi_mean by
        # integrating xi_mean = no_rpc * int_(m_left)^(m_left+dm) dm f(m)
        # print(pt_n, 'm_left=', m_left)
        # m = m_left
        bin_size = a * m_left + b
        m_right = m_left + bin_size
        
        intl = dist.num_int_expo(m_left, m_right, par, steps=1.0E5)
        
        xi_mean = no_rpc * intl
        if xi_mean > 10:
            xi = np.random.normal(xi_mean, 0.2*xi_mean)
        else:
            xi = np.random.poisson(xi_mean)
        xi = int(math.ceil(xi))
        if xi <= 0: xi = 1
        if no_pt + xi >= no_rpc:
            xi = no_rpc - no_pt
            M_sys = np.sum( np.array(xis)*np.array(masses) )
            M_should = no_rpc * dist.num_int_expo_mean(0.0,m_left,par,1.0E7)
            masses = [m * M_should / M_sys for m in masses]
            # M = p_max * total_mass_in_cell - M
            M_diff = total_mass_in_cell - M_should
            if M_diff <= 0.0:
                M_diff = 1.0/par                
            mu = M_diff/xi
            if mu <= 1.02*masses[pt_n-1]:
                xi_sum = xi + xis[pt_n-1]
                m_sum = xi*mu + xis[pt_n-1] * masses[pt_n-1]
                xis[pt_n-1] = xi_sum
                masses[pt_n-1] = m_sum / xi_sum
                no_pt += xi
            else:                
                masses.append(mu)    
                xis.append(xi)
                no_pt += xi
                pt_n += 1            
        else:            
            ### iii) set the right right bin border
            # by no_rpc * int_(m_left)^m_right dm f(m) = xi
            
            m_right = dist.num_int_expo_impl_right_border(m_left, xi/no_rpc,
                                                          dm, par)
            
            intl = dist.num_int_expo_mean(m_left, m_right, par)
            
            mu = intl * no_rpc / xi
            masses.append(mu)
            xis.append(xi)
            no_pt += xi
            pt_n += 1
            m_left = m_right
        
        if m_left >= m_high and no_pt < no_rpc:

            xi = no_rpc - no_pt
            # last = True
            M_sys = np.sum( np.array(xis)*np.array(masses) )
            M_should = no_rpc * dist.num_int_expo_mean(0.0,m_left,par,1.0E7)
            masses = [m * M_should / M_sys for m in masses]
            # M = p_max * total_mass_in_cell - M
            M_diff = total_mass_in_cell - M_should
            if M_diff <= 0.0:
                M_diff = 1.0/par
            
            # mu = max(p_max*M/xi,m_left)
            mu = M_diff/xi
            if mu <= 1.02*masses[pt_n-1]:
                xi_sum = xi + xis[pt_n-1]
                m_sum = xi*mu + xis[pt_n-1] * masses[pt_n-1]
                xis[pt_n-1] = xi_sum
                masses[pt_n-1] = m_sum / xi_sum
                no_pt += xi
            else:                
                masses.append(mu)    
                xis.append(xi)
                no_pt += xi
                pt_n += 1
    
    return np.array(masses), np.array(xis, dtype=np.int64), m_low, m_high


