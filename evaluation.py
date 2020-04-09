#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TROPOS LAGRANGIAN CLOUD MODEL
Super-Droplet method in two-dimensional kinematic framework
(Test Case 1 ICMW 2012)
Author: Jan Bohrer (bohrer@tropos.de)
Further contact: Oswald Knoth (knoth@tropos.de)

DATA ANALYSIS, PROCESSING AND PREPARATION FOR PLOTTING

basic units:
particle mass, water mass, solute mass in femto gram = 10^-18 kg
particle radius in micro meter ("mu")
all other quantities in SI units
"""

#%% MODULE IMPORTS
import numpy as np
from numba import njit
import timeit

import constants as c
import microphysics as mp

from file_handling import load_grid_scalar_fields, load_particle_data_all

#%% RUNTIME OF FUNCTIONS

def compare_functions_run_time(functions, pars, rs, ns, globals_):
    """Analyzes the run time of several functions via timeit module

    Example (assume the functions in 'funcs' and the
             variables in 'pars' are defined)
    
    funcs = ["compute_r_l_grid_field_np", "compute_r_l_grid_field"]
    pars = ["m_w, xi, cells, grid.mixing_ratio_water_liquid,
            grid.mass_dry_inv", "m_w, xi, cells, r_l, m_dry_inv"]
    rs = [5,7]
    ns = [1000,10000]

    compare_functions_run_time(funcs, pars, rs, ns, globals_=globals())
    
    Parameters
    ----------
    functions: list of str
        List of the function names (strings)
        e.g. ["compute_r_l_grid_field", "compute_r_l_grid_field_np"]
    pars: list of str
        List of function parameter names, e.g.
        ["m_w, xi, cells, grid.mixing_ratio_water_liquid, grid.mass_dry_inv",
         "m_w, xi, cells, r_l, m_dry_inv"]
    rs: list of int
        List of 'repeats' for each function in the function list.
        Is used as 'repeat' argument in timeit.repeat().
        This argument specifies how many times timeit() is called per function
        e.g. rs = [5,5]
    ns: list of int
        List of 'numbers' for each function in the function list.
        Is used as 'number' argument in timeit.repeat().
        This argument specifies how many repetitions are executed
        per timeit() call
        e.g. ns = [100,10000]
    globals_: dict
        Dictionary of global variables which are required from the executed
        module. Usually called with globals_=globals().
        Note that one needs to call globals_=globals() explicitly.
        A default argument for globals_ cannot be given
        in the function definition, because in that case,
        the globals would be taken from module "evaluation.py" and
        not from the environment of the executed module.
    
    """
    
    # print (__name__)
    t = []
    for i,func in enumerate(functions):
        print(func + ": repeats =", rs[i], "no reps = ", ns[i])
    # print(globals_)
    for i,func in enumerate(functions):
        statement = func + "(" + pars[i] + ")"
        t_ = timeit.repeat(statement, repeat=rs[i],
                           number=ns[i], globals=globals_)
        t.append(t_)
        print("best = ", f"{min(t_)/ns[i]*1.0E6:.4}", "us;",
              "worst = ", f"{max(t_)/ns[i]*1.0E6:.4}", "us;",
              "mean =", f"{np.mean(t_)/ns[i]*1.0E6:.4}",
              "+-", f"{np.std(t_, ddof = 1)/ns[i]*1.0E6:.3}", "us" )

#%% UPDATE FUNCTIONS

@njit()
def update_mixing_ratio(mixing_ratio, m_w, xi, cells, mass_dry_inv, 
                        id_list, mask):
    """Updates the water mixing ratio with masked droplet characterization
    
    'mask' can be used to classify droplet categories. The 'mixing_ratio'
    will only include contributions from the particle IDs, where 'mask'
    is True.
    
    Parameters
    ----------
    mixing_ratio: ndarray, dtype=float
        2D array of the discretized water vapor mixing ratio for the
        chosen droplet category. This array will be filled with zeros and
        then updated corresponding to 'm_w', 'xi' and 'mass_dry_inv'
    m_w: ndarray, dtype=float
        1D array holding the particle water masses (1E-18 kg)
    xi: ndarray, dtype=float
        1D array holding the particle multiplicities      
    cells: ndarray, dtype=int
        2D array, holding the particle cell indices, i.e.
        cells[0] = 1D array of horizontal indices
        cells[1] = 1D array of vertical indices
        (cells[0,n], cells[1,n]) gives the cell of particle 'n'
    mass_dry_inv: ndarray, dtype=float
        2D array: 1 / mass_dry,
        where mass_dry = mass_density_air_dry * volume_cell
    id_list: ndarray, dtype=float
        1D array holding the ordered particle IDs.
        Other arrays, like 'm_w', 'm_s', 'xi' etc. refer to this list.
        I.e. 'm_w[n]' is the water mass of particle with ID 'id_list[n]'
    mask: ndarray, dtype=bool
        1D array, which masks the particle IDs given in id_list.
        If mask[i] = True, the particle ID in id_list[i] will be considered
        when calculating the mixing ratio in the cell of this particle
        
    """
    
    mixing_ratio.fill(0.0)
    for ID in id_list[mask]:
        mixing_ratio[cells[0,ID],cells[1,ID]] += m_w[ID] * xi[ID]
    mixing_ratio *= 1.0E-18 * mass_dry_inv  

@njit()
def update_number_concentration_per_dry_mass(conc, xi, cells, mass_dry_inv, 
                                             id_list, mask):
    """Updates the number concentration with masked droplet characterization
    
    'mask' can be used to classify droplet categories. The 'conc'
    will only include contributions from the particle IDs, where 'mask'
    is True.
    
    Parameters
    ----------
    conc: ndarray, dtype=float
        2D array of the discretized number concentration per dry mass (1/kg)
        for the chosen droplet category. This array will be filled with
        zeros and then updated corresponding to 'xi' and 'mass_dry_inv'
    xi: ndarray, dtype=float
        1D array holding the particle multiplicities      
    cells: ndarray, dtype=int
        2D array, holding the particle cell indices, i.e.
        cells[0] = 1D array of horizontal indices
        cells[1] = 1D array of vertical indices
        (cells[0,n], cells[1,n]) gives the cell of particle 'n'
    mass_dry_inv: ndarray, dtype=float
        2D array: 1 / mass_dry,
        where mass_dry = mass_density_air_dry * volume_cell
    id_list: ndarray, dtype=float
        1D array holding the ordered particle IDs.
        Other arrays, like 'm_w', 'm_s', 'xi' etc. refer to this list.
        I.e. 'm_w[n]' is the water mass of particle with ID 'id_list[n]'
    mask: ndarray, dtype=bool
        1D array, which masks the particle IDs given in id_list.
        If mask[i] = True, the particle ID in id_list[i] will be considered
        when calculating the number concentration in the cell of this particle
        
    """
    
    conc.fill(0.0)
    for ID in id_list[mask]:
        conc[cells[0,ID],cells[1,ID]] += xi[ID]
    conc *= mass_dry_inv 

@njit()
def update_T_p(grid_temp, cells, T_p):
    """Updates the particle temperatures to the atmos. grid cell temperatures
    
    This function is also defined in "integration.py".
    Repeated definition to avoid import with large overhead.
    
    Parameters
    ----------
    grid_temp: ndarray, dtype=float
        2D array holding the discretized temperature of the atmosphere
    cells: ndarray, dtype=int
        2D array, holding the particle cell indices, i.e.
        cells[0] = 1D array of horizontal indices
        cells[1] = 1D array of vertical indices
        (cells[0,n], cells[1,n]) gives the cell of particle 'n'
    T_p: ndarray, dtype=float
        1D array holding the particle temperatures sorted by particle IDs
    
    """
    
    for ID in range(len(T_p)):
        T_p[ID] = grid_temp[cells[0,ID],cells[1,ID]] 

#%% ANALYSIS OF GRID FIELDS

def generate_field_frame_data_avg(load_path_list,
                                  field_indices, time_indices,
                                  derived_indices,
                                  mass_dry_inv, grid_volume_cell,
                                  no_cells, solute_type):
    """Statistical analysis of the atmos. grid fields of multiple runs
    
    For the applied stochastic simulation model, it is strongly advised
    to conduct several independent runs with different initial conditions
    and random number seeds applied in the particle collisions.
    The simulation runs are identified by the initial random number seed.
    Each independent run results in a time series of 'grid frames', 
    where each 'grid frame' includes the discretized atmospheric fields of
    0: r_v = water vapor mixing ratio
    1: r_l = liquid water mixing ratio
    2: Theta = dry potential temperature
    3: T = temperature
    4: p = pressure
    5: S = saturation
    The grid frame time series is stored on hard disk for each seed (run).
    
    This function provides a statistical analysis, yielding the average
    values of chosen fields (over the independent simulation runs)
    as well as the standard deviation for each grid cell.
    Average values and standard deviation are calculated for a number of
    time steps, which can be chosen by 'time_indices'.

    The function is further used in 'generate_plot_data.py' to provide
    plotable data.

    Parameters
    ----------
    load_path_list: list of str
        List of 'load_paths' [load_path_0, load_path_1, ...], 
        where each 'load_path' provides the directory, where the data
        of a single simulation seed is stored. Each 'load_path' must be
        given in the format '/path/to/directory/' and is called later on by
        load_grid_scalar_fields(load_path, grid_save_times)
    field_indices: ndarray, dtype=int
        Choose, which atmospheric fields shall be included in the analysis
        by providing a list of indices, for which
        0: r_v = water vapor mixing ratio
        1: r_l = liquid water mixing ratio
        2: Theta = dry potential temperature
        3: T = temperature
        4: p = pressure
        5: S = saturation
    time_indices: ndarray, dtype=int
        Choose, which simulation times shall be included in the analysis.
        The grid data was stored on hard disk in certain time intervals.
        The format is something like data = [data_t0, data_t1, data_t2, ...]
        This 1D array of indices selects, which of the recorded times in
        'data' is included.
    derived_indices: ndarray, dtype=int
        From the stored grid data, a number of quantities can further be
        derived. This array of indices selects, which derived fields
        shall be included in the analysis. The indices correspond to
        0: r_aero = water mixing ratio of 'aerosols' < 0.5 microns
        1: r_cloud = water mixing ratio of 'cloud droplets' < 25 microns
        2: r_rain = water mixing ratio of 'rain drops' > 25 microns
        3: n_aero = number concentration of 'aerosols' < 0.5 microns
        4: n_c = number concentration of 'cloud droplets' < 25 microns
        5: n_r = number concentration of 'rain drops' > 25 microns
        6: R_avg = average droplet radius (microns)
        7: R_1/2 = 2nd moment / 1st moment of the radius distribution
        8: R_eff = 3rd moment/ 2nd moment of radius distribution
    mass_dry_inv: ndarray, dtype=float
        2D array: 1 / mass_dry,
        where mass_dry = mass_density_air_dry * volume_cell.
        The dry air mass is stationary in the given kinematic model.
    volume_cell: float
        Grid cell volume (m^3). All grid cells have the same volume.
    no_cells: ndarray, dtype=int
        no_cells[0] = number of grid cells in x (horizontal)
        no_cells[1] = number of grid cells in z (vertical)
    solute_type: str
        Particle solute material.
        Either 'AS' (ammonium sulfate) or 'NaCl' (sodium chloride)
        
    Returns
    -------
    fields_with_time: ndarray, dtype=float
        fields_with_time[it,n] = 2D array with the average over
        independent simulation runs of analyzed field 'n' at
        the time corresponding to index 'it' in save_times_out
        Included are the fields chosen by 'field_indices' (n = 0, .., N_c)
        and the derived fields chosen by 'derived_indices'
        (n = N_c+1, .., N_tot)
    fields_with_time_std: ndarray, dtype=float
        fields_with_time_std[it,n] = 2D array with the standard deviation
        from analysis over independent simulation runs of
        analyzed field 'n' at the time corresponding to
        index 'it' in save_times_out.
        Included are the fields chosen by 'field_indices' (n = 0, .., N_c)
        and the derived fields chosen by 'derived_indices'
        (n = N_c+1, .., N_tot)
    save_times_out: ndarray, dtype=float
        1D array of simulation times, where the fields where analyzed
    field_names_out: list of str
        List of strings with the names of the analyzed fields used for
        plotting.
    units_out: list of str
        List of strings with the names of the units of the analyzed
        fields used for plotting.
    scales_out: list of float
        List of scaling factors for the analyzed fields. The scaling
        factors are used for plotting to obtain appropriate units.
        
    """
    
    V0 = grid_volume_cell
    
    if solute_type == "AS":
        compute_R_p_w_s_rho_p = mp.compute_R_p_w_s_rho_p_AS
    elif solute_type == "NaCl":
        compute_R_p_w_s_rho_p = mp.compute_R_p_w_s_rho_p_NaCl
    
    bins_R_p_drop_classif = [0.5, 25.]
    
    field_names_orig = ["r_v", "r_l", "\Theta", "T", "p", "S"]
    scales_orig = [1000., 1000., 1, 1, 0.01, 1]
    units_orig = ["g/kg", "g/kg", "K", "K", "hPa", "-"]
    
    field_names_deri = ["r_\mathrm{aero}", "r_c", "r_r",
                       "n_\mathrm{aero}", "n_c", "n_r",
                       r"R_\mathrm{avg}", r"R_{2/1}", r"R_\mathrm{eff}"]
    units_deri = ["g/kg", "g/kg", "g/kg", "1/mg", "1/mg", "1/mg",
                  r"$\mathrm{\mu m}$",
                  r"$\mathrm{\mu m}$", r"$\mathrm{\mu m}$"]
    scales_deri = [1000., 1000., 1000., 1E-6, 1E-6, 1E-6, 1., 1., 1.]    
    
    no_seeds = len(load_path_list)
    no_times = len(time_indices)
    no_fields_orig = len(field_indices)
    no_fields_derived = len(derived_indices)
    no_fields = no_fields_orig + no_fields_derived
    
    fields_with_time = np.zeros( (no_times, no_fields,
                                  no_cells[0], no_cells[1]),
                                dtype = np.float64)
    fields_with_time_sq = np.zeros( (no_times, no_fields,
                                  no_cells[0], no_cells[1]),
                                dtype = np.float64)
    
    
    load_path = load_path_list[0]
    frame_every, no_grid_frames, dump_every = \
        np.load(load_path+"data_saving_paras.npy")
    grid_save_times = np.load(load_path+"grid_save_times.npy")
    
    save_times_out = np.zeros(no_times, dtype = np.int64)
    
    field_names_out = []
    units_out = []
    scales_out = []
    
    for cnt in range(no_fields_orig):
        idx_f = field_indices[cnt]
        field_names_out.append(field_names_orig[idx_f])
        units_out.append(units_orig[idx_f])
        scales_out.append(scales_orig[idx_f])
    
    for cnt in range(no_fields_derived):
        idx_f = derived_indices[cnt]
        field_names_out.append(field_names_deri[idx_f])
        units_out.append(units_deri[idx_f])
        scales_out.append(scales_deri[idx_f])         
    
    for time_n in range(no_times):
        idx_t = time_indices[time_n]
        save_times_out[time_n] = grid_save_times[idx_t]
    
    for seed_n, load_path in enumerate(load_path_list):
        
        fields = load_grid_scalar_fields(load_path, grid_save_times)
        vec_data, cells_with_time, scal_data, xi_with_time,\
        active_ids_with_time =\
            load_particle_data_all(load_path, grid_save_times)
        m_w_with_time = scal_data[:,0]
        m_s_with_time = scal_data[:,1]
        
        for cnt in range(no_fields_orig):
            idx_f = field_indices[cnt]
            fields_with_time[:,cnt] += fields[time_indices,idx_f]
            fields_with_time_sq[:,cnt] += \
                fields[time_indices,idx_f]*fields[time_indices,idx_f]
        
        for time_n in range(no_times):
            idx_t = time_indices[time_n]
            
            no_SIPs = len(xi_with_time[idx_t])
            T_p = np.zeros(no_SIPs, dtype = np.float64)
            id_list = np.arange(no_SIPs)
            update_T_p(fields[idx_t, 3], cells_with_time[idx_t], T_p)
            R_p, w_s, rho_p = \
                compute_R_p_w_s_rho_p(m_w_with_time[idx_t],
                                      m_s_with_time[idx_t], T_p)
            idx_R_p = np.digitize(R_p, bins_R_p_drop_classif)
            idx_classification = np.arange(3).reshape((3,1))
            
            masks_R_p = idx_classification == idx_R_p
                   
            fields_derived = np.zeros((no_fields_derived, no_cells[0],
                                       no_cells[1]),
                                       dtype = np.float64)
            
            mom0 = compute_moment_R_grid(0, R_p, xi_with_time[idx_t], V0,
                                         cells_with_time[idx_t],
                                         active_ids_with_time[idx_t],
                                         id_list, no_cells)
            mom1 = compute_moment_R_grid(1, R_p, xi_with_time[idx_t], V0,
                                         cells_with_time[idx_t],
                                         active_ids_with_time[idx_t],
                                         id_list, no_cells)
#            mom2 = compute_moment_R_grid(2, R_p, xi_with_time[idx_t], V0,
#                                         cells_with_time[idx_t],
#                                         active_ids_with_time[idx_t],
#                                         id_list, no_cells)
#            mom3 = compute_moment_R_grid(3, R_p, xi_with_time[idx_t], V0,
#                                         cells_with_time[idx_t],
#                                         active_ids_with_time[idx_t],
#                                         id_list, no_cells)
            
            # calculate R_eff only from cloud range (as Arabas 2015)
            mom1_cloud = compute_moment_R_grid(
                             1,
                             R_p[masks_R_p[1]],
                             xi_with_time[idx_t][masks_R_p[1]], V0,
                             cells_with_time[idx_t][:,masks_R_p[1]],
                             active_ids_with_time[idx_t][masks_R_p[1]],
                             id_list, no_cells)
            mom2_cloud = compute_moment_R_grid(
                             2,
                             R_p[masks_R_p[1]],
                             xi_with_time[idx_t][masks_R_p[1]], V0,
                             cells_with_time[idx_t][:,masks_R_p[1]],
                             active_ids_with_time[idx_t][masks_R_p[1]],
                             id_list, no_cells)
            mom3_cloud = compute_moment_R_grid(
                             3,
                             R_p[masks_R_p[1]],
                             xi_with_time[idx_t][masks_R_p[1]], V0,
                             cells_with_time[idx_t][:,masks_R_p[1]],
                             active_ids_with_time[idx_t][masks_R_p[1]],
                             id_list, no_cells)

            for cnt in range(no_fields_derived):
                idx_f = derived_indices[cnt]
                if idx_f < 6:
                    mask = np.logical_and(masks_R_p[idx_f%3],
                                          active_ids_with_time[idx_t])
                    if idx_f in range(3):
                        update_mixing_ratio(fields_derived[cnt],
                                            m_w_with_time[idx_t],
                                            xi_with_time[idx_t],
                                            cells_with_time[idx_t],
                                            mass_dry_inv,
                                            id_list, mask)
                    elif idx_f in range(3,6):
                        update_number_concentration_per_dry_mass(
                                fields_derived[cnt],
                                xi_with_time[idx_t],
                                cells_with_time[idx_t],
                                mass_dry_inv,
                                id_list, mask)
                elif idx_f == 6:
                    # R_mean
                    fields_derived[cnt] = np.where(mom0 == 0.0, 0.0, mom1/mom0)
                elif idx_f == 7:
                    # R_2/1
                    fields_derived[cnt] = np.where(mom1_cloud == 0.0, 0.0,
                                                   mom2_cloud/mom1_cloud)
                elif idx_f == 8:
                    # R_eff
                    fields_derived[cnt] = np.where(mom2_cloud == 0.0, 0.0,
                                                   mom3_cloud/mom2_cloud)
            
            fields_with_time[time_n,no_fields_orig:no_fields] += \
                fields_derived
            fields_with_time_sq[time_n,no_fields_orig:no_fields] += \
                fields_derived * fields_derived
    
    fields_with_time /= no_seeds
    
    fields_with_time_std =\
        np.sqrt((fields_with_time_sq
                 - no_seeds*fields_with_time*fields_with_time)\
                / (no_seeds * (no_seeds-1)) )
    
    return fields_with_time, fields_with_time_std, \
           save_times_out, field_names_out, units_out, \
           scales_out 

#%% BINNING OF SIPs:

# modified binning with different smoothing approaches
# masses is a list of [masses0, masses1, ..., masses_no_sims]
# where masses[i] = array of masses of a spec. SIP ensemble
# use moments_an[1] for LWC0
def generate_myHisto_SIP_ensemble_np(masses, xis, m_min, m_max,
                                     dV, DNC0, LWC0,
                                     no_bins, no_sims,
                                     bin_mode, spread_mode, scale_factor,
                                     shift_factor, overflow_factor):
    """Generation of mass based histograms with several analysis methods
    
    This function builds histograms of the simulation particle masses
    and yields the discretized concentration distribution per mass f_m 
    with integral f_m(m) dm = DNC = droplet number concentration (1/m^3)
    and the discretized density distribution per mass g_m
    with integral g_m(m) dm = LWC = liquid water content (kg/m^3).
    Currently, bins are divided with equal distance on the logarithmic
    mass axis. The masses of the super particles are collected in these
    bins. One can further choose, which mass value should correspond to
    each bin.
    For example, a bin is given by the borders m0 < m1. The collected
    value in the bin can be assigned to the bin center mc = (m0 + m1) / 2
    or to the bin center on a logarithmic axis
    log(mc) = (log(m0) + log(m1)) / 2. There are several other ways
    to assign the bins mass value, which are are denoted by 'smoothing'
    of the histograms and are defined in the code below.
    For the plots in the GMD publication, we use exclusively the
    bin center assignment (mc = (m0 + m1) / 2).
    Histograms are built for data from several independent simulation runs
    with the same mass bins. The resulting distribution functions are
    averaged over the independent simulation runs in each bin.
    
    Parameters
    ----------
    masses: ndarray, dtype=float
        1D array of SIP masses (unit = 1E-18 kg)
    xis: ndarray, dtype=float
        1D array of SIP multiplicities (real numbers, non-integer)
    m_min: float
        Defines the lower border of the bin with the smallest mass 
    m_max: float
        Defines the upper border of the bin with the largest mass 
    dV: float
        Grid cell volume (m^3)
    DNC0: float
        Initial droplet number concentration (1/m^3)
    LWC0: float
        Initial liquid water content (kg/m^3)
    no_bins: int
        Number of bins of the histograms
    no_sims: int
        Number of independent simulations
    bin_mode: int
        Method for SIP binning.
        Only avail. option: bin_mode=1 (bins equal distance on log. axis)
    spread_mode: int
        spreading mode of the smoothed histogram
        choose 0 (based on lin-scale) or 1 (based on log-scale)
    scale_factor: float
        scaling factor for the 1st correction of the smoothed histogram        
    shift_factor: float
        center shift factor for the 2nd correction of the smoothed histogram
    overflow_factor: float
        factor for artificial bins of the smoothed histogram
    
    Returns
    -------
    f_m_num_avg: ndarray, dtype=float
        1D array with the discretized concentration distribution per mass.
        (1/(kg m^3)).
        In each bin, the function was average over the independent
        simulation runs
    f_m_num_std: ndarray, dtype=float
        1D array with the standard deviation of the discretized
        concentration distribution per mass.
        In each bin, the standard deviation is evaluated by statistical
        analysis of the independent simulation runs.
    g_m_num_avg: ndarray, dtype=float
        1D array with the discretized density distribution per mass (1/m^3).
        In each bin, the function was average over the independent
        simulation runs
    g_m_num_std: ndarray, dtype=float
        1D array with the standard deviation of the discretized
        density distribution per mass.
        In each bin, the standard deviation is evaluated by statistical
        analysis of the independent simulation runs.
    h_m_num_avg: ndarray, dtype=float
        1D array with the discretized distribution h_m = f_m(m) * m^2
        (kg/m^3).
        In each bin, the function was average over the independent
        simulation runs
    h_m_num_std: ndarray, dtype=float
        1D array with the standard deviation of the discretized
        distribution h_m = f_m(m) * m^2.
        In each bin, the standard deviation is evaluated by statistical
        analysis of the independent simulation runs.
    bins_mass: ndarray, dtype=float
        1D array of the bin-borders. len(bins_mass) = no_bins + 1
    bins_mass_width: ndarray, dtype=float
        1D array with the widths of the mass bins
    bins_mass_centers: ndarray, dtype=float
        Array, collecting several 1D arrays with bin 'center' variations,
        meaning the mass values assigned to the bins.
        bins_mass_centers[0] = bins_mass_center_lin (center on lin scale)
        bins_mass_centers[1] = bins_mass_center_log (center on log scale)
        bins_mass_centers[2] = bins_mass_center_COM (center of mass
            for each bin)
        bins_mass_centers[3] = bins_mass_center_exact (exactly weighted
            mass value for exponential distr.)
        bins_mass_centers[4] = bins_mass_centers_lin_fit (mass value from
            linear smoothing)
        bins_mass_centers[5] = bins_mass_centers_qfit (mass value from
            quadratic smoothing)
        bins_mass_centers[6] = bins_mass_center_h_g (mass value
            corresponding to h_m / g_m)
    bins_mass_center_lin: ndarray, dtype=float
        the same as 'bins_mass_centers[0]', but including two values left
        and right of the binned area.
    lin_par: ndarray, dtype=float
        Parameters of the linear smoothing
    a_par: ndarray, dtype=float
        Parameters of the quadratic smoothing
    
    """
    
    if bin_mode == 1:
            bin_factor = (m_max/m_min)**(1.0/no_bins)
            bin_log_dist = np.log(bin_factor)
            # bin_log_dist_half = 0.5 * bin_log_dist
            # add dummy bins for overflow
            # bins_mass = np.zeros(no_bins+3,dtype=np.float64)
            bins_mass = np.zeros(no_bins+1,dtype=np.float64)
            bins_mass[0] = m_min
            # bins_mass[0] = m_min / bin_factor
            for bin_n in range(1,no_bins+1):
                bins_mass[bin_n] = bins_mass[bin_n-1] * bin_factor
            # the factor 1.01 is for numerical stability: to be sure
            # that m_max does not contribute to a bin larger than the
            # last bin
            bins_mass[-1] *= 1.0001
            # the factor 0.99 is for numerical stability: to be sure
            # that m_min does not contribute to a bin smaller than the
            # 0-th bin
            bins_mass[0] *= 0.9999
            # m_0 = m_min / np.sqrt(bin_factor)
            bins_mass_log = np.log(bins_mass)

    bins_mass_width = np.zeros(no_bins+2,dtype=np.float64)
    bins_mass_width[1:-1] = bins_mass[1:]-bins_mass[:-1]
    # modify for overflow bins
    bins_mass_width[0] = bins_mass_width[1]
    bins_mass_width[-1] = bins_mass_width[-2]
    dm0 = 0.5*bins_mass_width[0]
    dmN = 0.5*bins_mass_width[-1]
    # dm0 = 0.5*(bins_mass[0] - bins_mass[0] / bin_factor)
    # dmN = 0.5*(bins_mass[-1] * bin_factor - bins_mass[-1])

    f_m_num = np.zeros( (no_sims,no_bins+2), dtype=np.float64 )
    g_m_num = np.zeros( (no_sims,no_bins), dtype=np.float64 )
    h_m_num = np.zeros( (no_sims,no_bins), dtype=np.float64 )

    for i,mass in enumerate(masses):
        histo = np.zeros(no_bins+2, dtype=np.float64)
        histo_g = np.zeros(no_bins+2, dtype=np.float64)
        histo_h = np.zeros(no_bins+2, dtype=np.float64)
        mass_log = np.log(mass)
        for n,m_ in enumerate(mass):
            xi = xis[i][n]
            bin_n = np.nonzero(np.histogram(m_, bins=bins_mass)[0])[0][0]

            # smear functions depending on weight of data point in the bin
            # on a lin base
            if spread_mode == 0:
                norm_dist = (bins_mass[bin_n+1] - mass[n]) \
                            / bins_mass_width[bin_n]
            # on a log base
            elif spread_mode == 1:
                norm_dist = (bins_mass_log[bin_n] - mass_log[n])/bin_log_dist
            if norm_dist < 0.5:
                s = 0.5 + norm_dist

                # +1 because of overflow bins left and right in "histo"-array
                bin_n += 1
                # print(n,s,"right")
                histo[bin_n+1] += (1.0-s)*xi
                histo_g[bin_n+1] += (1.0-s)*xi*m_
                histo_h[bin_n+1] += (1.0-s)*xi*m_*m_
                # if in last bin: no outflow,
                # just EXTRAPOLATION to overflow bin!
                if bin_n == no_bins:
                    histo[bin_n] += xi
                    histo_g[bin_n] += xi*m_
                    histo_h[bin_n] += xi*m_*m_
                else:
                    histo[bin_n] += s*xi
                    histo_g[bin_n] += s*xi*m_
                    histo_h[bin_n] += s*xi*m_*m_
            elif spread_mode == 0:
                # now left side of bin
                norm_dist = (mass[n] - bins_mass[bin_n]) \
                            / bins_mass_width[bin_n-1]
                # +1 because of overflow bins left and right in "histo"-array
                bin_n += 1
                # print(n,norm_dist, "left")
                if norm_dist < 0.5:
                    s = 0.5 + norm_dist
                    # print(n,s,"left")
                    histo[bin_n-1] += (1.0-s)*xi
                    histo_g[bin_n-1] += (1.0-s)*xi*m_
                    histo_h[bin_n-1] += (1.0-s)*xi*m_*m_
                    # if in first bin: no outflow,
                    # just EXTRAPOLATION to overflow bin!
                    if bin_n == 1:
                        histo[bin_n] += xi
                        histo_g[bin_n] += xi*m_
                        histo_h[bin_n] += xi*m_*m_
                    else:
                        histo[bin_n] += s*xi
                        histo_g[bin_n] += s*xi*m_
                        histo_h[bin_n] += s*xi*m_*m_
                else:
                    histo[bin_n] += xi
                    histo_g[bin_n] += xi*m_
                    histo_h[bin_n] += xi*m_*m_
            elif spread_mode == 1:
                # +1 because of overflow bins left and right in "histo"-array
                bin_n += 1
                s = 1.5 - norm_dist
                histo[bin_n] += s*xi
                histo[bin_n-1] += (1.0-s)*xi
                histo_g[bin_n] += s*xi*m_
                histo_g[bin_n-1] += (1.0-s)*xi*m_
                histo_h[bin_n] += s*xi*m_*m_
                histo_h[bin_n-1] += (1.0-s)*xi*m_*m_

            # on a log base
            # log_dist = mass_log[n] - bins_mass_log[bin_n]
            # if log_dist < bin_log_dist_half:
            #     s = 0.5 + log_dist/bin_log_dist
            #     # print(n,s,"left")
            #     histo[bin_n] += s*xi
            #     histo[bin_n-1] += (1.0-s)*xi
            #     histo_g[bin_n] += s*xi*m_
            #     histo_g[bin_n-1] += (1.0-s)*xi*m_
            # else:
            #     s = 1.5 - log_dist/bin_log_dist
            #     # print(n,s,"right")
            #     histo[bin_n] += s*xi
            #     histo[bin_n+1] += (1.0-s)*xi
            #     histo_g[bin_n] += s*xi*m_
            #     histo_g[bin_n+1] += (1.0-s)*xi*m_

        f_m_num[i,1:-1] = histo[1:-1] / (bins_mass_width[1:-1] * dV)

        # multiply the overflow-bins by factor to get an estimation of
        # f_m at the position m_0 - dm0/2
        # f_m at the position m_no_bins + dmN/2, where
        # dm0 = 0.5*(bins_mass[0] - bins_mass[0] / bin_factor)
        # dmN = 0.5*(bins_mass[-1] * bin_factor - bins_mass[-1])
        f_m_num[i,0] = overflow_factor * histo[0] / (dm0 * dV)
        f_m_num[i,-1] = overflow_factor * histo[-1] / (dmN * dV)

        g_m_num[i] = histo_g[1:-1] / (bins_mass_width[1:-1] * dV)
        h_m_num[i] = histo_h[1:-1] / (bins_mass_width[1:-1] * dV)


    f_m_num_avg = np.average(f_m_num, axis=0)
    f_m_num_std = np.std(f_m_num, axis=0, ddof=1) / np.sqrt(no_sims)
    g_m_num_avg = np.average(g_m_num, axis=0)
    g_m_num_std = np.std(g_m_num, axis=0, ddof=1) / np.sqrt(no_sims)
    h_m_num_avg = np.average(h_m_num, axis=0)
    h_m_num_std = np.std(h_m_num, axis=0, ddof=1) / np.sqrt(no_sims)
    
    # define centers on lin scale
    bins_mass_center_lin = np.zeros(no_bins+2, dtype=np.float64)
    bins_mass_center_lin[1:-1] = 0.5 * (bins_mass[:-1] + bins_mass[1:])
    # add dummy bin centers for quadratic approx
    bins_mass_center_lin[0] = bins_mass[0] - 0.5*dm0
    bins_mass_center_lin[-1] = bins_mass[-1] + 0.5*dmN

    # define centers on the logarithmic scale
    bins_mass_center_log = bins_mass[:-1] * np.sqrt(bin_factor)
    
    # define the center of mass for each bin and set it as the "bin center"
    bins_mass_center_COM = g_m_num_avg / f_m_num_avg[1:-1]

    # def as 2nd moment/1st moment
    bins_mass_center_h_g = h_m_num_avg / g_m_num_avg

    ### LINEAR APPROX OF f_m
    # to get an idea of the shape
    # for bin n take f[n-1], f[n], f[n+1]
    # make linear approx from n-1 to n and from n to n+1
    # to get idea of shape of function
    # lin fct: f = a0 + a1*m
    # a1 = (f[n+1]-f[n])/(m[n+1] - m[n])
    # a0 = f[n] - a1*m[n]
    # bins_mass_centers_lin_fit = np.zeros(no_bins, dtype = np.float64)
    lin_par0 = np.zeros(no_bins+1, dtype = np.float64)
    lin_par1 = np.zeros(no_bins+1, dtype = np.float64)

    lin_par1 = (f_m_num_avg[1:] - f_m_num_avg[:-1]) \
               / (bins_mass_center_lin[1:] - bins_mass_center_lin[:-1])
    lin_par0 = f_m_num_avg[:-1] - lin_par1 * bins_mass_center_lin[:-1]

    f_bin_border = lin_par0 + lin_par1 * bins_mass
    # f_bin_border_delta_left = np.zeros(no_bins+1, dtype = np.float64)
    # f_bin_border_delta_left = np.abs(f_m_num_avg[1:-1]-f_bin_border[:-1])
    # f_bin_border_delta_right = np.abs(f_bin_border[1:] - f_m_num_avg[1:-1])

    ### FIRST CORRECTION:
    # by spreading over several bins the bins with higher f_avg
    # "loose" counts to bins with smaller f_avg
    # by a loss/gain analysis, one can estimate the lost counts
    # using the linear approximation of f_m(m) calc. above

    # delta of counts (estimated)
    delta_N = np.zeros(no_bins, dtype=np.float64)

    delta_N[1:-1] = 0.25 * bins_mass_width[1:-3] \
                    * ( f_m_num_avg[1:-3] - f_bin_border[1:-2] ) \
                    + 0.25 * bins_mass_width[2:-2] \
                      * ( -f_m_num_avg[2:-2] + f_bin_border[2:-1] ) \
                    + 0.083333333 \
                      * ( lin_par1[1:-2] * bins_mass_width[1:-3]**2
                          - lin_par1[2:-1] * bins_mass_width[2:-2]**2)
    # first bin: only exchange with the bin to the right
    delta_N[0] = 0.25 * bins_mass_width[1] \
                 * ( -f_m_num_avg[1] + f_bin_border[1] ) \
                 - 0.083333333 \
                   * ( lin_par1[1] * bins_mass_width[1]**2 )
    # last bin: only exchange with the bin to the left
    # bin_n = no_bins-1
    delta_N[no_bins-1] = 0.25 * bins_mass_width[no_bins-1] \
                         * (f_m_num_avg[no_bins-1] - f_bin_border[no_bins-1]) \
                         + 0.083333333 \
                           * ( lin_par1[no_bins-1]
                               * bins_mass_width[no_bins-1]**2 )
    scale = delta_N / (f_m_num_avg[1:-1] * bins_mass_width[1:-1])
    scale = np.where(scale < -0.9,
                     -0.9,
                     scale)
    scale *= scale_factor
    f_m_num_avg[1:-1] = f_m_num_avg[1:-1] / (1.0 + scale)
    f_m_num_avg[0] = f_m_num_avg[0] / (1.0 + scale[0])
    f_m_num_avg[-1] = f_m_num_avg[-1] / (1.0 + scale[-1])

    ## REPEAT LIN APPROX AFTER FIRST CORRECTION
    lin_par0 = np.zeros(no_bins+1, dtype = np.float64)
    lin_par1 = np.zeros(no_bins+1, dtype = np.float64)

    lin_par1 = (f_m_num_avg[1:] - f_m_num_avg[:-1]) \
                / (bins_mass_center_lin[1:] - bins_mass_center_lin[:-1])
    lin_par0 = f_m_num_avg[:-1] - lin_par1 * bins_mass_center_lin[:-1]

    f_bin_border = lin_par0 + lin_par1 * bins_mass

    ### SECOND CORRECTION:
    # try to estimate the position of m in the bin where f(m) = f_avg (of bin)
    # bin avg based on the linear approximations
    # NOTE that this is just to get an idea of the function FORM
    # f_bin_border_delta_left = np.zeros(no_bins+1, dtype = np.float64)
    f_bin_border_delta_left = np.abs(f_m_num_avg[1:-1]-f_bin_border[:-1])
    f_bin_border_delta_right = np.abs(f_bin_border[1:] - f_m_num_avg[1:-1])

    bins_mass_centers_lin_fit = np.zeros(no_bins, dtype = np.float64)

    f_avg2 = 0.25 * (f_bin_border[:-1] + f_bin_border[1:]) \
             + 0.5 * f_m_num_avg[1:-1]

    for bin_n in range(no_bins):
        if f_bin_border_delta_left[bin_n] >= f_bin_border_delta_right[bin_n]:
            m_c = (f_avg2[bin_n] - lin_par0[bin_n]) / lin_par1[bin_n]
        else:
            m_c = (f_avg2[bin_n] - lin_par0[bin_n+1]) / lin_par1[bin_n+1]

        # if f_bin_border_abs[bin_n] >= f_bin_border_abs[bin_n+1]:
        #     # take left side of current bin
        #     m_c = 0.5 * ( (bins_mass[bin_n] + 0.25*bins_mass_width[bin_n]) \
        #           + lin_par1[bin_n+1]/lin_par1[bin_n] \
        #             * (bins_mass[bin_n+1] - 0.25*bins_mass_width[bin_n]) \
        #           + (lin_par0[bin_n+1] - lin_par0[bin_n]))
        # else:
        #     m_c = 0.5 * ( lin_par1[bin_n]/lin_par1[bin_n+1] \
        #                   * (bins_mass[bin_n]+0.25*bins_mass_width[bin_n])\
        #                   + (bins_mass[bin_n+1]-0.25*bins_mass_width[bin_n])\
        #                   + (lin_par0[bin_n] - lin_par0[bin_n+1]) )
        # add additional shift because of two effects:
        # 1) adding xi-"mass" to bins with smaller f_avg
        # 2) wrong setting of "center" if f_avg[n] > f_avg[n+1]

        m_c = shift_factor * m_c \
              + bins_mass_center_lin[bin_n+1] * (1.0 - shift_factor)

        if m_c < bins_mass[bin_n]:
            m_c = bins_mass[bin_n]
        elif m_c > bins_mass[bin_n+1]:
            m_c = bins_mass[bin_n+1]

        bins_mass_centers_lin_fit[bin_n] = m_c
        # shift more to center: -> is covered by shift_factor=0.5
        # bins_mass_centers_lin_fit[bin_n] = \
        #     0.5 * (m_c + bins_mass_center_lin[bin_n+1])

    ### bin mass center quad approx: -->>> BIG ISSUES: no monoton. interpol.
    # possible for three given points with quadr. fct.
    # for every bin:
    # assume that the coordinate pairs are right with
    # (m_center_lin, f_avg)
    # approximate the function f_m(m) locally with a parabola to get
    # an estimate of the form of the function
    # assume this parabola in the bin and calculate bin_center_exact

    D_10 = bins_mass_center_lin[1:-1] - bins_mass_center_lin[0:-2]
    D_20 = bins_mass_center_lin[2:] - bins_mass_center_lin[0:-2]
    D_21 = bins_mass_center_lin[2:] - bins_mass_center_lin[1:-1]

    CD_10 = (bins_mass_center_lin[1:-1] + bins_mass_center_lin[0:-2])*D_10
    CD_20 = (bins_mass_center_lin[2:] + bins_mass_center_lin[0:-2])*D_20
    CD_21 = (bins_mass_center_lin[2:] + bins_mass_center_lin[1:-1])*D_21

    a2 = f_m_num_avg[2:]/(D_21*D_20) - f_m_num_avg[1:-1]/(D_21*D_10) \
         + f_m_num_avg[:-2]/(D_10*D_20)
    a1_a2 = (-f_m_num_avg[0:-2]*CD_21 + f_m_num_avg[1:-1]*CD_20
             - f_m_num_avg[2:]*CD_10  ) \
            / (f_m_num_avg[0:-2]*D_21 - f_m_num_avg[1:-1]*D_20
               + f_m_num_avg[2:]*D_10  )
    a1 = a2 * a1_a2
    a0 = f_m_num_avg[1:-1] - a1*bins_mass_center_lin[1:-1] \
         - a2*bins_mass_center_lin[1:-1]**2

    bins_mass_sq = bins_mass*bins_mass

    bins_mass_centers_qfit =\
        -0.5*a1_a2 \
        + np.sqrt( 0.25*(a1_a2)**2
                   + 0.5*a1_a2 * (bins_mass[:-1] + bins_mass[1:])
                   + 0.33333333 * (bins_mass_sq[:-1]
                                   + bins_mass[:-1]*bins_mass[1:]
                                   + bins_mass_sq[1:]) )

    bins_mass_center_lin2 = bins_mass_center_lin[1:-1]

    bins_mass_width = bins_mass_width[1:-1]
    
    # set the bin "mass centers" at the right spot for exponential dist
    # such that f_avg_i in bin in = f(mm_i), where mm_i is the "mass center"
    # use moments_an[1] for LWC0 if not given (e.g. for lognormal distr.)
    m_avg = LWC0 / DNC0
    bins_mass_center_exact = bins_mass[:-1]\
                             + m_avg * np.log(bins_mass_width\
          / (m_avg * (1-np.exp(-bins_mass_width/m_avg))))

    bins_mass_centers = np.array((bins_mass_center_lin2,
                                  bins_mass_center_log,
                                  bins_mass_center_COM,
                                  bins_mass_center_exact,
                                  bins_mass_centers_lin_fit,
                                  bins_mass_centers_qfit,
                                  bins_mass_center_h_g))

    return f_m_num_avg, f_m_num_std, g_m_num_avg, g_m_num_std,\
           h_m_num_avg, h_m_num_std, \
           bins_mass, bins_mass_width, \
           bins_mass_centers, bins_mass_center_lin, \
           np.array((lin_par0,lin_par1)), np.array((a0,a1,a2))

#%% PARTICLE SAMPLING FOR SIZE SPECTRA

# active ids not necessary: choose target cell and no_cells_x
# such that the region is included in the valid domain
# weights_out = xi/mass_dry_inv (in that respective cell) 
# weights_out in number/kg_dry_air
@njit()
def sample_masses_per_m_dry(m_w, m_s, xi, cells, id_list, grid_temperature,
                            grid_mass_dry_inv,
                            target_cell, no_cells_x, no_cells_z):
    """Method collects necessary data from a chosen evaluation volume
    
    To build droplet size spectra at a certain 'target_cell' of the grid,
    all super-particles of a chosen evaluation volume of
    'no_cells_x' * 'no_cells_z' centered at the 'target_cell' are
    included in the analysis.
    
    Parameters
    ----------
    m_w: ndarray, dtype=float
        1D array holding the particle water masses (1E-18 kg)
        This array gets updated by the function.
    m_s: ndarray, dtype=float
        1D array holding the particle solute masses (1E-18 kg)
    xi: ndarray, dtype=float
        1D array holding the particle multiplicities  
    cells: ndarray, dtype=int
        2D array, holding the particle cell indices, i.e.
        cells[0] = 1D array of horizontal indices
        cells[1] = 1D array of vertical indices
        (cells[0,n], cells[1,n]) gives the cell of particle 'n'
    id_list: ndarray, dtype=float
        1D array holding the ordered particle IDs.
        Other arrays, like 'm_w', 'm_s', 'xi' etc. refer to this list.
        I.e. 'm_w[n]' is the water mass of particle with ID 'id_list[n]'
    grid_temperature: ndarray, dtype=float
        2D array holding the discretized temperature of the atmosphere
    grid_mass_dry_inv: ndarray, dtype=float
        2D array holding 1/m_dry, where m_dry is the inverse
        discretized atmos. dry air mass in each cell (kg)
    target_cell: ndarray, dtype=int
        1D array holding two indices of the cell, which is the center of
        the evaluated volume of the grid
        target_cell[0] = horizontal index of the grid cell
        target_cell[1] = vertical index of the grid cell
    no_cells_x: int
        A volume of no_cells_x * no_cells_z grid cells centered at
        'target_cell' is included in the evaluation.
        Provide as uneven integer.
    no_cells_z: int
        A volume of no_cells_x * no_cells_z grid cells centered at
        'target_cell' is included in the evaluation.
        Provide as uneven integer.
    
    Returns
    -------
    m_w_out: ndarray, dtype=float
        1D array holding the water masses of all particles in the evaluation
        volume
    m_s_out: ndarray, dtype=float
        1D array holding the solute masses of all particles in the evaluation
        volume
    xi_out: ndarray, dtype=float
        1D array holding the multiplicities of all particles in the evaluation
        volume
    weights_out: ndarray, dtype=float
        1D array holding the ratio of multiplicities and atmospheric dry
        air mass of the grid cells for all particles in the evaluation volume
    no_cells_eval: int
        Number of grid cells included in the evaluation volume
    
    """
        
    dx = no_cells_x // 2
    dz = no_cells_z // 2
    
    i_low = target_cell[0] - dx
    i_high = target_cell[0] + dx
    j_low = target_cell[1] - dz
    j_high = target_cell[1] + dz
    
    no_cells_eval = (dx * 2 + 1) * (dz * 2 + 1)
    
    mask =   ((cells[0] >= i_low) & (cells[0] <= i_high)) \
           & ((cells[1] >= j_low) & (cells[1] <= j_high))
    
    no_masses = mask.sum()
    
    m_s_out = np.zeros(no_masses, dtype = np.float64)
    m_w_out = np.zeros(no_masses, dtype = np.float64)
    xi_out = np.zeros(no_masses, dtype = np.float64)
    weights_out = np.zeros(no_masses, dtype = np.float64)
    T_p = np.zeros(no_masses, dtype = np.float64)
    
    for cnt, ID in enumerate(id_list[mask]):
        m_s_out[cnt] = m_s[ID]
        m_w_out[cnt] = m_w[ID]
        xi_out[cnt] = xi[ID]
        weights_out[cnt] = xi[ID] * grid_mass_dry_inv[cells[0,ID], cells[1,ID]]
        T_p[cnt] = grid_temperature[ cells[0,ID], cells[1,ID] ]
    
    return m_w_out, m_s_out, xi_out, weights_out, T_p, no_cells_eval

# weights_out in number/kg_dry_air
# we always assume the only quantities stored are m_s, m_w, xi
def sample_radii_per_m_dry(m_w, m_s, xi, cells, solute_type, id_list,
                           grid_temperature, grid_mass_dry_inv,
                           target_cell, no_cells_x, no_cells_z):
    """Method collects necessary data from a chosen evaluation volume
    
    To build droplet size spectra at a certain 'target_cell' of the grid,
    all super-particles of a chosen evaluation volume of
    'no_cells_x' * 'no_cells_z' centered at the 'target_cell' are
    included in the analysis.
    
    Parameters
    ----------
    m_w: ndarray, dtype=float
        1D array holding the particle water masses (1E-18 kg)
        This array gets updated by the function.
    m_s: ndarray, dtype=float
        1D array holding the particle solute masses (1E-18 kg)
    xi: ndarray, dtype=float
        1D array holding the particle multiplicities  
    cells: ndarray, dtype=int
        2D array, holding the particle cell indices, i.e.
        cells[0] = 1D array of horizontal indices
        cells[1] = 1D array of vertical indices
        (cells[0,n], cells[1,n]) gives the cell of particle 'n'
    solute_type: str
        Particle solute material.
        Either 'AS' (ammonium sulfate) or 'NaCl' (sodium chloride)
    id_list: ndarray, dtype=float
        1D array holding the ordered particle IDs.
        Other arrays, like 'm_w', 'm_s', 'xi' etc. refer to this list.
        I.e. 'm_w[n]' is the water mass of particle with ID 'id_list[n]'
    grid_temperature: ndarray, dtype=float
        2D array holding the discretized temperature of the atmosphere
    grid_mass_dry_inv: ndarray, dtype=float
        2D array holding 1/m_dry, where m_dry is the inverse
        discretized atmos. dry air mass in each cell (kg)
    target_cell: ndarray, dtype=int
        1D array holding two indices of the cell, which is the center of
        the evaluated volume of the grid
        target_cell[0] = horizontal index of the grid cell
        target_cell[1] = vertical index of the grid cell
    no_cells_x: int
        A volume of no_cells_x * no_cells_z grid cells centered at
        'target_cell' is included in the evaluation.
        Provide as uneven integer.
    no_cells_z: int
        A volume of no_cells_x * no_cells_z grid cells centered at
        'target_cell' is included in the evaluation.
        Provide as uneven integer.
    
    Returns
    -------
    R_p: ndarray, dtype=float
        1D array holding the particle radii of all particles in the evaluation
        volume
    R_s: ndarray, dtype=float
        1D array holding the dry particle radii of all particles
        in the evaluation volume
    xi_out: ndarray, dtype=float
        1D array holding the multiplicities of all particles in the evaluation
        volume
    weights_out: ndarray, dtype=float
        1D array holding the ratio of multiplicities and atmospheric dry
        air mass of the grid cells for all particles in the evaluation volume
    no_cells_eval: int
        Number of grid cells included in the evaluation volume
    
    """
    
    m_w_out, m_s_out, xi_out, weights_out, T_p, no_cells_eval = \
        sample_masses_per_m_dry(m_w, m_s, xi, cells, id_list, grid_temperature,
                                grid_mass_dry_inv,
                                target_cell, no_cells_x, no_cells_z)
    if solute_type == "AS":
        mass_density_dry = c.mass_density_AS_dry
        compute_R_p_w_s_rho_p = mp.compute_R_p_w_s_rho_p_AS
    elif solute_type == "NaCl":
        mass_density_dry = c.mass_density_NaCl_dry
        compute_R_p_w_s_rho_p = mp.compute_R_p_w_s_rho_p_NaCl
    
    R_s = mp.compute_radius_from_mass_vec(m_s_out, mass_density_dry)
    R_p, w_s, rho_p = compute_R_p_w_s_rho_p(m_w_out, m_s_out, T_p)
    
    return R_p, R_s, xi_out, weights_out, no_cells_eval        

#%% SIZE SPECTRA GENERATION

# for one seed only for now
# load_path_list = [[load_path0]] 
# target_cell_list = [ [tgc1], [tgc2], ... ]; tgc1 = [i1, j1]
# ind_time = [it1, it2, ..] = ind. of save times belonging to tgc1, tgc2, ...
# -> to create one cycle cf with particle trajectories
# grid scalar fields have been saved in this order on hard disk
# 0 = r_v
# 1 = r_l
# 2 = Theta    
# 3 = T
# 4 = p
# 5 = S    
def generate_size_spectra_R(load_path_list,
                            ind_time,
                            grid_mass_dry_inv,
                            grid_no_cells,
                            solute_type,
                            target_cell_list,
                            no_cells_x, no_cells_z,
                            no_bins_R_p, no_bins_R_s):
    """


    Parameters
    ----------
    load_path_list: list of str
        List of 'load_paths' [load_path_0, load_path_1, ...], 
        where each 'load_path' provides the directory, where the data
        of a single simulation seed is stored. Each 'load_path' must be
        given in the format '/path/to/directory/' and is called later on by
        load_grid_scalar_fields(load_path, grid_save_times) and
        load_particle_data_all(load_path, grid_save_times)
    ind_time: ndarray, dtype=int
        One size spectrum is generated for each grid target cell.
        Grid and particle data were stored in time intervals collected in
        'grid_save_times'. 'grid_save_times' is loaded from hard disk.
        When generating the spectrum of target cell number 'n',
        it=ind_time[n] holds the time index corresponding to the
        simulation time 'grid_save_times[it]', at which the spectrum
        is evaluated.
    grid_mass_dry_inv: ndarray, dtype=float
        2D array holding 1/m_dry, where m_dry is the inverse
        discretized atmos. dry air mass in each cell (kg)
    grid_no_cells: ndarray, dtype=int
        grid_no_cells[0] = number of grid cells in x (horizontal)
        grid_no_cells[1] = number of grid cells in z (vertical)
    solute_type: str
        Particle solute material.
        Either 'AS' (ammonium sulfate) or 'NaCl' (sodium chloride)
    target_cell_list: ndarray, dtype=int
        Collects all target cells, for which a size spectrum shall be
        generated.
        target_cell_list[n] =
        1D array holding two indices of the cell, which is the center of
        the evaluated volume of the grid
        target_cell_list[n][0] = horizontal index of the grid cell
        target_cell_list[n][1] = vertical index of the grid cell
    no_cells_x: int
        A volume of no_cells_x * no_cells_z grid cells centered at
        'target_cell' is included in the evaluation.
        Provide as uneven integer.
    no_cells_z: int
        A volume of no_cells_x * no_cells_z grid cells centered at
        'target_cell' is included in the evaluation.
        Provide as uneven integer.
    no_bins_R_p: int
        Number of bins for the particle radius histograms
    no_bins_R_p: int
        Number of bins for the particle dry radius histograms
        
    Returns
    -------
    f_R_p_list: ndarray, dtype=float
    
    
    """
    
        
    no_seeds = len(load_path_list)
    no_times = len(ind_time)
    no_tg_cells = len(target_cell_list[0])
    
    load_path = load_path_list[0]
    frame_every, no_grid_frames, dump_every = \
        np.load(load_path+"data_saving_paras.npy")
    grid_save_times = np.load(load_path+"grid_save_times.npy")
    
    save_times_out = np.zeros(no_times, dtype = np.int64)
    
    R_p_list = []
    R_s_list = []
    weights_list = []
    R_min_list =[]    
    R_max_list =[]    
    
    grid_r_l_list = np.zeros( (no_times, grid_no_cells[0], grid_no_cells[1]),
                        dtype = np.float64)
    
    for tg_cell_n in range(no_tg_cells):
        R_p_list.append([])
        R_s_list.append([])
        weights_list.append([])
        save_times_out[tg_cell_n] = grid_save_times[ind_time[tg_cell_n]]
        
    for seed_n, load_path in enumerate(load_path_list):
        fields = load_grid_scalar_fields(load_path, grid_save_times)
        grid_temperature_with_time = fields[:,3]
        
        grid_r_l_with_time = fields[:,1]
        
        vec_data, cells_with_time, scal_data,\
        xi_with_time, active_ids_with_time =\
            load_particle_data_all(load_path, grid_save_times)
        m_w_with_time = scal_data[:,0]
        m_s_with_time = scal_data[:,1]    
        
        for tg_cell_n in range(no_tg_cells):
            target_cell = target_cell_list[:,tg_cell_n]
            idx_t = ind_time[tg_cell_n]
            
            
            id_list = np.arange(len(xi_with_time[idx_t]))
            
            R_p_tg, R_s_tg, xi_tg, weights_tg, no_cells_eval = \
                sample_radii_per_m_dry(m_w_with_time[idx_t],
                                       m_s_with_time[idx_t],
                                       xi_with_time[idx_t],
                                       cells_with_time[idx_t],
                                       solute_type, id_list,
                                       grid_temperature_with_time[idx_t],
                                       grid_mass_dry_inv,
                                       target_cell, no_cells_x, no_cells_z)
                
            R_p_list[tg_cell_n].append(R_p_tg)
            R_s_list[tg_cell_n].append(R_s_tg)
            weights_list[tg_cell_n].append(weights_tg)
            
            grid_r_l_list[tg_cell_n] += grid_r_l_with_time[idx_t]
    
    grid_r_l_list /= no_seeds
        
    f_R_p_list = np.zeros( (no_tg_cells, no_seeds, no_bins_R_p),
                          dtype = np.float64 )
    f_R_s_list = np.zeros( (no_tg_cells, no_seeds, no_bins_R_s),
                          dtype = np.float64 )
    bins_R_p_list = np.zeros( (no_tg_cells, no_bins_R_p+1),
                             dtype = np.float64 )
    bins_R_s_list = np.zeros( (no_tg_cells, no_bins_R_s+1),
                             dtype = np.float64 )
    
    for tg_cell_n in range(no_tg_cells):
            
        R_p_min = np.amin(np.concatenate(R_p_list[tg_cell_n]))
        R_p_max = np.amax(np.concatenate(R_p_list[tg_cell_n]))
        
        R_min_list.append(R_p_min)
        R_max_list.append(R_p_max)
        
        R_s_min = np.amin(np.concatenate(R_s_list[tg_cell_n]))
        R_s_max = np.amax(np.concatenate(R_s_list[tg_cell_n]))
        
        R_min_factor = 0.5
        R_max_factor = 2.
        
        bins_R_p = np.logspace(np.log10(R_p_min*R_min_factor),
                               np.log10(R_p_max*R_max_factor), no_bins_R_p+1 )
        
        bins_R_p_list[tg_cell_n] = np.copy(bins_R_p)
        
        bins_width_R_p = bins_R_p[1:] - bins_R_p[:-1]

        bins_R_s = np.logspace(np.log10(R_s_min*R_min_factor),
                               np.log10(R_s_max*R_max_factor), no_bins_R_s+1)
    
        bins_width_R_s = bins_R_s[1:] - bins_R_s[:-1]

        bins_R_s_list[tg_cell_n] = np.copy(bins_R_s)
        
        for seed_n in range(no_seeds):
            R_p_tg = R_p_list[tg_cell_n][seed_n]
            R_s_tg = R_s_list[tg_cell_n][seed_n]
            weights_tg = weights_list[tg_cell_n][seed_n]
    
            h_p, b_p = np.histogram(R_p_tg, bins_R_p, weights= weights_tg)

            # convert from 1/(kg*micrometer) to unit 1/(milligram*micro_meter)
            # (per dry air mass and per particle radius)
            f_R_p_list[tg_cell_n, seed_n] =\
                1E-6 * h_p / bins_width_R_p / no_cells_eval
        
            h_s, b_s = np.histogram(R_s_tg, bins_R_s, weights= weights_tg)
            
            # convert from 1/(kg*micrometer) to unit 1/(milligram*micro_meter)
            # (per dry air mass and per particle radius)
            f_R_s_list[tg_cell_n, seed_n] =\
                1E-6 * h_s / bins_width_R_s / no_cells_eval
    
    return f_R_p_list, f_R_s_list, bins_R_p_list, bins_R_s_list, \
           save_times_out, grid_r_l_list, R_min_list, R_max_list

#%% MOMENT ANALYSIS

@njit()
# V0 = volume grid cell
def compute_moment_R_grid(n, R_p, xi, V0,
                          cells, active_ids, id_list, no_cells):
    moment = np.zeros( (no_cells[0], no_cells[1]), dtype = np.float64 )
    if n == 0:
        for ID in id_list[active_ids]:
            moment[cells[0,ID],cells[1,ID]] += xi[ID]
    else:
        for ID in id_list[active_ids]:
            moment[cells[0,ID],cells[1,ID]] += xi[ID] * R_p[ID]**n
    return moment / V0

def avg_moments_over_boxes(
        moments_vs_time_all_seeds, no_seeds, idx_t, no_moments,
        target_cells_x, target_cells_z,
        no_cells_per_box_x, no_cells_per_box_z):
    no_times_eval = len(idx_t)
    no_target_cells_x = len(target_cells_x)
    no_target_cells_z = len(target_cells_z)
    di_cell = no_cells_per_box_x // 2
    dj_cell = no_cells_per_box_z // 2    
    moments_at_boxes_all_seeds = np.zeros( (no_seeds,no_times_eval,no_moments,
                                  no_target_cells_x,no_target_cells_z),
                                 dtype = np.float64)

    for seed_n in range(no_seeds):
        for time_n, time_ind in enumerate(idx_t):
            for mom_n in range(no_moments):
                for box_n_x, tg_cell_x in enumerate(target_cells_x):
                    for box_n_z , tg_cell_z in enumerate(target_cells_z):
                        moment_box = 0.
                        i_tg_corner = tg_cell_x - di_cell
                        j_tg_corner = tg_cell_z - dj_cell
                        cells_box_x = np.arange(i_tg_corner,
                                                i_tg_corner+no_cells_per_box_x)
                        cells_box_z = np.arange(j_tg_corner,
                                                j_tg_corner+no_cells_per_box_z)
                        MG = np.meshgrid(cells_box_x, cells_box_z)
                        
                        cells_box_x = MG[0].flatten()
                        cells_box_z = MG[1].flatten()
                        
                        moment_box = moments_vs_time_all_seeds[seed_n, time_n,
                                                               mom_n,
                                                               cells_box_x,
                                                               cells_box_z]
                        
                        moment_box = np.average(moment_box)
                        moments_at_boxes_all_seeds[seed_n, time_n, mom_n,
                                                   box_n_x, box_n_z] = \
                            moment_box
    return moments_at_boxes_all_seeds 

def generate_moments_avg_std(load_path_list,
                             no_moments, time_indices,
                             grid_volume_cell,
                             no_cells, solute_type):

    if solute_type == "AS":
        compute_R_p_w_s_rho_p = mp.compute_R_p_w_s_rho_p_AS
    elif solute_type == "NaCl":
        compute_R_p_w_s_rho_p = mp.compute_R_p_w_s_rho_p_NaCl

    no_seeds = len(load_path_list)
    no_times = len(time_indices)
    
    moments_vs_time_all_seeds = np.zeros( (no_seeds, no_times, no_moments,
                                  no_cells[0], no_cells[1]),
                                dtype = np.float64)
    
    load_path = load_path_list[0]
    frame_every, no_grid_frames, dump_every = \
        np.load(load_path+"data_saving_paras.npy")
    grid_save_times = np.load(load_path+"grid_save_times.npy")
    
    save_times_out = np.zeros(no_times, dtype = np.int64)
    
    for time_n in range(no_times):
        idx_t = time_indices[time_n]        
        save_times_out[time_n] = grid_save_times[idx_t]
    
    V0 = grid_volume_cell
    
    for seed_n, load_path in enumerate(load_path_list):
        
        fields = load_grid_scalar_fields(load_path, grid_save_times)
        vec_data,cells_with_time,scal_data,xi_with_time,active_ids_with_time =\
            load_particle_data_all(load_path, grid_save_times)
        m_w_with_time = scal_data[:,0]
        m_s_with_time = scal_data[:,1]
        
        for time_n in range(no_times):
            idx_t = time_indices[time_n]
            
            no_SIPs = len(xi_with_time[idx_t])
            T_p = np.zeros(no_SIPs, dtype = np.float64)
            id_list = np.arange(no_SIPs)
            update_T_p(fields[idx_t, 3], cells_with_time[idx_t], T_p)
            R_p, w_s, rho_p = \
                compute_R_p_w_s_rho_p(m_w_with_time[idx_t],
                                      m_s_with_time[idx_t], T_p)
            
            for mom_n in range(no_moments):
            
                moments_vs_time_all_seeds[seed_n, time_n, mom_n] =\
                    compute_moment_R_grid(mom_n, R_p, xi_with_time[idx_t], V0,
                                             cells_with_time[idx_t],
                                             active_ids_with_time[idx_t],
                                             id_list, no_cells)

    return moments_vs_time_all_seeds, save_times_out
