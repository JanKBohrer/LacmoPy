#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 15:06:04 2020

@author: bohrer
"""

### COMPUTE CHANGE Delta m_liquid and gamma (=mass rate)
# Newton method with no_iter iterations, the derivative is calculated only once
def compute_dml_and_gamma_impl_Newton_lin_NaCl_np(
        dt_sub, no_iter, m_w, m_s, w_s, R_p, T_p, rho_p,
        T_amb, p_amb, S_amb, e_s_amb, L_v, K, D_v, sigma_w):
    
    m_w_effl = m_s * (w_s_max_NaCl_inv - 1.0)
    gamma0, dgamma_dm = mp.compute_mass_rate_and_derivative_NaCl(
                            m_w, m_s, w_s, R_p, T_p, rho_p,
                            T_amb, p_amb, S_amb, e_s_amb,
                            L_v, K, D_v, sigma_w)
    dt_sub_times_dgamma_dm = dt_sub * dgamma_dm
    denom_inv = np.where(dt_sub_times_dgamma_dm < 0.9,
                         1.0 / (1.0 - dt_sub_times_dgamma_dm),
                         np.ones_like(dt_sub_times_dgamma_dm)*10.0)
     
    mass_new = np.maximum(m_w_effl, m_w + dt_sub * gamma0 * denom_inv)
    
    for cnt in range(no_iter-1):
        m_p = mass_new + m_s
        w_s = m_s / m_p
        rho = mat.compute_density_NaCl_solution(w_s, T_p)
        R = mp.compute_radius_from_mass(m_p, rho)
        gamma = mp.compute_mass_rate_NaCl(
                    mass_new, m_s, w_s, R, T_p, rho,
                    T_amb, p_amb, S_amb, e_s_amb, L_v, K, D_v, sigma_w)
                    
        mass_new += ( dt_sub * gamma + m_w - mass_new) * denom_inv
        mass_new = np.maximum( m_w_effl, mass_new )
        
    return mass_new - m_w, gamma0
compute_dml_and_gamma_impl_Newton_lin_NaCl =\
    njit()(compute_dml_and_gamma_impl_Newton_lin_NaCl_np)

# Full Newton method with no_iter iterations,
# the derivative is calculated every iteration
def compute_dml_and_gamma_impl_Newton_full_NaCl_np(
        dt_sub, no_iter_impl_mass, m_w, m_s, w_s, R_p, T_p, rho_p,
        T_amb, p_amb, S_amb, e_s_amb, L_v, K, D_v, sigma_p):
    m_w_effl = m_s * (w_s_max_NaCl_inv - 1.0)
    
    gamma0, dgamma_dm = mp.compute_mass_rate_and_derivative_NaCl(
                            m_w, m_s, w_s, R_p, T_p, rho_p,
                            T_amb, p_amb, S_amb, e_s_amb,
                            L_v, K, D_v, sigma_p)
    dt_sub_times_dgamma_dm = dt_sub * dgamma_dm
    denom_inv = np.where(dt_sub_times_dgamma_dm < 0.9,
                         1.0 / (1.0 - dt_sub_times_dgamma_dm),
                         np.ones_like(dt_sub_times_dgamma_dm) * 10.0)
     
    mass_new = np.maximum(m_w_effl, m_w + dt_sub * gamma0 * denom_inv)
    
    for cnt in range(no_iter_impl_mass-1):
        m_p = mass_new + m_s
        w_s = m_s / m_p
        rho = mat.compute_density_NaCl_solution(w_s, T_p)
        R = mp.compute_radius_from_mass(m_p, rho)
        gamma, dgamma_dm = mp.compute_mass_rate_and_derivative_NaCl(
                               mass_new, m_s, w_s, R, T_p, rho,
                               T_amb, p_amb, S_amb, e_s_amb,
                               L_v, K, D_v, sigma_p)
                               
        dt_sub_times_dgamma_dm = dt_sub * dgamma_dm
        denom_inv = np.where(dt_sub_times_dgamma_dm < 0.9,
                             1.0 / (1.0 - dt_sub_times_dgamma_dm),
                     np.ones_like(dt_sub_times_dgamma_dm) * 10.0)
        mass_new += ( dt_sub * gamma + m_w - mass_new) * denom_inv
        mass_new = np.maximum( m_w_effl, mass_new )
        
    return mass_new - m_w, gamma0
compute_dml_and_gamma_impl_Newton_full_NaCl =\
njit()(compute_dml_and_gamma_impl_Newton_full_NaCl_np)

w_s_max_AS_inv = 1. / mat.w_s_max_AS
# Newton method with no_iter iterations, the derivative is calculated only once
def compute_dml_and_gamma_impl_Newton_lin_AS_np(
        dt_sub, no_iter, m_w, m_s, w_s, R_p, T_p, rho_p,
        T_amb, p_amb, S_amb, e_s_amb, L_v, K, D_v, sigma_p):
    
    m_w_effl = m_s * (w_s_max_AS_inv - 1.0)
    gamma0, dgamma_dm = mp.compute_mass_rate_and_derivative_AS(
                            m_w, m_s, w_s, R_p, T_p, rho_p,
                                           T_amb, p_amb, S_amb, e_s_amb,
                                           L_v, K, D_v, sigma_p)
    dt_sub_times_dgamma_dm = dt_sub * dgamma_dm
    denom_inv = np.where(dt_sub_times_dgamma_dm < 0.9,
                         1.0 / (1.0 - dt_sub_times_dgamma_dm),
                         np.ones_like(dt_sub_times_dgamma_dm)*10.0)
     
    mass_new = np.maximum(m_w_effl, m_w + dt_sub * gamma0 * denom_inv)
    
    for cnt in range(no_iter-1):
        m_p = mass_new + m_s
        w_s = m_s / m_p
        rho = mat.compute_density_AS_solution(w_s, T_p)
        R = mp.compute_radius_from_mass(m_p, rho)
        gamma = mp.compute_mass_rate_AS(w_s, R, T_p, rho,
                         T_amb, p_amb, S_amb, e_s_amb, L_v, K, D_v, sigma_p)
                    
        mass_new += ( dt_sub * gamma + m_w - mass_new) * denom_inv
        mass_new = np.maximum( m_w_effl, mass_new )
        
    return mass_new - m_w, gamma0
compute_dml_and_gamma_impl_Newton_lin_AS =\
    njit()(compute_dml_and_gamma_impl_Newton_lin_AS_np)

# Full Newton method with no_iter iterations,
# the derivative is calculated every iteration
def compute_dml_and_gamma_impl_Newton_full_AS_np(
        dt_sub, no_iter_impl_mass, m_w, m_s, w_s, R_p, T_p, rho_p,
        T_amb, p_amb, S_amb, e_s_amb, L_v, K, D_v, sigma_p):
    m_w_effl = m_s * (w_s_max_AS_inv - 1.0)
    
    gamma0, dgamma_dm = mp.compute_mass_rate_and_derivative_AS(
            m_w, m_s, w_s, R_p, T_p, rho_p,
            T_amb, p_amb, S_amb, e_s_amb,
            L_v, K, D_v, sigma_p)
    dt_sub_times_dgamma_dm = dt_sub * dgamma_dm
    denom_inv = np.where(dt_sub_times_dgamma_dm < 0.9,
                         1.0 / (1.0 - dt_sub_times_dgamma_dm),
                         np.ones_like(dt_sub_times_dgamma_dm) * 10.0)
     
    mass_new = np.maximum(m_w_effl, m_w + dt_sub * gamma0 * denom_inv)
    
    for cnt in range(no_iter_impl_mass-1):
        m_p = mass_new + m_s
        w_s = m_s / m_p
        rho = mat.compute_density_AS_solution(w_s, T_p)
        R = mp.compute_radius_from_mass(m_p, rho)
        sigma = mat.compute_surface_tension_AS(w_s,T_p)
        gamma, dgamma_dm = mp.compute_mass_rate_and_derivative_AS(
                               mass_new, m_s, w_s, R, T_p, rho,
                               T_amb, p_amb, S_amb, e_s_amb,
                               L_v, K, D_v, sigma)
                               
        dt_sub_times_dgamma_dm = dt_sub * dgamma_dm
        denom_inv = np.where(dt_sub_times_dgamma_dm < 0.9,
                             1.0 / (1.0 - dt_sub_times_dgamma_dm),
                     np.ones_like(dt_sub_times_dgamma_dm) * 10.0)
        mass_new += ( dt_sub * gamma + m_w - mass_new) * denom_inv
        mass_new = np.maximum( m_w_effl, mass_new )
        
    return mass_new - m_w, gamma0
compute_dml_and_gamma_impl_Newton_full_AS =\
njit()(compute_dml_and_gamma_impl_Newton_full_AS_np)



#%%

# runtime test:
# no_spt = 400
# update_m_w_and_delta_m_l_impl_Newton_np: repeats = 7 no reps =  100
# update_m_w_and_delta_m_l_impl_Newton: repeats = 7 no reps =  1000
# update_m_w_and_delta_m_l_impl_Newton_par: repeats = 7 no reps =  1000
# best =  2.856e+03 us; worst =  6.707e+03 us; mean = 3.427e+03 +- 1.45e+03 us
# best =  390.7 us; worst =  751.5 us; mean = 443.3 +- 1.36e+02 us
# best =  391.3 us; worst =  721.1 us; mean = 439.3 +- 1.24e+02 us
# no_spt = 112500
# update_m_w_and_delta_m_l_impl_Newton_np: repeats = 7 no reps =  100
# update_m_w_and_delta_m_l_impl_Newton: repeats = 7 no reps =  1000
# update_m_w_and_delta_m_l_impl_Newton_par: repeats = 7 no reps =  1000
# best =  8.1e+04 us; worst =  8.393e+04 us; mean = 8.213e+04 +- 9.13e+02 us
# best =  1.102e+04 us; worst =  1.136e+04 us; mean = 1.115e+04 +- 1.56e+02 us
# best =  1.103e+04 us; worst =  1.142e+04 us; mean = 1.114e+04 +- 1.29e+02 us            
def update_m_w_and_delta_m_l_impl_Newton_NaCl_np(
        grid_temperature, grid_pressure, grid_saturation,
        grid_saturation_pressure, grid_thermal_conductivity, 
        grid_diffusion_constant, grid_heat_of_vaporization,
        grid_surface_tension, cells, m_w, m_s, xi, id_list, active_ids,
        R_p, w_s, rho_p, T_p, 
        delta_m_l, delta_Q_p, dt_sub,  no_iter_impl_mass):
    delta_m_l.fill(0.0)
    ### ACTIVATE
    # delta_Q_p.fill(0.0)
#    for ID, xi_ in enumerate(xi):
#        if xi_ != 0:
    for ID in id_list[active_ids]:
        cell = (cells[0,ID], cells[1,ID])
        T_amb = grid_temperature[cell]
        p_amb = grid_pressure[cell]
        S_amb = grid_saturation[cell]
        e_s_amb = grid_saturation_pressure[cell]
        # rho_f_amb = grid.mass_density_fluid[cell]
        L_v = grid_heat_of_vaporization[cell]
        K = grid_thermal_conductivity[cell]
        D_v = grid_diffusion_constant[cell]
        # sigma w is right now calc. with the ambient temperature...
        # can be changed to the particle temp, if tracked
        sigma_p = grid_surface_tension[cell]
#        sigma_p = compute_surface_tension grid_surface_tension[cell]
        
        # c_p_f = grid.specific_heat_capacity[cell]
        # mu_f = grid.viscosity[cell]
        
        # req. w_s, R_p, rho_p, T_p (check)
        dm, gamma = compute_dml_and_gamma_impl_Newton_full_NaCl(
                        dt_sub, no_iter_impl_mass, m_w[ID], m_s[ID],
                        w_s[ID], R_p[ID],
                        T_p[ID], rho_p[ID],
                        T_amb, p_amb, S_amb, e_s_amb, L_v, K, D_v, sigma_p)
        # ### 3.
        # ### ACTIVATE
        # T_eq_old = particle.equilibrium_temperature

        # ### 4.
        # # T_eq_new req. radius, which req. density,
        # which req. self.temperature, self.mass_fraction_solute
        # ### ACTIVATE
        # particle.equilibrium_temperature =\
        #   T_amb + L_v * mass_rate * 1.0E-12\
        #           / (4.0 * np.pi * particle.radius * K)

        # ### 5.
        # ### ACTIVATE
        # delta_Q_p[cell] += particle.compute_heat_capacity()\
        #                    * particle.multiplicity \
        #                    * particle.mass\
        #                    * (particle.equilibrium_temperature - T_eq_old)

        ### 6.
        m_w[ID] += dm

        ### 7. 
        delta_m_l[cell] += dm * xi[ID]
        
        ### 8.
        ### ACTIVATE
        # delta_Q_p[cell] += particle.compute_heat_capacity()\
        #                    * delta_m * particle.multiplicity\
        #                    * (particle.equilibrium_temperature - T_amb)    
    delta_m_l *= 1.0E-18
update_m_w_and_delta_m_l_impl_Newton_NaCl =\
    njit()(update_m_w_and_delta_m_l_impl_Newton_NaCl_np)

# runtime test:
# no_spt = 400
# update_m_w_and_delta_m_l_impl_Newton_np: repeats = 7 no reps =  100
# update_m_w_and_delta_m_l_impl_Newton: repeats = 7 no reps =  1000
# update_m_w_and_delta_m_l_impl_Newton_par: repeats = 7 no reps =  1000
# best =  2.856e+03 us; worst =  6.707e+03 us; mean = 3.427e+03 +- 1.45e+03 us
# best =  390.7 us; worst =  751.5 us; mean = 443.3 +- 1.36e+02 us
# best =  391.3 us; worst =  721.1 us; mean = 439.3 +- 1.24e+02 us
# no_spt = 112500
# update_m_w_and_delta_m_l_impl_Newton_np: repeats = 7 no reps =  100
# update_m_w_and_delta_m_l_impl_Newton: repeats = 7 no reps =  1000
# update_m_w_and_delta_m_l_impl_Newton_par: repeats = 7 no reps =  1000
# best =  8.1e+04 us; worst =  8.393e+04 us; mean = 8.213e+04 +- 9.13e+02 us
# best =  1.102e+04 us; worst =  1.136e+04 us; mean = 1.115e+04 +- 1.56e+02 us
# best =  1.103e+04 us; worst =  1.142e+04 us; mean = 1.114e+04 +- 1.29e+02 us            
def update_m_w_and_delta_m_l_impl_Newton_AS_np(
        grid_temperature, grid_pressure, grid_saturation,
        grid_saturation_pressure, grid_thermal_conductivity, 
        grid_diffusion_constant, grid_heat_of_vaporization,
        grid_surface_tension, cells, m_w, m_s, xi, id_list, active_ids,
        R_p, w_s, rho_p, T_p, 
        delta_m_l, delta_Q_p, dt_sub,  no_iter_impl_mass):
    delta_m_l.fill(0.0)
    ### ACTIVATE
    # delta_Q_p.fill(0.0)
#    for ID, xi_ in enumerate(xi):
#        if xi_ != 0:
    for ID in id_list[active_ids]:
        cell = (cells[0,ID], cells[1,ID])
        T_amb = grid_temperature[cell]
        p_amb = grid_pressure[cell]
        S_amb = grid_saturation[cell]
        e_s_amb = grid_saturation_pressure[cell]
        # rho_f_amb = grid.mass_density_fluid[cell]
        L_v = grid_heat_of_vaporization[cell]
        K = grid_thermal_conductivity[cell]
        D_v = grid_diffusion_constant[cell]
        # sigma w is right now calc. with the ambient temperature...
        # can be changed to the particle temp, if tracked
#        sigma_p = grid_surface_tension[cell]
        sigma_p = mat.compute_surface_tension_AS(w_s[ID], T_p[ID])
        
        # c_p_f = grid.specific_heat_capacity[cell]
        # mu_f = grid.viscosity[cell]
        
        # req. w_s, R_p, rho_p, T_p (check)
        dm, gamma = compute_dml_and_gamma_impl_Newton_full_AS(
                        dt_sub, no_iter_impl_mass, m_w[ID], m_s[ID],
                        w_s[ID], R_p[ID],
                        T_p[ID], rho_p[ID],
                        T_amb, p_amb, S_amb, e_s_amb, L_v, K, D_v, sigma_p)
        # ### 3.
        # ### ACTIVATE
        # T_eq_old = particle.equilibrium_temperature

        # ### 4.
        # # T_eq_new req. radius, which req. density,
        # which req. self.temperature, self.mass_fraction_solute
        # ### ACTIVATE
        # particle.equilibrium_temperature =\
        #   T_amb + L_v * mass_rate * 1.0E-12\
        #           / (4.0 * np.pi * particle.radius * K)

        # ### 5.
        # ### ACTIVATE
        # delta_Q_p[cell] += particle.compute_heat_capacity()\
        #                    * particle.multiplicity \
        #                    * particle.mass\
        #                    * (particle.equilibrium_temperature - T_eq_old)

        ### 6.
        m_w[ID] += dm

        ### 7. 
        delta_m_l[cell] += dm * xi[ID]
        
        ### 8.
        ### ACTIVATE
        # delta_Q_p[cell] += particle.compute_heat_capacity()\
        #                    * delta_m * particle.multiplicity\
        #                    * (particle.equilibrium_temperature - T_amb)    
    delta_m_l *= 1.0E-18
update_m_w_and_delta_m_l_impl_Newton_AS =\
    njit()(update_m_w_and_delta_m_l_impl_Newton_AS_np)


#%%
    
# dt_sub_pos is ONLY for the particle location step x_new = x_old + v*dt_sub_loc
# -->> give the opportunity to adjust this step, usually dt_sub_pos = dt_sub
# at the end of subloop 2: dt_sub_pos = dt_sub_half
# UPDATE 25.02.19: use full Newton implicit for delta_m
# UPDATE 26.02: use one step implicit for velocity and mass,
# for velocity use m_(n+1) and v_n in |u-v_n| for calcing the Reynolds number
# IN WORK: NOTE THAT dt_sub_half is not used anymore...
# grid_scalar_fields = np.array([T, p, Theta, rho_dry, r_v, r_l, S, e_s])
# grid_mat_prop = np.array([K, D_v, L, sigma_w, c_p_f, mu_f, rho_f])
def propagate_particles_subloop_step_NaCl_np(grid_scalar_fields, grid_mat_prop,
                                        grid_velocity,
                                        grid_no_cells, grid_ranges, grid_steps,
                                        pos, vel, cells, rel_pos, m_w, m_s, xi,
                                        water_removed, id_list, active_ids,
                                        T_p,
                                        delta_m_l, delta_Q_p,
                                        dt_sub, dt_sub_pos,
                                        no_iter_impl_mass, g_set):
                                        # ,ind):
    # removed_ids_step = []
    # delta_m_l.fill(0.0)
    ### 1. T_p = T_f
    # use this in vectorized version, because the subfunction compute_radius ..
    # in compute_R_p_w_s_rho_p (below) is defined by vectorize() and not by jit
    # update_T_p(grid_scalar_fields[ind["T"]], cells, T_p)
    # update_T_p(grid_scalar_fields[idx_T], cells, T_p)
    # update_T_p(grid_scalar_fields[i_T], cells, T_p)
    update_T_p(grid_scalar_fields[0], cells, T_p)
    # update_T_p(grid_scalar_fields[i_T], cells, T_p)
    # update_T_p(grid_scalar_fields[ind[0]], cells, T_p)
    # update_T_p(grid_scalar_fields[0], cells, T_p)
    
    # not possible with numba: getitem<> from array with (tuple(int64) x 2)
    # T_p = grid_scalar_fields[i.T][cells[0],cells[1]]
    
    ### 2. to 8. compute mass rate and delta_m and update m_w
    # use this in vectorized version, because the subfunction compute_radius ..
    # is defined by vectorize() and not by jit
    R_p, w_s, rho_p = mp.compute_R_p_w_s_rho_p_NaCl(m_w, m_s, T_p)
    
    # update_m_w_and_delta_m_l_impl_Newton(
    update_m_w_and_delta_m_l_impl_Newton_NaCl(
        grid_scalar_fields[0], grid_scalar_fields[1],
        grid_scalar_fields[6], grid_scalar_fields[7],
        grid_mat_prop[0], grid_mat_prop[1],
        grid_mat_prop[2], grid_mat_prop[3],
        cells, m_w, m_s, xi, id_list, active_ids, R_p, w_s, rho_p, T_p,
        delta_m_l, delta_Q_p, dt_sub, no_iter_impl_mass)
    # update_m_w_and_delta_m_l_impl_Newton(
    #     grid_scalar_fields[ind["T"]], grid_scalar_fields[ind["p"]],
    #     grid_scalar_fields[ind["S"]], grid_scalar_fields[ind["es"]],
    #     grid_mat_prop[ind["K"]], grid_mat_prop[ind["Dv"]],
    #     grid_mat_prop[ind["L"]], grid_mat_prop[ind["sigmaw"]],
    #     cells, m_w, m_s, xi, R_p, w_s, rho_p, T_p,
    #     delta_m_l, delta_Q_p, dt_sub, no_iter_impl_mass)
    
    # update_m_w_and_delta_m_l_impl_Newton(
    #     grid_scalar_fields[i.T], grid_scalar_fields[i.p],
    #     grid_scalar_fields[i.S],
    #     grid_scalar_fields[i.es], grid_mat_prop[i.K], 
    #     grid_mat_prop[i.Dv], grid_mat_prop[i.L],
    #     grid_mat_prop[i.sigmaw], cells, m_w, m_s, xi, R_p, w_s, rho_p, T_p,
    #     delta_m_l, delta_Q_p, dt_sub, no_iter_impl_mass)
    
    ### 9. v_n -> v_n+1
    # (CHANGED TO FULL TIMESTEP WITH k_d(m_(n+1), x_(n+1/2), |u_n+1/2 - v_n|) )
    # req. R_p -> self.density + m_p (changed)
    # -> mass_fraction (changed) + temperature (changed)
    # req. cell (check, unchanged) + location (check, unchanged)
    # use this in vectorized version, because the subfunction compute_radius...
    # is defined by vectorize() and not by jit
    R_p, w_s, rho_p = mp.compute_R_p_w_s_rho_p_NaCl(m_w, m_s, T_p)
    ### ACTIVATE
    # particle.temperature = particle.equilibrium_temperature
    # g_set >= 0 !!
    # update_vel_impl(
    update_vel_impl(vel, cells, rel_pos, xi, id_list, active_ids,
                    R_p, rho_p, grid_velocity,
                    grid_mat_prop[5], grid_mat_prop[6], 
                    grid_no_cells, g_set, dt_sub)
    # update_vel_impl(vel, cells, rel_pos, xi, R_p, rho_p, grid_velocity,
    #                 grid_mat_prop[ind["muf"]], grid_mat_prop[ind["rhof"]], 
    #                 grid_no_cells, g_set, dt_sub)
    ### 10.
    update_pos_from_vel_BC_PS(m_w, pos, vel, xi, cells,
                              water_removed, id_list,
                              active_ids,
                              grid_ranges, grid_steps, dt_sub_pos)

    # update_pos_from_vel_BC_PS(pos, vel, xi, grid_ranges, dt_sub_pos)
    ### 11.
    update_cells_and_rel_pos(pos, cells, rel_pos, active_ids,
                             grid_ranges, grid_steps)
propagate_particles_subloop_step_NaCl = \
    njit()(propagate_particles_subloop_step_NaCl_np)


# dt_sub_pos is ONLY for the particle location step x_new = x_old + v*dt_sub_loc
# -->> give the opportunity to adjust this step, usually dt_sub_pos = dt_sub
# at the end of subloop 2: dt_sub_pos = dt_sub_half
# UPDATE 25.02.19: use full Newton implicit for delta_m
# UPDATE 26.02: use one step implicit for velocity and mass,
# for velocity use m_(n+1) and v_n in |u-v_n| for calcing the Reynolds number
# IN WORK: NOTE THAT dt_sub_half is not used anymore...
# grid_scalar_fields = np.array([T, p, Theta, rho_dry, r_v, r_l, S, e_s])
# grid_mat_prop = np.array([K, D_v, L, sigma_w, c_p_f, mu_f, rho_f])
def propagate_particles_subloop_step_AS_np(grid_scalar_fields, grid_mat_prop,
                                        grid_velocity,
                                        grid_no_cells, grid_ranges, grid_steps,
                                        pos, vel, cells, rel_pos, m_w, m_s, xi,
                                        water_removed, id_list, active_ids,
                                        T_p,
                                        delta_m_l, delta_Q_p,
                                        dt_sub, dt_sub_pos,
                                        no_iter_impl_mass, g_set):
                                        # ,ind):
    # removed_ids_step = []
    # delta_m_l.fill(0.0)
    ### 1. T_p = T_f
    # use this in vectorized version, because the subfunction compute_radius ..
    # in compute_R_p_w_s_rho_p (below) is defined by vectorize() and not by jit
    # update_T_p(grid_scalar_fields[ind["T"]], cells, T_p)
    # update_T_p(grid_scalar_fields[idx_T], cells, T_p)
    # update_T_p(grid_scalar_fields[i_T], cells, T_p)
    update_T_p(grid_scalar_fields[0], cells, T_p)
    # update_T_p(grid_scalar_fields[i_T], cells, T_p)
    # update_T_p(grid_scalar_fields[ind[0]], cells, T_p)
    # update_T_p(grid_scalar_fields[0], cells, T_p)
    
    # not possible with numba: getitem<> from array with (tuple(int64) x 2)
    # T_p = grid_scalar_fields[i.T][cells[0],cells[1]]
    
    ### 2. to 8. compute mass rate and delta_m and update m_w
    # use this in vectorized version, because the subfunction compute_radius ..
    # is defined by vectorize() and not by jit
    R_p, w_s, rho_p = mp.compute_R_p_w_s_rho_p_AS(m_w, m_s, T_p)
    
    # update_m_w_and_delta_m_l_impl_Newton(
    update_m_w_and_delta_m_l_impl_Newton_AS(
        grid_scalar_fields[0], grid_scalar_fields[1],
        grid_scalar_fields[6], grid_scalar_fields[7],
        grid_mat_prop[0], grid_mat_prop[1],
        grid_mat_prop[2], grid_mat_prop[3],
        cells, m_w, m_s, xi, id_list, active_ids, R_p, w_s, rho_p, T_p,
        delta_m_l, delta_Q_p, dt_sub, no_iter_impl_mass)
    # update_m_w_and_delta_m_l_impl_Newton(
    #     grid_scalar_fields[ind["T"]], grid_scalar_fields[ind["p"]],
    #     grid_scalar_fields[ind["S"]], grid_scalar_fields[ind["es"]],
    #     grid_mat_prop[ind["K"]], grid_mat_prop[ind["Dv"]],
    #     grid_mat_prop[ind["L"]], grid_mat_prop[ind["sigmaw"]],
    #     cells, m_w, m_s, xi, R_p, w_s, rho_p, T_p,
    #     delta_m_l, delta_Q_p, dt_sub, no_iter_impl_mass)
    
    # update_m_w_and_delta_m_l_impl_Newton(
    #     grid_scalar_fields[i.T], grid_scalar_fields[i.p],
    #     grid_scalar_fields[i.S],
    #     grid_scalar_fields[i.es], grid_mat_prop[i.K], 
    #     grid_mat_prop[i.Dv], grid_mat_prop[i.L],
    #     grid_mat_prop[i.sigmaw], cells, m_w, m_s, xi, R_p, w_s, rho_p, T_p,
    #     delta_m_l, delta_Q_p, dt_sub, no_iter_impl_mass)
    
    ### 9. v_n -> v_n+1
    # (CHANGED TO FULL TIMESTEP WITH k_d(m_(n+1), x_(n+1/2), |u_n+1/2 - v_n|) )
    # req. R_p -> self.density + m_p (changed)
    # -> mass_fraction (changed) + temperature (changed)
    # req. cell (check, unchanged) + location (check, unchanged)
    # use this in vectorized version, because the subfunction compute_radius...
    # is defined by vectorize() and not by jit
    R_p, w_s, rho_p = mp.compute_R_p_w_s_rho_p_AS(m_w, m_s, T_p)
    ### ACTIVATE
    # particle.temperature = particle.equilibrium_temperature
    # g_set >= 0 !!
    # update_vel_impl(
    update_vel_impl(vel, cells, rel_pos, xi, id_list, active_ids,
                    R_p, rho_p, grid_velocity,
                    grid_mat_prop[5], grid_mat_prop[6], 
                    grid_no_cells, g_set, dt_sub)
    # update_vel_impl(vel, cells, rel_pos, xi, R_p, rho_p, grid_velocity,
    #                 grid_mat_prop[ind["muf"]], grid_mat_prop[ind["rhof"]], 
    #                 grid_no_cells, g_set, dt_sub)
    ### 10.
    update_pos_from_vel_BC_PS(m_w, pos, vel, xi, cells,
                              water_removed, id_list,
                              active_ids,
                              grid_ranges, grid_steps, dt_sub_pos)

    # update_pos_from_vel_BC_PS(pos, vel, xi, grid_ranges, dt_sub_pos)
    ### 11.
    update_cells_and_rel_pos(pos, cells, rel_pos, active_ids,
                             grid_ranges, grid_steps)
propagate_particles_subloop_step_AS = \
    njit()(propagate_particles_subloop_step_AS_np)    



#%%
    
# integrated a number of no_col_steps collision steps in here
# if 
def integrate_subloop_w_col_n_steps_np(
            grid_scalar_fields, grid_mat_prop, grid_velocity,
            grid_no_cells, grid_ranges, grid_steps,
            grid_volume_cell, p_ref, p_ref_inv,
            pos, vel, cells, rel_pos, m_w, m_s, xi,
            dt_col_over_dV, E_col_grid, no_kernel_bins,
            R_kernel_low_log, bin_factor_R_log, no_cols,
            water_removed,
            id_list, active_ids, T_p,
            delta_m_l, delta_Q_p, delta_Theta_ad, delta_r_v_ad,
            dt_sub, dt_sub_pos, no_cond_steps, no_col_steps,
            no_iter_impl_mass, g_set, solute_type):
    
    
    if no_col_steps == 1:
        collision_step_Long_Bott_Ecol_grid_R_all_cells_2D_multicomp_np(
            xi, m_w, m_s, vel, grid_scalar_fields[0], cells, grid_no_cells,
            dt_col_over_dV, E_col_grid, no_kernel_bins,
            R_kernel_low_log, bin_factor_R_log, no_cols, solute_type)
    
    no_col_steps_larger_one = no_col_steps > 1
    
    # d) subloop 1
    # for n_h = 0, ..., N_h-1
    if solute_type == "NaCl":
        for n_sub in range(no_cond_steps):
            if no_col_steps_larger_one:
                collision_step_Long_Bott_Ecol_grid_R_all_cells_2D_multicomp_np(
                    xi, m_w, m_s, vel, grid_scalar_fields[0], cells, grid_no_cells,
                    dt_col_over_dV, E_col_grid, no_kernel_bins,
                    R_kernel_low_log, bin_factor_R_log, no_cols, solute_type)
            # i) for all particles
            # updates delta_m_l and delta_Q_p
            propagate_particles_subloop_step_NaCl(
                grid_scalar_fields, grid_mat_prop,
                grid_velocity,
                grid_no_cells, grid_ranges, grid_steps,
                pos, vel, cells, rel_pos, m_w, m_s, xi,
                water_removed, id_list, active_ids,
                T_p,
                delta_m_l, delta_Q_p,
                dt_sub, dt_sub_pos,
                no_iter_impl_mass, g_set)
    
            # ii) to vii)
            propagate_grid_subloop_step(grid_scalar_fields, grid_mat_prop,
                                            p_ref, p_ref_inv,
                                            delta_Theta_ad, delta_r_v_ad,
                                            delta_m_l, delta_Q_p,
                                            grid_volume_cell)
            
            # viii) to ix) included in "propagate_particles_subloop"
            # delta_Q_p.fill(0.0)
            # delta_m_l.fill(0.0)
            
    elif solute_type == "AS":
        for n_sub in range(no_cond_steps):
            if no_col_steps_larger_one:
                collision_step_Long_Bott_Ecol_grid_R_all_cells_2D_multicomp_np(
                    xi, m_w, m_s, vel, grid_scalar_fields[0], cells, grid_no_cells,
                    dt_col_over_dV, E_col_grid, no_kernel_bins,
                    R_kernel_low_log, bin_factor_R_log, no_cols, solute_type)
            # i) for all particles
            # updates delta_m_l and delta_Q_p
            propagate_particles_subloop_step_AS(
                grid_scalar_fields, grid_mat_prop,
                grid_velocity,
                grid_no_cells, grid_ranges, grid_steps,
                pos, vel, cells, rel_pos, m_w, m_s, xi,
                water_removed, id_list, active_ids,
                T_p,
                delta_m_l, delta_Q_p,
                dt_sub, dt_sub_pos,
                no_iter_impl_mass, g_set)
    
            # ii) to vii)
            propagate_grid_subloop_step(grid_scalar_fields, grid_mat_prop,
                                            p_ref, p_ref_inv,
                                            delta_Theta_ad, delta_r_v_ad,
                                            delta_m_l, delta_Q_p,
                                            grid_volume_cell)
            
            # viii) to ix) included in "propagate_particles_subloop"
            # delta_Q_p.fill(0.0)
            # delta_m_l.fill(0.0)
        
        # tau += dt_sub
    # subloop 1 end    
    # put the additional col step here, if no_col_steps > no_cond_steps
    if no_col_steps > no_cond_steps:
        collision_step_Long_Bott_Ecol_grid_R_all_cells_2D_multicomp_np(
            xi, m_w, m_s, vel, grid_scalar_fields[0], cells, grid_no_cells,
            dt_col_over_dV, E_col_grid, no_kernel_bins,
            R_kernel_low_log, bin_factor_R_log, no_cols, solute_type)    

#def integrate_subloop_n_steps_np(grid_scalar_fields, grid_mat_prop,
#                                 grid_velocity, grid_no_cells, grid_ranges,
#                                 grid_steps, grid_volume_cell,
#                                 p_ref, p_ref_inv, pos, vel, cells, rel_pos,
#                                 m_w, m_s, xi, water_removed, id_list,
#                                 active_ids, T_p, delta_m_l, delta_Q_p,
#                                 delta_Theta_ad, delta_r_v_ad,
#                                 dt_sub, dt_sub_pos, no_steps,
#                                 no_iter_impl_mass, g_set, solute_type):
#    '''
#    description
#    
#    
#    Parameters
#    ----------
#    
#    
#    Returns
#    -------
#    '''
#    
#    # d) subloop 1
#    # for n_h = 0, ..., N_h-1
#    if solute_type == "NaCl":
#        for n_sub in range(no_steps):
#            # i) for all particles
#            # updates delta_m_l and delta_Q_p
#            propagate_particles_subloop_step_NaCl(
#                grid_scalar_fields, grid_mat_prop,
#                grid_velocity,
#                grid_no_cells, grid_ranges, grid_steps,
#                pos, vel, cells, rel_pos, m_w, m_s, xi,
#                water_removed, id_list, active_ids,
#                T_p,
#                delta_m_l, delta_Q_p,
#                dt_sub, dt_sub_pos,
#                no_iter_impl_mass, g_set)
#    
#            # ii) to vii)
#            propagate_grid_subloop_step(grid_scalar_fields, grid_mat_prop,
#                                            p_ref, p_ref_inv,
#                                            delta_Theta_ad, delta_r_v_ad,
#                                            delta_m_l, delta_Q_p,
#                                            grid_volume_cell)
#            
#            # viii) to ix) included in "propagate_particles_subloop"
#            # delta_Q_p.fill(0.0)
#            # delta_m_l.fill(0.0)
#            
#    elif solute_type == "AS":
#        for n_sub in range(no_steps):
#            # i) for all particles
#            # updates delta_m_l and delta_Q_p
#            propagate_particles_subloop_step_AS(
#                grid_scalar_fields, grid_mat_prop,
#                grid_velocity,
#                grid_no_cells, grid_ranges, grid_steps,
#                pos, vel, cells, rel_pos, m_w, m_s, xi,
#                water_removed, id_list, active_ids,
#                T_p,
#                delta_m_l, delta_Q_p,
#                dt_sub, dt_sub_pos,
#                no_iter_impl_mass, g_set)
#    
#            # ii) to vii)
#            propagate_grid_subloop_step(grid_scalar_fields, grid_mat_prop,
#                                            p_ref, p_ref_inv,
#                                            delta_Theta_ad, delta_r_v_ad,
#                                            delta_m_l, delta_Q_p,
#                                            grid_volume_cell)
#            
#            # viii) to ix) included in "propagate_particles_subloop"
#            # delta_Q_p.fill(0.0)
#            # delta_m_l.fill(0.0)
#        
#        # tau += dt_sub
#    # subloop 1 end    
#integrate_subloop_n_steps = njit()(integrate_subloop_n_steps_np)

    
##################################################################
def integrate_adv_and_cond_one_adv_step_np(
        grid_scalar_fields, grid_mat_prop, grid_velocity,
        grid_mass_flux_air_dry, p_ref, p_ref_inv,
        grid_no_cells, grid_ranges,
        grid_steps, grid_volume_cell,
        pos, vel, cells, rel_pos, m_w, m_s, xi, solute_type,
        water_removed,
        id_list, active_ids, T_p,
        delta_m_l, delta_Q_p,
        dt, dt_sub, dt_sub_half, scale_dt_cond, no_adv_steps,
        no_iter_impl_mass, g_set, act_collisions):
    ### one timestep dt:
    # a) dt_sub is set

    # b) for all particles: x_n+1/2 = x_n + h/2 v_n
    # removed_ids_step = []        
    update_pos_from_vel_BC_PS(m_w, pos, vel, xi, cells,
                              water_removed, id_list,
                              active_ids, grid_ranges, grid_steps,
                              dt_sub_half)
    update_cells_and_rel_pos(pos, cells, rel_pos,
                             active_ids,
                             grid_ranges, grid_steps)

    # c) advection change of r_v and T
    delta_r_v_ad = -dt_sub\
                   * compute_divergence_upwind(
                         grid_scalar_fields[4],
                         grid_mass_flux_air_dry,
                         grid_no_cells, grid_steps,
                         boundary_conditions = np.array([0, 1]))\
                   * grid_scalar_fields[9]
    delta_Theta_ad = -dt_sub\
                   * compute_divergence_upwind(
                         grid_scalar_fields[2],
                         grid_mass_flux_air_dry,
                         grid_no_cells, grid_steps,
                         boundary_conditions = np.array([0, 1]))\
                     * grid_scalar_fields[9]

    # d) subloop 1
    # for n_h = 0, ..., N_h-1
    integrate_subloop_n_steps(
            grid_scalar_fields, grid_mat_prop, grid_velocity,
            grid_no_cells, grid_ranges, grid_steps,
            grid_volume_cell, p_ref, p_ref_inv,
            pos, vel, cells, rel_pos, m_w, m_s, xi, solute_type, water_removed,
            id_list, active_ids, T_p,
            delta_m_l, delta_Q_p, delta_Theta_ad, delta_r_v_ad,
            dt_sub, dt_sub, scale_dt_cond, no_iter_impl_mass, g_set,
            act_collisions)
    # subloop 1 end
    
    # e) advection change of r_v and T for second subloop
    delta_r_v_ad = -2.0 * dt_sub\
                   * compute_divergence_upwind(
                         grid_scalar_fields[4],
                         grid_mass_flux_air_dry,
                         grid_no_cells, grid_steps,
                         boundary_conditions = np.array([0, 1]))\
                   * grid_scalar_fields[9] - delta_r_v_ad
    delta_Theta_ad = -2.0 * dt_sub\
                   * compute_divergence_upwind(
                         grid_scalar_fields[2],
                         grid_mass_flux_air_dry,
                         grid_no_cells, grid_steps,
                         boundary_conditions = np.array([0, 1]))\
                     * grid_scalar_fields[9] - delta_Theta_ad
    # f) subloop 2
    # for n_h = 0, ..., N_h-2
    integrate_subloop_n_steps(
            grid_scalar_fields, grid_mat_prop, grid_velocity,
            grid_no_cells, grid_ranges, grid_steps,
            grid_volume_cell, p_ref, p_ref_inv,
            pos, vel, cells, rel_pos, m_w, m_s, xi,solute_type, water_removed,
            id_list, active_ids, T_p,
            delta_m_l, delta_Q_p, delta_Theta_ad, delta_r_v_ad,
            dt_sub, dt_sub, scale_dt_cond-1,
            no_iter_impl_mass, g_set, solute_type, act_collisions)
    # subloop 2 end

    # add one step, where pos is moved only by half timestep x_n+1/2 -> x_n
    # i) for all particles
    # updates delta_m_l and delta_Q_p as well
    if solute_type == "NaCl":
        propagate_particles_subloop_step_NaCl(
                grid_scalar_fields, grid_mat_prop,
                grid_velocity,
                grid_no_cells, grid_ranges, grid_steps,
                pos, vel, cells, rel_pos, m_w, m_s, xi,
                water_removed, id_list, active_ids,
                T_p, delta_m_l, delta_Q_p,
                dt_sub, dt_sub_half,
                no_iter_impl_mass, g_set)
    elif solute_type == "AS":
        propagate_particles_subloop_step_AS(
                grid_scalar_fields, grid_mat_prop,
                grid_velocity,
                grid_no_cells, grid_ranges, grid_steps,
                pos, vel, cells, rel_pos, m_w, m_s, xi,
                water_removed, id_list, active_ids,
                T_p, delta_m_l, delta_Q_p,
                dt_sub, dt_sub_half,
                no_iter_impl_mass, g_set)
    # ii) to vii)
    propagate_grid_subloop_step(grid_scalar_fields, grid_mat_prop,
                                p_ref, p_ref_inv,
                                delta_Theta_ad, delta_r_v_ad,
                                delta_m_l, delta_Q_p,
                                grid_volume_cell)    
integrate_adv_and_cond_one_adv_step = \
    njit()(integrate_adv_and_cond_one_adv_step_np)

#%%
    
# dt = dt_adv
# no_col_per_adv = 1,2 OR scale_dt_cond * 2
# grid_scalar_fields[0] = grid.temperature
# grid_scalar_fields[1] = grid.pressure
# grid_scalar_fields[2] = grid.potential_temperature
# grid_scalar_fields[3] = grid.mass_density_air_dry
# grid_scalar_fields[4] = grid.mixing_ratio_water_vapor
# grid_scalar_fields[5] = grid.mixing_ratio_water_liquid
# grid_scalar_fields[6] = grid.saturation
# grid_scalar_fields[7] = grid.saturation_pressure
# grid_scalar_fields[8] = grid.mass_dry_inv
# grid_scalar_fields[9] = grid.rho_dry_inv

# grid_mat_prop[0] = grid.thermal_conductivity
# grid_mat_prop[1] = grid.diffusion_constant
# grid_mat_prop[2] = grid.heat_of_vaporization
# grid_mat_prop[3] = grid.surface_tension
# grid_mat_prop[4] = grid.specific_heat_capacity
# grid_mat_prop[5] = grid.viscosity
# grid_mat_prop[6] = grid.mass_density_fluid

def integrate_adv_cond_coll_one_adv_step_np(
        grid_scalar_fields, grid_mat_prop, grid_velocity,
        grid_mass_flux_air_dry, p_ref, p_ref_inv,
        grid_no_cells, grid_ranges,
        grid_steps, grid_volume_cell,
        pos, vel, cells, rel_pos, m_w, m_s, xi,
        water_removed,
        id_list, active_ids, T_p,
        delta_m_l, delta_Q_p,
        dt, dt_sub, dt_sub_half, dt_col_over_dV, scale_dt_cond,
        no_col_per_adv,        
        no_iter_impl_mass, g_set, solute_type,
        E_col_grid, no_kernel_bins,
        R_kernel_low_log, bin_factor_R_log, no_cols):
    ### one timestep dt:
    # a) dt_sub is set
    
    # b) for all particles: x_n+1/2 = x_n + h/2 v_n
    # the velocity is stored from the step before
    # this is why the collision step is done AFTER this position shift
    update_pos_from_vel_BC_PS(m_w, pos, vel, xi, cells,
                              water_removed, id_list,
                              active_ids, grid_ranges, grid_steps,
                              dt_sub_half)
    update_cells_and_rel_pos(pos, cells, rel_pos,
                             active_ids,
                             grid_ranges, grid_steps)

    # c) advection change of r_v and T
    delta_r_v_ad = -dt_sub\
                   * compute_divergence_upwind(
                         grid_scalar_fields[4],
                         grid_mass_flux_air_dry,
                         grid_no_cells, grid_steps,
                         boundary_conditions = np.array([0, 1]))\
                   * grid_scalar_fields[9]
    delta_Theta_ad = -dt_sub\
                   * compute_divergence_upwind(
                         grid_scalar_fields[2],
                         grid_mass_flux_air_dry,
                         grid_no_cells, grid_steps,
                         boundary_conditions = np.array([0, 1]))\
                     * grid_scalar_fields[9]
    
    ### d1) added collision here
#    if no_col_per_adv == 2:
#        collision_step_Long_Bott_Ecol_grid_R_all_cells_2D_multicomp_np(
#            xi, m_w, m_s, vel, grid_scalar_fields[0], cells, grid_no_cells,
#            dt_col_over_dV, E_col_grid, no_kernel_bins,
#            R_kernel_low_log, bin_factor_R_log, no_cols, solute_type)
    # d) subloop 1
    # for n_h = 0, ..., N_h-1
    # in here, we add e.g. 5 = 5 collision steps if no_col_per_adv > 2
    
    if no_col_per_adv == 1:
        no_col_steps = 0
    elif no_col_per_adv == 2:
        no_col_steps = 1
    else: no_col_steps = scale_dt_cond
    
    integrate_subloop_w_col_n_steps_np(
                grid_scalar_fields, grid_mat_prop, grid_velocity,
                grid_no_cells, grid_ranges, grid_steps,
                grid_volume_cell, p_ref, p_ref_inv,
                pos, vel, cells, rel_pos, m_w, m_s, xi,
                dt_col_over_dV, E_col_grid, no_kernel_bins,
                R_kernel_low_log, bin_factor_R_log, no_cols,
                water_removed,
                id_list, active_ids, T_p,
                delta_m_l, delta_Q_p, delta_Theta_ad, delta_r_v_ad,
                dt_sub, dt_sub, scale_dt_cond, no_col_steps,
                no_iter_impl_mass, g_set, solute_type)
#    if no_col_per_adv > 2:
#        integrate_subloop_w_col_n_steps(
#                grid_scalar_fields, grid_mat_prop, grid_velocity,
#                grid_no_cells, grid_ranges, grid_steps,
#                grid_volume_cell, p_ref, p_ref_inv,
#                pos, vel, cells, rel_pos, m_w, m_s, xi, water_removed,
#                id_list, active_ids, T_p,
#                delta_m_l, delta_Q_p, delta_Theta_ad, delta_r_v_ad,
#                dt_sub, dt_sub, scale_dt_cond-1,
#                no_iter_impl_mass, g_set, solute_type)
#    else:
#        integrate_subloop_n_steps(
#                grid_scalar_fields, grid_mat_prop, grid_velocity,
#                grid_no_cells, grid_ranges, grid_steps,
#                grid_volume_cell, p_ref, p_ref_inv,
#                pos, vel, cells, rel_pos, m_w, m_s, xi, water_removed,
#                id_list, active_ids, T_p,
#                delta_m_l, delta_Q_p, delta_Theta_ad, delta_r_v_ad,
#                dt_sub, dt_sub, scale_dt_cond-1,
#                no_iter_impl_mass, g_set, solute_type)
#    integrate_subloop_n_steps(
#            grid_scalar_fields, grid_mat_prop, grid_velocity,
#            grid_no_cells, grid_ranges, grid_steps,
#            grid_volume_cell, p_ref, p_ref_inv,
#            pos, vel, cells, rel_pos, m_w, m_s, xi, water_removed,
#            id_list, active_ids, T_p,
#            delta_m_l, delta_Q_p, delta_Theta_ad, delta_r_v_ad,
#            dt_sub, dt_sub, scale_dt_cond, no_iter_impl_mass, g_set, solute_type)
    # subloop 1 end

    
    # e) advection change of r_v and T for second subloop
    delta_r_v_ad = -2.0 * dt_sub\
                   * compute_divergence_upwind(
                         grid_scalar_fields[4],
                         grid_mass_flux_air_dry,
                         grid_no_cells, grid_steps,
                         boundary_conditions = np.array([0, 1]))\
                   * grid_scalar_fields[9] - delta_r_v_ad
    delta_Theta_ad = -2.0 * dt_sub\
                   * compute_divergence_upwind(
                         grid_scalar_fields[2],
                         grid_mass_flux_air_dry,
                         grid_no_cells, grid_steps,
                         boundary_conditions = np.array([0, 1]))\
                     * grid_scalar_fields[9] - delta_Theta_ad

    
    ### f2) added collision here
#    collision_step_Long_Bott_Ecol_grid_R_all_cells_2D_multicomp_np(
#        xi, m_w, m_s, vel, grid_scalar_fields[0], cells, grid_no_cells,
#        dt_col_over_dV, E_col_grid, no_kernel_bins,
#        R_kernel_low_log, bin_factor_R_log, no_cols, solute_type)  
    # f) subloop 2
    # for n_h = 0, ..., N_h-2
    # in here, we add e.g. 5 = 4 + 1 collision steps if no_col_per_adv > 2
    # the step which is shifted back for particle condensation and acceleration
    # is already made in here
    # in this example, we have 5 + 4 + 1 col steps in total per adv step
#    if no_col_per_adv > 2:
    if no_col_per_adv in (1,2):
        no_col_steps = 1
    else: no_col_steps = scale_dt_cond

    integrate_subloop_w_col_n_steps_np(
                grid_scalar_fields, grid_mat_prop, grid_velocity,
                grid_no_cells, grid_ranges, grid_steps,
                grid_volume_cell, p_ref, p_ref_inv,
                pos, vel, cells, rel_pos, m_w, m_s, xi,
                dt_col_over_dV, E_col_grid, no_kernel_bins,
                R_kernel_low_log, bin_factor_R_log, no_cols,
                water_removed,
                id_list, active_ids, T_p,
                delta_m_l, delta_Q_p, delta_Theta_ad, delta_r_v_ad,
                dt_sub, dt_sub, scale_dt_cond-1, no_col_steps,
                no_iter_impl_mass, g_set, solute_type)
#    else:
#        integrate_subloop_n_steps(
#                grid_scalar_fields, grid_mat_prop, grid_velocity,
#                grid_no_cells, grid_ranges, grid_steps,
#                grid_volume_cell, p_ref, p_ref_inv,
#                pos, vel, cells, rel_pos, m_w, m_s, xi, water_removed,
#                id_list, active_ids, T_p,
#                delta_m_l, delta_Q_p, delta_Theta_ad, delta_r_v_ad,
#                dt_sub, dt_sub, scale_dt_cond-1,
#                no_iter_impl_mass, g_set, solute_type)
    # subloop 2 end

    # add one step, where pos is moved only by half timestep x_n+1/2 -> x_n
    # i) for all particles
    # updates delta_m_l and delta_Q_p as well
    # NOTE that the additional collisions step is already in the method above
    if solute_type == "NaCl":
        propagate_particles_subloop_step_NaCl(
                grid_scalar_fields, grid_mat_prop,
                grid_velocity,
                grid_no_cells, grid_ranges, grid_steps,
                pos, vel, cells, rel_pos, m_w, m_s, xi,
                water_removed, id_list, active_ids,
                T_p, delta_m_l, delta_Q_p,
                dt_sub, dt_sub_half,
                no_iter_impl_mass, g_set)
    elif solute_type == "AS":
        propagate_particles_subloop_step_AS(
                grid_scalar_fields, grid_mat_prop,
                grid_velocity,
                grid_no_cells, grid_ranges, grid_steps,
                pos, vel, cells, rel_pos, m_w, m_s, xi,
                water_removed, id_list, active_ids,
                T_p, delta_m_l, delta_Q_p,
                dt_sub, dt_sub_half,
                no_iter_impl_mass, g_set)
    # ii) to vii)
    propagate_grid_subloop_step(grid_scalar_fields, grid_mat_prop,
                                p_ref, p_ref_inv,
                                delta_Theta_ad, delta_r_v_ad,
                                delta_m_l, delta_Q_p,
                                grid_volume_cell)    
    
integrate_adv_cond_coll_one_adv_step = \
    njit()(integrate_adv_cond_coll_one_adv_step_np)

   
#%%

### SIMULATE INTERVAL WITH COLLISIONS
# grid_scalar_fields[0] = grid.temperature
# grid_scalar_fields[1] = grid.pressure
# grid_scalar_fields[2] = grid.potential_temperature
# grid_scalar_fields[3] = grid.mass_density_air_dry
# grid_scalar_fields[4] = grid.mixing_ratio_water_vapor
# grid_scalar_fields[5] = grid.mixing_ratio_water_liquid
# grid_scalar_fields[6] = grid.saturation
# grid_scalar_fields[7] = grid.saturation_pressure
# grid_scalar_fields[8] = grid.mass_dry_inv
# grid_scalar_fields[9] = grid.rho_dry_inv

# grid_mat_prop[0] = grid.thermal_conductivity
# grid_mat_prop[1] = grid.diffusion_constant
# grid_mat_prop[2] = grid.heat_of_vaporization
# grid_mat_prop[3] = grid.surface_tension
# grid_mat_prop[4] = grid.specific_heat_capacity
# grid_mat_prop[5] = grid.viscosity
# grid_mat_prop[6] = grid.mass_density_fluid

# simulates grid and particles for no_adv_steps advection timesteps dt = dt_adv
# with subloop integration with timestep dt_sub
# since coll step is 2 times faster in np-version, there is no njit() here
# the np version is the "normal" version
# test also to use numpy version of integrate_adv_cond_coll_one_adv_step_np
# should not make much of a difference, but who knows?    
def simulate_interval_col(grid_scalar_fields, grid_mat_prop, grid_velocity,
                         grid_mass_flux_air_dry, p_ref, p_ref_inv,
                         grid_no_cells, grid_ranges,
                         grid_steps, grid_volume_cell,
                         pos, vel, cells, rel_pos, m_w, m_s, xi, water_removed,
                         id_list, active_ids, T_p,
                         delta_m_l, delta_Q_p,
                         dt, dt_sub, dt_sub_half, dt_col,
                         scale_dt_cond, no_col_per_adv, no_adv_steps,
                         no_iter_impl_mass, g_set,
                         dump_every, trace_ids,
                         traced_vectors, traced_scalars,
                         traced_xi, traced_water,
                         E_col_grid, no_kernel_bins,
                         R_kernel_low_log, bin_factor_R_log, no_cols,
                         solute_type):
                         # , traced_grid_fields
    # two coll steps per one adv step                         
    dt_col_over_dV = dt_col / grid_volume_cell
#    dt_col_over_dV = 0.5 * dt / grid_volume_cell
    dump_N = 0
    for cnt in range(no_adv_steps):
        if cnt % dump_every == 0:
            traced_vectors[dump_N,0] = pos[:,trace_ids]
            traced_vectors[dump_N,1] = vel[:,trace_ids]
            traced_scalars[dump_N,0] = m_w[trace_ids]
            traced_scalars[dump_N,1] = m_s[trace_ids]
            traced_scalars[dump_N,2] = T_p[trace_ids]
            traced_xi[dump_N] = xi[trace_ids]
            traced_water[dump_N] = water_removed[0]
            # traced_grid_fields[dump_N,0] = np.copy(grid_scalar_fields[0])
            # traced_grid_fields[dump_N,1] = np.copy(grid_scalar_fields[4])

            dump_N +=1
            
#        if act_collision:
        
        integrate_adv_cond_coll_one_adv_step_np(
                grid_scalar_fields, grid_mat_prop, grid_velocity,
                grid_mass_flux_air_dry, p_ref, p_ref_inv,
                grid_no_cells, grid_ranges,
                grid_steps, grid_volume_cell,
                pos, vel, cells, rel_pos, m_w, m_s, xi,
                water_removed,
                id_list, active_ids, T_p,
                delta_m_l, delta_Q_p,
                dt, dt_sub, dt_sub_half, dt_col_over_dV, scale_dt_cond,
                no_col_per_adv,
                no_iter_impl_mass, g_set, solute_type,
                E_col_grid, no_kernel_bins,
                R_kernel_low_log, bin_factor_R_log, no_cols)
        
        

### WORKING VERSION WITH dt_col = dt_adv
#def simulate_interval_col(grid_scalar_fields, grid_mat_prop, grid_velocity,
#                         grid_mass_flux_air_dry, p_ref, p_ref_inv,
#                         grid_no_cells, grid_ranges,
#                         grid_steps, grid_volume_cell,
#                         pos, vel, cells, rel_pos, m_w, m_s, xi, water_removed,
#                         id_list, active_ids, T_p,
#                         delta_m_l, delta_Q_p,
#                         dt, dt_sub, dt_sub_half, scale_dt_cond, no_adv_steps,
#                         no_iter_impl_mass, g_set,
#                         dump_every, trace_ids,
#                         traced_vectors, traced_scalars,
#                         traced_xi, traced_water,
#                         E_col_grid, no_kernel_bins,
#                         R_kernel_low_log, bin_factor_R_log, no_cols,
#                         solute_type):
#                         # , traced_grid_fields
#    dt_over_dV = dt / grid_volume_cell
#    dump_N = 0
#    for cnt in range(no_adv_steps):
#        if cnt % dump_every == 0:
#            traced_vectors[dump_N,0] = pos[:,trace_ids]
#            traced_vectors[dump_N,1] = vel[:,trace_ids]
#            traced_scalars[dump_N,0] = m_w[trace_ids]
#            traced_scalars[dump_N,1] = m_s[trace_ids]
#            traced_scalars[dump_N,2] = T_p[trace_ids]
#            traced_xi[dump_N] = xi[trace_ids]
#            traced_water[dump_N] = water_removed[0]
#            # traced_grid_fields[dump_N,0] = np.copy(grid_scalar_fields[0])
#            # traced_grid_fields[dump_N,1] = np.copy(grid_scalar_fields[4])
#
#            dump_N +=1
#            
#        integrate_adv_and_cond_one_adv_step(
#            grid_scalar_fields, grid_mat_prop, grid_velocity,
#            grid_mass_flux_air_dry, p_ref, p_ref_inv,
#            grid_no_cells, grid_ranges,
#            grid_steps, grid_volume_cell,
#            pos, vel, cells, rel_pos, m_w, m_s, xi,
#            water_removed,
#            id_list, active_ids, T_p,
#            delta_m_l, delta_Q_p,
#            dt, dt_sub, dt_sub_half, scale_dt_cond, no_adv_steps,
#            no_iter_impl_mass, g_set, solute_type)    
#        
#        collision_step_Long_Bott_Ecol_grid_R_all_cells_2D_multicomp_np(
#            xi, m_w, m_s, vel, grid_scalar_fields[0], cells, grid_no_cells,
#            dt_over_dV, E_col_grid, no_kernel_bins,
#            R_kernel_low_log, bin_factor_R_log, no_cols, solute_type)




###
        
# since coll step is 2 times faster in np-version, there is no jitting here
#simulate_interval_col = njit()(simulate_interval_col_np)

### SIMULATE INTERVAL WITHOUT COLLISIONS
# grid_scalar_fields[0] = grid.temperature
# grid_scalar_fields[1] = grid.pressure
# grid_scalar_fields[2] = grid.potential_temperature
# grid_scalar_fields[3] = grid.mass_density_air_dry
# grid_scalar_fields[4] = grid.mixing_ratio_water_vapor
# grid_scalar_fields[5] = grid.mixing_ratio_water_liquid
# grid_scalar_fields[6] = grid.saturation
# grid_scalar_fields[7] = grid.saturation_pressure
# grid_scalar_fields[8] = grid.mass_dry_inv
# grid_scalar_fields[9] = grid.rho_dry_inv

# grid_mat_prop[0] = grid.thermal_conductivity
# grid_mat_prop[1] = grid.diffusion_constant
# grid_mat_prop[2] = grid.heat_of_vaporization
# grid_mat_prop[3] = grid.surface_tension
# grid_mat_prop[4] = grid.specific_heat_capacity
# grid_mat_prop[5] = grid.viscosity
# grid_mat_prop[6] = grid.mass_density_fluid

# simulates grid and particles for no_adv_steps advection timesteps dt = dt_adv
# with subloop integration with timestep dt_sub
# no jit here, because its faster without jit        
def simulate_interval_wout_col(grid_scalar_fields, grid_mat_prop, grid_velocity,
                                  grid_mass_flux_air_dry, p_ref, p_ref_inv,
                                  grid_no_cells, grid_ranges,
                                  grid_steps, grid_volume_cell,
                                  pos, vel, cells, rel_pos, m_w, m_s, xi,
                                  water_removed,
                                  id_list, active_ids, T_p,
                                  delta_m_l, delta_Q_p,
                                  dt, dt_sub, dt_sub_half, scale_dt_cond, no_adv_steps,
                                  no_iter_impl_mass, g_set,
                                  dump_every, trace_ids,
                                  traced_vectors, traced_scalars,
                                  traced_xi, traced_water, solute_type
                                  ):
                                  # , traced_grid_fields
    dump_N = 0
    for cnt in range(no_adv_steps):
        if cnt % dump_every == 0:
            traced_vectors[dump_N,0] = pos[:,trace_ids]
            traced_vectors[dump_N,1] = vel[:,trace_ids]
            traced_scalars[dump_N,0] = m_w[trace_ids]
            traced_scalars[dump_N,1] = m_s[trace_ids]
            traced_scalars[dump_N,2] = T_p[trace_ids]
            traced_xi[dump_N] = xi[trace_ids]
            traced_water[dump_N] = water_removed[0]
            # traced_grid_fields[dump_N,0] = np.copy(grid_scalar_fields[0])
            # traced_grid_fields[dump_N,1] = np.copy(grid_scalar_fields[4])
            dump_N +=1
            
        integrate_adv_and_cond_one_adv_step(
                grid_scalar_fields, grid_mat_prop, grid_velocity,
                grid_mass_flux_air_dry, p_ref, p_ref_inv,
                grid_no_cells, grid_ranges,
                grid_steps, grid_volume_cell,
                pos, vel, cells, rel_pos, m_w, m_s, xi,
                water_removed,
                id_list, active_ids, T_p,
                delta_m_l, delta_Q_p,
                dt, dt_sub, dt_sub_half, scale_dt_cond, no_adv_steps,
                no_iter_impl_mass, g_set, solute_type)            
# no jit here, because its faster without jit        
#simulate_interval_wout_col = njit()(simulate_interval_wout_col_np)        
        
        
#%%

#%% SIMULATE 191010 Before changes:
# added inpar dict
# switched order of arguments  

#def simulate(grid, pos, vel, cells, m_w, m_s, xi, solute_type,
#                 water_removed,
#                 active_ids,
#                 dt, dt_col, scale_dt_cond, no_col_per_adv,
#                 t_start, t_end, no_iter_impl_mass, g_set,
#                 act_collisions,
#                 frame_every, dump_every, trace_ids, 
#                 E_col_grid, no_kernel_bins,
#                 R_kernel_low_log, bin_factor_R_log,
#                 kernel_type, kernel_method,
#                 no_cols,
#                 rnd_seed,
#                 path, simulation_mode):
#    
#    log_file = path + f"log_sim_t_{int(t_start)}_{int(t_end)}.txt"
#    
#    start_time = datetime.now()
#    
#    # init particles
#    rel_pos = np.zeros_like(pos)
#    update_cells_and_rel_pos(pos, cells, rel_pos, active_ids,
#                             grid.ranges, grid.steps)
#    T_p = np.ones_like(m_w)
#    
#    id_list = np.arange(xi.shape[0])
#    
#    # init grid properties
#    grid.update_material_properties()
#    V0_inv = 1.0 / grid.volume_cell
#    grid.rho_dry_inv =\
#        np.ones_like(grid.mass_density_air_dry) / grid.mass_density_air_dry
#    grid.mass_dry_inv = V0_inv * grid.rho_dry_inv
#    
#    delta_Q_p = np.zeros_like(grid.temperature)
#    delta_m_l = np.zeros_like(grid.temperature)
#    
#    # prepare for jit-compilation
#    grid_scalar_fields = np.array( ( grid.temperature,
#                                     grid.pressure,
#                                     grid.potential_temperature,
#                                     grid.mass_density_air_dry,
#                                     grid.mixing_ratio_water_vapor,
#                                     grid.mixing_ratio_water_liquid,
#                                     grid.saturation,
#                                     grid.saturation_pressure,
#                                     grid.mass_dry_inv,
#                                     grid.rho_dry_inv ) )
#    
#    grid_mat_prop = np.array( ( grid.thermal_conductivity,
#                                grid.diffusion_constant,
#                                grid.heat_of_vaporization,
#                                grid.surface_tension,
#                                grid.specific_heat_capacity,
#                                grid.viscosity,
#                                grid.mass_density_fluid ) )
#    
#    # constants of the grid
#    grid_velocity = grid.velocity
#    grid_mass_flux_air_dry = grid.mass_flux_air_dry
#    grid_ranges = grid.ranges
#    grid_steps = grid.steps
#    grid_no_cells = grid.no_cells
#    grid_volume_cell = grid.volume_cell
#    p_ref = grid.p_ref
#    p_ref_inv = grid.p_ref_inv
#    
#    dt_sub = dt/(2 * scale_dt_cond)
#    dt_sub_half = 0.5 * dt_sub
#    print("dt = ", dt)
#    print("dt_col = ", dt_col)
#    print("dt_sub = ", dt_sub)
#    with open(log_file, "w+") as f:
#        f.write(f"simulation mode = {simulation_mode}\n")
#        f.write(f"gravitation const = {g_set}\n")
#        f.write(f"collisions activated = {act_collisions}\n")
#        f.write(f"kernel_type = {kernel_type}\n")
#        f.write(f"kernel_method = {kernel_method}")
#        if kernel_method == "Ecol_const":
#            f.write(f", E_col = {E_col_grid}")
#        f.write(f"\nsolute material = {solute_type}\n")        
#        f.write(f"dt = {dt}\n")    
#        f.write(f"dt_col = {dt_col}\n")    
#        f.write(f"dt_sub = {dt_sub}\n")    
#    cnt_max = (t_end - t_start) /dt
#    no_grid_frames = int(math.ceil(cnt_max / frame_every))
#    np.save(path+"data_saving_paras.npy",
#            (frame_every, no_grid_frames, dump_every))
#    dt_save = int(frame_every * dt)
#    grid_save_times =\
#        np.arange(t_start, t_end + 0.5 * dt_save, dt_save).astype(int)
#    np.save(path+"grid_save_times.npy", grid_save_times)
#    # frame_every = int(math.ceil(cnt_max / no_grid_frames))
#    # full_save_every = int(full_save_time_interval // dt)
#    
#    if isinstance(trace_ids, int):
#        trace_id_dist = int(math.floor(len(xi)/(trace_ids)))
#        trace_ids = np.arange(int(trace_id_dist*0.5), len(xi), trace_id_dist)
#    np.save(path+"trace_ids.npy", trace_ids)
#    no_trace_ids = len(trace_ids)
#    
#    dump_factor = frame_every // dump_every
#    print("frame_every, no_grid_frames")
#    print(frame_every, no_grid_frames)
#    print("dump_every, dump_factor")
#    print(dump_every, dump_factor)
#    with open(log_file, "a") as f:
#        f.write( f"frame_every, no_grid_frames\n" )
#        f.write( f"{frame_every} {no_grid_frames}\n" )
#        f.write( f"dump_every, dump_factor\n" )
#        f.write( f"{dump_every} {dump_factor}\n" )
#    traced_vectors = np.zeros((dump_factor, 2, 2, no_trace_ids))
#    traced_scalars = np.zeros((dump_factor, 3, no_trace_ids))
#    traced_xi = np.zeros((dump_factor, no_trace_ids))
#    traced_water = np.zeros(dump_factor)
##    traced_water = np.zeros(dump_factor)
#    # traced_grid_fields =\
#    #     np.zeros((dump_factor, 2, grid_no_cells[0], grid_no_cells[1]))
#    
#    sim_paras = [dt, dt_sub, no_iter_impl_mass, rnd_seed]
#    sim_par_names = "dt dt_sub no_iter_impl_mass rnd_seed_sim"
#    # sim_paras = [dt, dt_sub, no_iter_impl_mass, no_trace_ids]
#    # sim_par_names = "dt dt_sub no_iter_impl_mass no_trace_ids"
#    # sim_para_file = path + "sim_paras_t_" + str(t_start) + ".txt"
#    save_sim_paras_to_file(sim_paras, sim_par_names, t_start, path)
#    
#    date = datetime.now()
#    print("### simulation starts ###")
#    print("start date and time =", date)
#    print("sim time =", datetime.now() - start_time)
#    print()
#    with open(log_file, "a") as f:
#        f.write( f"### simulation starts ###\n" )    
#        f.write( f"start date and time = {date}\n" )    
#        f.write( f"sim time = {date-start_time}\n" )    
#    ### INTEGRATION LOOP START
#    if act_collisions: np.random.seed(rnd_seed)
#    for frame_N in range(no_grid_frames):
#        t = t_start + frame_N * frame_every * dt 
#        update_grid_r_l(m_w, xi, cells,
#                        grid_scalar_fields[5],
#                        grid_scalar_fields[8],
#                        active_ids,
#                        id_list)
#        save_grid_scalar_fields(t, grid_scalar_fields, path, start_time)
#        dump_particle_data_all(t, pos, vel, cells,
#                               m_w, m_s, xi, active_ids, path)
#        np.save(path + f"no_cols_{int(t)}.npy", no_cols)
#        if act_collisions:
#            simulate_interval_col(
#                    grid_scalar_fields, grid_mat_prop, grid_velocity,
#                    grid_mass_flux_air_dry, p_ref, p_ref_inv,
#                    grid_no_cells, grid_ranges,
#                    grid_steps, grid_volume_cell,
#                    pos, vel, cells, rel_pos, m_w, m_s, xi, water_removed,
#                    id_list, active_ids, T_p,
#                    delta_m_l, delta_Q_p,
#                    dt, dt_sub, dt_sub_half, dt_col,                    
#                    scale_dt_cond, no_col_per_adv, frame_every,
#                    no_iter_impl_mass, g_set,
#                    dump_every, trace_ids,
#                    traced_vectors, traced_scalars,
#                    traced_xi, traced_water,
#                    E_col_grid, no_kernel_bins,
#                    R_kernel_low_log, bin_factor_R_log, no_cols,
#                    solute_type)
#        else:
#            simulate_interval_wout_col(
#                    grid_scalar_fields,
#                    grid_mat_prop, grid_velocity,
#                    grid_mass_flux_air_dry, p_ref, p_ref_inv,
#                    grid_no_cells, grid_ranges,
#                    grid_steps, grid_volume_cell,
#                    pos, vel, cells, rel_pos, m_w, m_s, xi,
#                    water_removed,
#                    id_list, active_ids, T_p,
#                    delta_m_l, delta_Q_p,
#                    dt, dt_sub, dt_sub_half, scale_dt_cond,
#                    frame_every,
#                    no_iter_impl_mass, g_set,
#                    dump_every, trace_ids,
#                    traced_vectors, traced_scalars,
#                    traced_xi, traced_water, solute_type)
#            
#        time_block =\
#            np.arange(t, t + frame_every * dt, dump_every * dt).astype(int)
#        dump_particle_tracer_data_block(time_block,
#                                 traced_vectors, traced_scalars, traced_xi,
#                                 traced_water,
#                                 # traced_grid_fields,
#                                 path)
#    ### INTEGRATION LOOP END
#    
#    t = t_start + no_grid_frames * frame_every * dt
#    update_grid_r_l(m_w, xi, cells,
#                   grid_scalar_fields[5],
#                   grid_scalar_fields[8],
#                   active_ids,
#                   id_list)
#    save_grid_scalar_fields(t, grid_scalar_fields, path, start_time)        
#    dump_particle_data_all(t, pos, vel, cells, m_w, m_s, xi, active_ids, path)
#    np.save(path + f"no_cols_{int(t)}.npy", no_cols)
#    dump_particle_data(t, pos[:,trace_ids], vel[:,trace_ids],
#                       m_w[trace_ids], m_s[trace_ids], xi[trace_ids],
#                       grid_scalar_fields[0], grid_scalar_fields[4], path)
#    
#    np.save(path + f"water_removed_{int(t)}", water_removed)
#    
#    # full save at t_end
#    grid.temperature = grid_scalar_fields[0]
#    grid.pressure = grid_scalar_fields[1]
#    grid.potential_temperature = grid_scalar_fields[2]
#    grid.mass_density_air_dry = grid_scalar_fields[3]
#    grid.mixing_ratio_water_vapor = grid_scalar_fields[4]
#    grid.mixing_ratio_water_liquid = grid_scalar_fields[5]
#    grid.saturation = grid_scalar_fields[6]
#    grid.saturation_pressure = grid_scalar_fields[7]
#    
#    grid.thermal_conductivity = grid_mat_prop[0]
#    grid.diffusion_constant = grid_mat_prop[1]
#    grid.heat_of_vaporization = grid_mat_prop[2]
#    grid.surface_tension = grid_mat_prop[3]
#    grid.specific_heat_capacity = grid_mat_prop[4]
#    grid.viscosity = grid_mat_prop[5]
#    grid.mass_density_fluid = grid_mat_prop[6]
#    
##    active_ids = np.nonzero(xi)[0]
##    removed_ids = np.where(xi == 0)[0]
##    removed_ids = np.invert(active_ids)
#    
#    save_grid_and_particles_full(t, grid, pos, cells, vel, m_w, m_s, xi,
#                                     active_ids,
#                                     path)
#    # total water removed by hitting the ground
#    # convert to kg
#    # water_removed *= 1.0E-18
#    print()
#    print("### simulation ended ###")
#    print("t_start = ", t_start)
#    print("t_end = ", t_end)
#    print("dt = ", dt, "; dt_sub = ", dt_sub)
#    print("simulation time:")
#    end_time = datetime.now()
#    print(end_time - start_time)
#    with open(log_file, "a") as f:
#        f.write( f"### simulation ended ###\n" )    
#        f.write( f"t_start = {t_start}\n" )    
#        f.write( f"t_end = {t_end}\n" ) 
#        f.write( f"dt = {dt}; dt_sub = {dt_sub}\n" ) 
#        f.write( f"simulation time = {end_time - start_time}\n" )  



#%% SIMULATE FULL WORKING VERSION WITHOUT COLLISIONS
### NOT LONGER REQUIRED, INCLUDED IN FUNCTION ABOVE        
### SIMULATE FULL WORKING VERSION WITHOUT COLLISIONS
# grid_scalar_fields[0] = grid.temperature
# grid_scalar_fields[1] = grid.pressure
# grid_scalar_fields[2] = grid.potential_temperature
# grid_scalar_fields[3] = grid.mass_density_air_dry
# grid_scalar_fields[4] = grid.mixing_ratio_water_vapor
# grid_scalar_fields[5] = grid.mixing_ratio_water_liquid
# grid_scalar_fields[6] = grid.saturation
# grid_scalar_fields[7] = grid.saturation_pressure
# grid_scalar_fields[8] = grid.mass_dry_inv
# grid_scalar_fields[9] = grid.rho_dry_inv

# grid_mat_prop[0] = grid.thermal_conductivity
# grid_mat_prop[1] = grid.diffusion_constant
# grid_mat_prop[2] = grid.heat_of_vaporization
# grid_mat_prop[3] = grid.surface_tension
# grid_mat_prop[4] = grid.specific_heat_capacity
# grid_mat_prop[5] = grid.viscosity
# grid_mat_prop[6] = grid.mass_density_fluid

### INPUT
# takes (grid, pos, vel, cells, m_w, m_s, xi) in a certain state
# and integrates the EOM between t_start and t_end with advection time step dt
# i.e. for a number of (t_start - t_end) / dt advection time steps
# the particle subloop timestep is defined by dt_sub = dt / (2 * scale_dt_cond)
# no_iter_impl_mass (int) = number of iterations in the implicit mass condensation
# algorithm
# g_set = gravity constant (>=0): can be either 0.0 for spin up or 9.8...
# frame_every, dump every, trace_ids, path -> see below at OUTPUT
### OUTPUT
# saves grid fields every "frame_every" advection steps (int)
# saves ("dumps") trajectories and velocities of a number of traced particles
# every "dump every" advection steps (int)
# trace ids can be an np.array = [ID0, ID1, ..] or an integer
# -> if integer: spread this number of tracers uniformly over the ID-range
# path: path to save data, the file notation is chosen internally
#def simulate_wout_col(grid, pos, vel, cells, m_w, m_s, xi, solute_type,
#                      water_removed,
#                      active_ids,
#                      dt, scale_dt_cond, t_start, t_end, no_iter_impl_mass, g_set,
#                      frame_every, dump_every, trace_ids, path,
#                      simulation_mode):
#    log_file = path + f"log_sim_t_{int(t_start)}_{int(t_end)}.txt"
#    
#    start_time = datetime.now()
#    
#    # init particles
#    rel_pos = np.zeros_like(pos)
#    update_cells_and_rel_pos(pos, cells, rel_pos, active_ids,
#                             grid.ranges, grid.steps)
#    T_p = np.ones_like(m_w)
#    
#    id_list = np.arange(xi.shape[0])
#    
#    # init grid properties
#    grid.update_material_properties()
#    V0_inv = 1.0 / grid.volume_cell
#    grid.rho_dry_inv =\
#        np.ones_like(grid.mass_density_air_dry) / grid.mass_density_air_dry
#    grid.mass_dry_inv = V0_inv * grid.rho_dry_inv
#    
#    delta_Q_p = np.zeros_like(grid.temperature)
#    delta_m_l = np.zeros_like(grid.temperature)
#    
#    # prepare for jit-compilation
#    grid_scalar_fields = np.array( ( grid.temperature,
#                                     grid.pressure,
#                                     grid.potential_temperature,
#                                     grid.mass_density_air_dry,
#                                     grid.mixing_ratio_water_vapor,
#                                     grid.mixing_ratio_water_liquid,
#                                     grid.saturation,
#                                     grid.saturation_pressure,
#                                     grid.mass_dry_inv,
#                                     grid.rho_dry_inv ) )
#    
#    grid_mat_prop = np.array( ( grid.thermal_conductivity,
#                                grid.diffusion_constant,
#                                grid.heat_of_vaporization,
#                                grid.surface_tension,
#                                grid.specific_heat_capacity,
#                                grid.viscosity,
#                                grid.mass_density_fluid ) )
#    
#    # constants of the grid
#    grid_velocity = grid.velocity
#    grid_mass_flux_air_dry = grid.mass_flux_air_dry
#    grid_ranges = grid.ranges
#    grid_steps = grid.steps
#    grid_no_cells = grid.no_cells
#    grid_volume_cell = grid.volume_cell
#    p_ref = grid.p_ref
#    p_ref_inv = grid.p_ref_inv
#    
#    dt_sub = dt/(2 * scale_dt_cond)
#    dt_sub_half = 0.5 * dt_sub
#    print("dt_sub = ", dt_sub)
#    with open(log_file, "w+") as f:
#        f.write(f"simulation mode = {simulation_mode}\n")
#        f.write(f"gravitation const = {g_set}\n")
#        f.write(f"collisions activated = {False}\n")
#        f.write(f"solute material = {solute_type}\n")
#        f.write(f"dt_sub = {dt_sub}\n")    
#    cnt_max = (t_end - t_start) /dt
#    no_grid_frames = int(math.ceil(cnt_max / frame_every))
#    np.save(path+"data_saving_paras.npy",
#            (frame_every, no_grid_frames, dump_every))
#    dt_save = int(frame_every * dt)
#    grid_save_times =\
#        np.arange(t_start, t_end + 0.5 * dt_save, dt_save).astype(int)
#    np.save(path+"grid_save_times.npy", grid_save_times)
#    # frame_every = int(math.ceil(cnt_max / no_grid_frames))
#    # full_save_every = int(full_save_time_interval // dt)
#    
#    if isinstance(trace_ids, int):
#        trace_id_dist = int(math.floor(len(xi)/(trace_ids)))
#        trace_ids = np.arange(int(trace_id_dist*0.5), len(xi), trace_id_dist)
#    np.save(path+"trace_ids.npy", trace_ids)
#    no_trace_ids = len(trace_ids)
#    
#    dump_factor = frame_every // dump_every
#    print("frame_every, no_grid_frames")
#    print(frame_every, no_grid_frames)
#    print("dump_every, dump_factor")
#    print(dump_every, dump_factor)
#    with open(log_file, "a") as f:
#        f.write( f"frame_every, no_grid_frames\n" )
#        f.write( f"{frame_every} {no_grid_frames}\n" )
#        f.write( f"dump_every, dump_factor\n" )
#        f.write( f"{dump_every} {dump_factor}\n" )
#    traced_vectors = np.zeros((dump_factor, 2, 2, no_trace_ids))
#    traced_scalars = np.zeros((dump_factor, 2, no_trace_ids))
#    traced_xi = np.zeros((dump_factor, no_trace_ids))
#    traced_water = np.zeros(dump_factor)
##    traced_water = np.zeros(dump_factor)
#    # traced_grid_fields =\
#    #     np.zeros((dump_factor, 2, grid_no_cells[0], grid_no_cells[1]))
#    
#    sim_paras = [dt, dt_sub, no_iter_impl_mass]
#    sim_par_names = "dt dt_sub no_iter_impl_mass"
#    # sim_paras = [dt, dt_sub, no_iter_impl_mass, no_trace_ids]
#    # sim_par_names = "dt dt_sub no_iter_impl_mass no_trace_ids"
#    # sim_para_file = path + "sim_paras_t_" + str(t_start) + ".txt"
#    save_sim_paras_to_file(sim_paras, sim_par_names, t_start, path)
#    
#    date = datetime.now()
#    print("### simulation starts ###")
#    print("start date and time =", date)
#    print("sim time =", datetime.now() - start_time)
#    print()
#    with open(log_file, "a") as f:
#        f.write( f"### simulation starts ###\n" )    
#        f.write( f"start date and time = {date}\n" )    
#        f.write( f"sim time = {date-start_time}\n" )    
#    ### INTEGRATION LOOP START
#    for frame_N in range(no_grid_frames):
#        t = t_start + frame_N * frame_every * dt 
#        update_grid_r_l(m_w, xi, cells,
#                        grid_scalar_fields[5],
#                        grid_scalar_fields[8],
#                        active_ids,
#                        id_list)
#        save_grid_scalar_fields(t, grid_scalar_fields, path, start_time)
#        dump_particle_data_all(t, pos, vel, m_w, m_s, xi, path)
#        simulate_interval_wout_col(grid_scalar_fields, grid_mat_prop,
#                                   grid_velocity,
#                                 grid_mass_flux_air_dry, p_ref, p_ref_inv,
#                                 grid_no_cells, grid_ranges,
#                                 grid_steps, grid_volume_cell,
#                                 pos, vel, cells, rel_pos, m_w, m_s, xi,
#                                 water_removed,
#                                 id_list, active_ids, T_p,
#                                 delta_m_l, delta_Q_p,
#                                 dt, dt_sub, dt_sub_half, scale_dt_cond, frame_every,
#                                 no_iter_impl_mass, g_set,
#                                 dump_every, trace_ids,
#                                 traced_vectors, traced_scalars,
#                                 traced_xi, traced_water,
#                                 solute_type
#                                 )        
#        time_block =\
#            np.arange(t, t + frame_every * dt, dump_every * dt).astype(int)
#        dump_particle_tracer_data_block(time_block,
#                                 traced_vectors, traced_scalars, traced_xi,
#                                 traced_water,
#                                 # traced_grid_fields,
#                                 path)
#    ### INTEGRATION LOOP END
#    
#    t = t_start + no_grid_frames * frame_every * dt
#    update_grid_r_l(m_w, xi, cells,
#                   grid_scalar_fields[5],
#                   grid_scalar_fields[8],
#                   active_ids,
#                   id_list)
#    save_grid_scalar_fields(t, grid_scalar_fields, path, start_time)        
#    dump_particle_data_all(t, pos, vel, m_w, m_s, xi, path)
#    dump_particle_data(t, pos[:,trace_ids], vel[:,trace_ids],
#                       m_w[trace_ids], m_s[trace_ids], xi[trace_ids],
#                       grid_scalar_fields[0], grid_scalar_fields[4], path)
#    
#    # full save at t_end
#    grid.temperature = grid_scalar_fields[0]
#    grid.pressure = grid_scalar_fields[1]
#    grid.potential_temperature = grid_scalar_fields[2]
#    grid.mass_density_air_dry = grid_scalar_fields[3]
#    grid.mixing_ratio_water_vapor = grid_scalar_fields[4]
#    grid.mixing_ratio_water_liquid = grid_scalar_fields[5]
#    grid.saturation = grid_scalar_fields[6]
#    grid.saturation_pressure = grid_scalar_fields[7]
#    
#    grid.thermal_conductivity = grid_mat_prop[0]
#    grid.diffusion_constant = grid_mat_prop[1]
#    grid.heat_of_vaporization = grid_mat_prop[2]
#    grid.surface_tension = grid_mat_prop[3]
#    grid.specific_heat_capacity = grid_mat_prop[4]
#    grid.viscosity = grid_mat_prop[5]
#    grid.mass_density_fluid = grid_mat_prop[6]
#    
##    active_ids = np.nonzero(xi)[0]
##    removed_ids = np.where(xi == 0)[0]
##    removed_ids = np.invert(active_ids)
#    
#    save_grid_and_particles_full(t, grid, pos, cells, vel, m_w, m_s, xi,
#                                     active_ids,
#                                     path)
#    # total water removed by hitting the ground
#    # convert to kg
#    # water_removed *= 1.0E-18
#    print()
#    print("### simulation ended ###")
#    print("t_start = ", t_start)
#    print("t_end = ", t_end)
#    print("dt = ", dt, "; dt_sub = ", dt_sub)
#    print("simulation time:")
#    end_time = datetime.now()
#    print(end_time - start_time)
#    with open(log_file, "a") as f:
#        f.write( f"### simulation ended ###\n" )    
#        f.write( f"t_start = {t_start}\n" )    
#        f.write( f"t_end = {t_end}\n" ) 
#        f.write( f"dt = {dt}; dt_sub = {dt_sub}\n" ) 
#        f.write( f"simulation time = {end_time - start_time}\n" )          
        