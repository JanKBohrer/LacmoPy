#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 11:39:43 2020

@author: bohrer
"""


#%% FROM MICROPHYISCS

#%% NOT UPDATED WITH NUMBA
# NOT UPDATED WITH NUMBA
# def compute_delta_water_liquid_and_mass_rate_implicit_Newton_full_const_l(
#         dt_sub_, Newton_iter_, mass_water_, mass_solute_,
#         mass_particle_, mass_fraction_solute_, radius_particle_,
#         temperature_particle_, density_particle_,
#         amb_temp_, amb_press_,
#         amb_sat_, amb_sat_press_,
#         diffusion_constant_,
#         thermal_conductivity_air_,
#         specific_heat_capacity_air_,
#         heat_of_vaporization_,
#         surface_tension_,
#         adiabatic_index_,
#         accomodation_coefficient_,
#         condensation_coefficient_):
#     w_s_effl_inv = 1.0 / compute_efflorescence_mass_fraction_NaCl(
#                              temperature_particle_)
#     m_w_effl = mass_solute_ * (w_s_effl_inv - 1.0)
    
#     gamma0, dgamma_dm =\
#         compute_mass_rate_and_mass_rate_derivative_Szumowski_const_l(
#             mass_water_, mass_solute_,
#             mass_particle_, mass_fraction_solute_, radius_particle_,
#             temperature_particle_, density_particle_,
#             amb_temp_, amb_press_,
#             amb_sat_, amb_sat_press_,
#             diffusion_constant_,
#             thermal_conductivity_air_,
#             specific_heat_capacity_air_,
#             heat_of_vaporization_,
#             surface_tension_,
#             adiabatic_index_,
#             accomodation_coefficient_,
#             condensation_coefficient_)
# #    Newton_iter = 3
#     dt_sub_times_dgamma_dm = dt_sub_ * dgamma_dm
#     denom_inv = np.where(dt_sub_times_dgamma_dm < 0.9,
#                          1.0 / (1.0 - dt_sub_times_dgamma_dm),
#                          10.0)
# #    if (dt_sub_ * dgamma_dm < 0.9):
# #        denom_inv = 1.0 / (1.0 - dt_sub_ * dgamma_dm)
# #    else:
# #        denom_inv = 10.0
     
#     mass_new = np.maximum(m_w_effl,
#                           mass_water_ + dt_sub_ * gamma0 * denom_inv)
    
#     for cnt in range(no_iter_-1):
#         m_p = mass_new + mass_solute_
#         w_s = mass_solute_ / m_p
#         rho = compute_density_particle(w_s, temperature_particle_)
#         R = compute_radius_from_mass(m_p, rho)
#         gamma, dgamma_dm =\
#             compute_mass_rate_and_mass_rate_derivative_Szumowski_const_l(
#                 mass_new, mass_solute_,
#                 m_p, w_s, R,
#                 temperature_particle_, rho,
#                 amb_temp_, amb_press_,
#                 amb_sat_, amb_sat_press_,
#                 diffusion_constant_,
#                 thermal_conductivity_air_,
#                 specific_heat_capacity_air_,
#                 heat_of_vaporization_,
#                 surface_tension_,
#                 adiabatic_index_,
#                 accomodation_coefficient_,
#                 condensation_coefficient_)
#         dt_sub_times_dgamma_dm = dt_sub_ * dgamma_dm
#         denom_inv = np.where(dt_sub_times_dgamma_dm < 0.9,
#                              1.0 / (1.0 - dt_sub_times_dgamma_dm),
#                      10.0)
# #        if (dt_sub_ * dgamma_dm < 0.9):
# #            denom_inv = 1.0 / (1.0 - dt_sub_ * dgamma_dm)
# #        else:
# #            denom_inv = 10.0
#         mass_new += ( dt_sub_* gamma + mass_water_ - mass_new) * denom_inv
#         mass_new = np.maximum( m_w_effl, mass_new )
        
#     return mass_new - mass_water_, gamma0

# NOT UPDATED WITH NUMBA
# def compute_delta_water_liquid_and_mass_rate_implicit_Newton_inverse_full(
#         dt_sub_, no_iter_, mass_water_, mass_solute_,
#         mass_particle_, mass_fraction_solute_, radius_particle_,
#         temperature_particle_, density_particle_,
#         amb_temp_, amb_press_,
#         amb_sat_, amb_sat_press_,
#         diffusion_constant_,
#         thermal_conductivity_air_,
#         specific_heat_capacity_air_,
#         heat_of_vaporization_,
#         surface_tension_,
#         adiabatic_index_,
#         accomodation_coefficient_,
#         condensation_coefficient_):
#     w_s_effl_inv = 1.0 / compute_efflorescence_mass_fraction_NaCl(
#                              temperature_particle_)
#     m_w_effl = mass_solute_ * (w_s_effl_inv - 1.0)
    
#     gamma0, dgamma_dm = compute_mass_rate_and_mass_rate_derivative_Szumowski(
#                             mass_water_, mass_solute_,
#                             mass_particle_, mass_fraction_solute_,
#                             radius_particle_,
#                             temperature_particle_, density_particle_,
#                             amb_temp_, amb_press_,
#                             amb_sat_, amb_sat_press_,
#                             diffusion_constant_,
#                             thermal_conductivity_air_,
#                             specific_heat_capacity_air_,
#                             heat_of_vaporization_,
#                             surface_tension_,
#                             adiabatic_index_,
#                             accomodation_coefficient_,
#                             condensation_coefficient_)
# #    mass_new = mass_water_
# #    no_iter = 3
#     dgamma_factor = dt_sub_ * dgamma_dm
#     # dgamma_factor = dt_sub * F'(m) * m
#     dgamma_factor = np.where(dgamma_factor < 0.9,
#                              mass_water_ * (1.0 - dgamma_factor),
#                              mass_water_ * 0.1)
# #    if (dt_sub_ * dgamma_dm < 0.9):
# #        denom_inv = 1.0 / (1.0 - dt_sub_ * dgamma_dm)
# #    else:
# #        denom_inv = 10.0
    
#     print("iter = 1",
#           mass_water_ * dgamma_factor / (dgamma_factor - gamma0 * dt_sub_))
#     mass_new = np.maximum( m_w_effl,
#                            mass_water_ * dgamma_factor\
#                            / (dgamma_factor - gamma0 * dt_sub_) )
#     print("iter = 1", mass_new)
    
#     for cnt in range(no_iter_-1):
#         m_p = mass_new + mass_solute_
#         w_s = mass_solute_ / m_p
#         rho = compute_density_particle(w_s, temperature_particle_)
#         R = compute_radius_from_mass(m_p, rho)
#         gamma, dgamma_dm =\
#             compute_mass_rate_and_mass_rate_derivative_Szumowski(
#                                mass_new, mass_solute_,
#                                m_p, w_s, R,
#                                temperature_particle_, rho,
#                                amb_temp_, amb_press_,
#                                amb_sat_, amb_sat_press_,
#                                diffusion_constant_,
#                                thermal_conductivity_air_,
#                                specific_heat_capacity_air_,
#                                heat_of_vaporization_,
#                                surface_tension_,
#                                adiabatic_index_,
#                                accomodation_coefficient_,
#                                condensation_coefficient_)
#         dgamma_factor = dt_sub_ * dgamma_dm
#         # dgamma_factor = dt_sub * F'(m) * m
#         dgamma_factor = np.where(dgamma_factor < 0.9,
#                                  mass_new * (1.0 - dgamma_factor),
#                                  mass_new * 0.1)
# #        mass_new *= ( dt_sub_* gamma + mass_water_ - mass_new) * denom_inv
#         print("iter = ", cnt + 2 ,
#               mass_new * dgamma_factor\
#               / ( dgamma_factor - gamma * dt_sub_ + mass_new - mass_water_ ))
#         mass_new = np.maximum( m_w_effl,
#                                mass_new * dgamma_factor\
#                                / ( dgamma_factor - gamma * dt_sub_
#                                    + mass_new - mass_water_ ) )
#         print("iter = ", cnt+2, mass_new)
#     return mass_new - mass_water_, gamma0
    
# returns the difference dm_w = m_w_n+1 - m_w_n
# during condensation/evaporation
# during one timestep using linear implicit explicit euler
# also: returns mass_rate
# NOT UPDATED WITH NUMBA
# def compute_delta_water_liquid_and_mass_rate_imex_linear(
#         dt_, mass_water_, mass_solute_, #  in femto gram
#         temperature_particle_,
#         amb_temp_, amb_press_,
#         amb_sat_, amb_sat_press_,
#         diffusion_constant_,
#         thermal_conductivity_air_,
#         specific_heat_capacity_air_,
#         adiabatic_index_,
#         accomodation_coefficient_,
#         condensation_coefficient_, 
#         heat_of_vaporization_,
#         verbose = False):

#     dt_left = dt_
# #    dt = dt_
#     mass_water_new = mass_water_
    
#     mass_fraction_solute_effl = compute_efflorescence_mass_fraction_NaCl(
#                                     temperature_particle_)
    
#     while (dt_left > 0.0):
#         mass_rate = compute_mass_rate_from_water_mass_Szumowski(
#                         mass_water_new, mass_solute_, #  in femto gram
#                         temperature_particle_,
#                         amb_temp_, amb_press_,
#                         amb_sat_, amb_sat_press_,
#                         diffusion_constant_,
#                         thermal_conductivity_air_,
#                         specific_heat_capacity_air_,
#                         adiabatic_index_,
#                         accomodation_coefficient_,
#                         condensation_coefficient_, 
#                         heat_of_vaporization_)
#         mass_rate_derivative =\
#             compute_mass_rate_derivative_Szumowski_numerical(
                  # in femto gram
                  # mass_water_new, mass_solute_,
                  # temperature_particle_,
                  # amb_temp_, amb_press_,
                  # amb_sat_, amb_sat_press_,
                  # diffusion_constant_,
                  # thermal_conductivity_air_,
                  # specific_heat_capacity_air_,
                  # adiabatic_index_,
                  # accomodation_coefficient_,
                  # condensation_coefficient_, 
                  # heat_of_vaporization_
                  # )
#         if (verbose):
#             print('mass_rate, mass_rate_derivative:')
#             print(mass_rate, mass_rate_derivative)
#         # safety to avoid (1 - dt/2 * f'(m_n)) going to zero
#         if mass_rate_derivative * dt_ < 1.0:
#             dt = dt_left
#             dt_left = -1.0
#         else:
#             dt = 1.0 / mass_rate_derivative
#             dt_left -= dt
    
#         mass_water_new += mass_rate * dt\
#                           / ( 1.0 - 0.5 * mass_rate_derivative * dt )
        
#         mass_water_effl = mass_solute_\
#                           * (1.0 / mass_fraction_solute_effl - 1.0)
        
# #        mass_fraction_solute_new = mass_solute_\
# #           / (mass_water_new + mass_solute_)
        
# #        if (mass_fraction_solute_new >
# #             mass_fraction_solute_effl or mass_water_new < 0.0):
#         if (mass_water_new  < mass_water_effl):
# # mass_water_new = mass_solute_ * (1.0 / mass_fraction_solute_effl - 1.0)
#             mass_water_new = mass_water_effl
#             dt_left = -1.0
# #            print('w_s_effl reached')
    
#     return mass_water_new - mass_water_, mass_rate

# returns the difference dm_w = m_w_n+1 - m_w_n
# during condensation/evaporation
# during one timestep using linear implicit euler
# masses in femto gram
# NOT UPDATED WITH NUMBA
# def compute_delta_water_liquid_implicit_linear( dt_, mass_water_,
#                                                 mass_solute_,
#                                                 temperature_particle_,
#                                                 amb_temp_, amb_press_,
#                                                 amb_sat_, amb_sat_press_,
#                                                 diffusion_constant_,
#                                                 thermal_conductivity_air_,
#                                                 specific_heat_capacity_air_,
#                                                 adiabatic_index_,
#                                                 accomodation_coefficient_,
#                                                 condensation_coefficient_, 
#                                                 heat_of_vaporization_,
#                                                 verbose = False):

#     dt_left = dt_
# #    dt = dt_
#     mass_water_new = mass_water_
    
#     mass_fraction_solute_effl = compute_efflorescence_mass_fraction_NaCl(
#                                     temperature_particle_)
    
#     surface_tension_ = compute_surface_tension_water(temperature_particle_)
    
#     while (dt_left > 0.0):
# #        mass_rate =\
# #            compute_mass_rate_from_water_mass_Szumowski(
                    # mass_water_new, mass_solute_, #  in femto gram
                    #                   temperature_particle_,
                    #                   amb_temp_, amb_press_,
                    #                   amb_sat_, amb_sat_press_,
                    #                   diffusion_constant_,
                    #                   thermal_conductivity_air_,
                    #                   specific_heat_capacity_air_,
                    #                   adiabatic_index_,
                    #                   accomodation_coefficient_,
                    #                   condensation_coefficient_, 
                    #                   heat_of_vaporization_)
#         m_p = mass_water_new + mass_solute_
#         w_s = mass_solute_ / m_p
#         rho = compute_density_particle(w_s, temperature_particle_)
#         R = compute_radius_from_mass(m_p, rho)
#         mass_rate, mass_rate_derivative =\
#             compute_mass_rate_and_mass_rate_derivative_Szumowski(
#                 mass_water_new, mass_solute_,
#                 m_p, w_s, R,
#                 temperature_particle_, rho,
#                 amb_temp_, amb_press_,
#                 amb_sat_, amb_sat_press_,
#                 diffusion_constant_,
#                 thermal_conductivity_air_,
#                 specific_heat_capacity_air_,
#                 heat_of_vaporization_,
#                 surface_tension_,
#                 adiabatic_index_,
#                 accomodation_coefficient_,
#                 condensation_coefficient_)
#         if (verbose):
#             print('mass_rate, mass_rate_derivative:')
#             print(mass_rate, mass_rate_derivative)
#         if mass_rate_derivative * dt_ < 0.5:
#             dt = dt_left
#             dt_left = -1.0
#         else:
#             dt = 0.5 / mass_rate_derivative
#             dt_left -= dt
    
#         mass_water_new += mass_rate * dt\
#                           / ( 1.0 - mass_rate_derivative * dt )
        
#         mass_water_effl = mass_solute_\
#                           * (1.0 / mass_fraction_solute_effl - 1.0)
        
# #        mass_fraction_solute_new =\
# #  mass_solute_ / (mass_water_new + mass_solute_)
        
# #        if (mass_fraction_solute_new
# #            > mass_fraction_solute_effl or mass_water_new < 0.0):
#         if (mass_water_new  < mass_water_effl):
# #            mass_water_new = mass_solute_\
# #                             * (1.0 / mass_fraction_solute_effl - 1.0)
#             mass_water_new = mass_water_effl
#             dt_left = -1.0
# #            print('w_s_effl reached')
    
#     return mass_water_new - mass_water_

# def compute_mass_rate_from_surface_partial_pressure(amb_temp_,
#                                                     amb_sat_, amb_sat_press_,
#                                                     diffusion_constant_,
#                                                     radius_,
#                                                     surface_partial_pressure_,
#                                                     particle_temperature_,
#                                                     ):
#     return 4.0E12 * np.pi * radius_ * diffusion_constant_\
#            / c.specific_gas_constant_water_vapor \
#            * ( amb_sat_ * amb_sat_press_ / amb_temp_
#                - surface_partial_pressure_ / particle_temperature_ )