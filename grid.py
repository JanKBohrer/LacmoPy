#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TROPOS LAGRANGIAN CLOUD MODEL
Super-Droplet method in two-dimensional kinematic framework
(Test Case 1 ICMW 2012)
Author: Jan Bohrer (bohrer@tropos.de)
Further contact: Oswald Knoth (knoth@tropos.de)

module for Grid class and grid operations

basic units:
particle mass, water mass, solute mass in femto gram = 10^-18 kg
particle radius in micro meter ("mu")
all other quantities in SI units
"""

#%% MODULE IMPORTS

import matplotlib.pyplot as plt
import numpy as np
import math
from numba import njit

import constants as c
import material_properties as mat
import atmosphere as atm
from plotting import plot_scalar_field_2D

#%% STANDARD FIELDS AND PROFILES

omega = 0.3
def u_rot_field(x, y):
    """Computes the x-component of a rotational velocity field.
    
    The field is vel = (u, v, w) = (-omega * y, omega * x, 0).
    The rotation of the field is rot vel = 2 * omega * e_z,
    where e_z is the unit vector in z-direction.
    
    Parameters
    ----------
    x: float
        x-Position, where the field is evaluated
    y: float
        y-Position, where the field is evaluated
    
    Returns
    -------
        float
            x-component of the velocity field
    
    """
    
    return -omega * y
def v_rot_field(x, y):
    """Computes the y-component of a rotational velocity field.
    
    The field is vel = (u, v, w) = (-omega * y, omega * x, 0).
    The rotation of the field is rot vel = 2 * omega * e_z,
    where e_z is the unit vector in z-direction.
    
    Parameters
    ----------
    x: float
        x-Position, where the field is evaluated
    y: float
        y-Position, where the field is evaluated
    
    Returns
    -------
        float
        y-component of the velocity field
    
    """
    
    return omega * x

T_ref = 288.15 # K
adiabatic_lapse_rate_dry = 0.0065 # K/m
def temperature_field_linear(x, y):
    """Linear temperature profile with constant adiabatic lapse rate
    
    Parameters
    ----------
    x: float
        x-Position, where the field is evaluated (m)
    y: float
        y-Position, where the field is evaluated (m)
    
    Returns
    -------
        float
        Temperature at (x,y)
    
    """
    
    return T_ref - adiabatic_lapse_rate_dry * y

p_ref = 101325.0 # Pa
def pressure_field_exponential(x, y):
    """Pressure field, which decreases exponentially with height
    
    Parameters
    ----------
    x: float
        x-Position, where the field is evaluated (m)
    y: float
        y-Position, where the field is evaluated (m)
    
    Returns
    -------
        float
        Pressure at (x,y)
    
    """
    return p_ref * np.exp( -y * c.earth_gravity * c.molar_mass_air_dry\
                           / ( T_ref * c.universal_gas_constant ) )

#%% OPERATIONS ON THE GRID CELLS

def compute_no_grid_cells_from_step_sizes(grid_ranges, grid_steps):
    """Compute number of grid cells from domain sizes and grid step sizes
    
    Parameters
    ----------
    grid_ranges: ndarray, dtype=float
        2D array holding the coordinates of the domain box in two dimensions
        grid_ranges[0] = [x_min, x_max]
        grid_ranges[1] = [z_min, z_max]
    grid_steps: ndarray, dtype=float
        grid_steps[0] = horizontal grid step size in x (m)
        grid_steps[1] = vertical grid step size in z (m)
    
    Returns
    -------
    grid_no_cells: ndarray, dtype=int
        grid_no_cells[0] = number of grid cells in x (horizontal)
        grid_no_cells[1] = number of grid cells in z (vertical)
    
    """    
    
    grid_no_cells = []
    for i, range_i in enumerate(grid_ranges):
        grid_no_cells.append(
            int(np.ceil( (range_i[1] - range_i[0]) / grid_steps[i] ) ) )
    return np.array(grid_no_cells)

@njit()
def compute_cell_and_relative_position(pos, grid_ranges, grid_steps):
    """Compute the cells and relative positions from full positions
    
    Parameters
    ----------
    pos: ndarray, dtype=float
        2D array, where
        pos[0] = 1D array of horizontal coordinates (m)
        pos[1] = 1D array of vertical coordinates (m)
        (pos[0,n], pos[1,n]) is the position of particle 'n'
    grid_ranges: ndarray, dtype=float
        2D array holding the coordinates of the domain box in two dimensions
        grid_ranges[0] = [x_min, x_max]
        grid_ranges[1] = [z_min, z_max]
    grid_steps: ndarray, dtype=float
        grid_steps[0] = horizontal grid step size in x (m)
        grid_steps[1] = vertical grid step size in z (m)
    
    Returns
    -------
    cells: ndarray, dtype=int
        2D array, holding the particle cell indices, i.e.
        cells[0] = 1D array of horizontal indices
        cells[1] = 1D array of vertical indices
        (cells[0,n], cells[1,n]) gives the cell of particle 'n'    
    rel_pos: ndarray, dtype=float
        2D array with relative cell positions corresponding to 'pos'  
    
    """
    
    x = pos[0]
    y = pos[1]
    cells = np.empty( (2,len(x)) , dtype = np.int64)
    rel_pos = np.empty( (2,len(x)) , dtype = np.float64 )
    # gridranges = arr [[x_min, x_max], [y_min, y_max]]
    rel_pos[0] = x - grid_ranges[0,0]
    rel_pos[1] = y - grid_ranges[1,0]
    cells[0] = np.floor(x/grid_steps[0]).astype(np.int64)
    cells[1] = np.floor(y/grid_steps[1]).astype(np.int64)
    
    rel_pos[0] = rel_pos[0] / grid_steps[0] - cells[0]
    rel_pos[1] = rel_pos[1] / grid_steps[1] - cells[1]
    
    return cells, rel_pos

@njit()
def weight_velocities_linear(i, j, a, b, u_n, v_n):
    """Linear interpolation two dimensional velocity field (weight based)
    
    Parameters
    ----------
    i: int
        Cell index in x
    j: int
        Cell index in y
    a: float
        Relative x-coordinate in cell [i,j]. Resides in interval [0,1).
    b: float
        Relative y-coordinate in cell [i,j]. Resides in interval [0,1).
    u_n: ndarray, dtype=float
        2D array holding the discretized x-component of the velocity field 
    v_n: ndarray, dtype=float
        2D array holding the discretized y-component of the velocity field 
    
    Returns
    -------
        tuple of floats
        [0]: Interpolated x-component of the velocity field
        [1]: Interpolated y-component of the velocity field
    
    """
    
    return a * u_n[i + 1, j] + (1 - a) * u_n[i, j], \
           b * v_n[i, j + 1] + (1 - b) * v_n[i, j]
         
@njit()
def bilinear_weight(i, j, a, b, f):
    """Bilinear interpolation of 2D scalar field (weight based)
    
    The interpolation is given for the normalized position [a,b]
    in a cell with 4 corners 
    [i, j+1]    [i+1, j+1]
    [i, j]      [i+1, j]    
    
    Parameters
    ----------
    i: int
        Cell index in x
    j: int
        Cell index in y
    a: float
        Relative x-coordinate in cell [i,j]. Resides in interval [0,1).
    b: float
        Relative y-coordinate in cell [i,j]. Resides in interval [0,1).
    f: ndarray, dtype=float
        2D array holding the discretized field
    
    Returns
    -------
        float
        Bilinear interpolation of the field f at the given coordinates
    
    """
    
    return a * (b * f[i+1, j+1] + (1 - b) * f[i+1, j]) + \
            (1 - a) * (b * f[i, j+1] + (1 - b) * f[i, j])

@njit()
def interpolate_velocity_from_cell_bilinear(cells, rel_pos,
                                            grid_vel, grid_no_cells):
    """Bilin. interpol. of 2D velocity grid at particle cells and rel. pos.
    
    Adjusted for periodic boundary conditions in x and
    solid BC in z.
    
    Parameters
    ----------
    cells: ndarray, dtype=int
        2D array, holding the particle cell indices, i.e.
        cells[0] = 1D array of horizontal indices
        cells[1] = 1D array of vertical indices
        (cells[0,n], cells[1,n]) gives the cell of particle 'n'    
    rel_pos: ndarray, dtype=float
        2D array with relative, normalized cell positions
        corresponding to 'pos'.
    grid_vel: ndarray, dtype=float
        Discretized velocity field (m/s) of dry air.
        grid_vel[0] is a 2D array holding the x-components of the vel. field
        projected onto the grid cell surface centers.
        grid_vel[1] is a 2D array holding the z-components of the vel. field
        projected onto the grid cell surface centers.
    grid_no_cells: ndarray, dtype=int
        grid_no_cells[0] = number of grid cells in x (horizontal)
        grid_no_cells[1] = number of grid cells in z (vertical)
        
    Returns
    -------
        tuple of floats
        [0]: Interpolated x-component of the velocity field
        [1]: Interpolated y-component of the velocity field
    
    """
    
    no_pt = len(rel_pos[0])
    vel_ipol = np.empty( (2, no_pt), dtype = np.float64 )
    u, v = (0., 0.)
    for n in range( no_pt ):
        i = cells[0,n]
        j = cells[1,n]
        weight_x = rel_pos[0,n]
        weight_y = rel_pos[1,n]
        if j >= 0:
            if ( j == 0 and weight_y <= 0.5 ):
                u, v = weight_velocities_linear(i, j, weight_x, weight_y,
                                                grid_vel[0], grid_vel[1])
            elif ( j == (grid_no_cells[1] - 1) and weight_y >= 0.5 ):
                u, v = weight_velocities_linear(i, j, weight_x, weight_y,
                                                grid_vel[0], grid_vel[1])
            else:
                if weight_y > 0.5:
                    u = bilinear_weight(i, j,
                                        weight_x, weight_y - 0.5, grid_vel[0])
                else:
                    u = bilinear_weight(i, j - 1,
                                        weight_x, weight_y + 0.5, grid_vel[0])
            if weight_x > 0.5:
                v = bilinear_weight(i, j,
                                    weight_x - 0.5, weight_y, grid_vel[1])
            else:
                v = bilinear_weight(i - 1, j,
                                    weight_x + 0.5, weight_y, grid_vel[1])
        
        vel_ipol[0,n] = u
        vel_ipol[1,n] = v
    return vel_ipol

@njit()
def interpolate_velocity_from_position_bilinear(pos, grid_vel, grid_no_cells,
                                                grid_ranges, grid_steps):
    """Bilinear interpol. of 2D velocity grid field at particle positions
    
    Adjusted for periodic boundary conditions in x and
    solid BC in z.
    
    Parameters
    ----------
    pos: ndarray, dtype=float
        2D array, where
        pos[0] = 1D array of horizontal coordinates (m)
        pos[1] = 1D array of vertical coordinates (m)
        (pos[0,n], pos[1,n]) is the position of particle 'n'
    grid_vel: ndarray, dtype=float
        Discretized velocity field (m/s) of dry air.
        grid_vel[0] is a 2D array holding the x-components of the vel. field
        projected onto the grid cell surface centers.
        grid_vel[1] is a 2D array holding the z-components of the vel. field
        projected onto the grid cell surface centers.
    grid_no_cells: ndarray, dtype=int
        grid_no_cells[0] = number of grid cells in x (horizontal)
        grid_no_cells[1] = number of grid cells in z (vertical)
    grid_ranges: ndarray, dtype=float
        2D array holding the coordinates of the domain box in two dimensions
        grid_ranges[0] = [x_min, x_max]
        grid_ranges[1] = [z_min, z_max]
    grid_steps: ndarray, dtype=float
        grid_steps[0] = horizontal grid step size in x (m)
        grid_steps[1] = vertical grid step size in z (m)

    Returns
    -------
        tuple of floats
        [0]: Interpolated x-component of the velocity field
        [1]: Interpolated y-component of the velocity field
    
    """
    
    cells, rel_pos = compute_cell_and_relative_position(pos, grid_ranges,
                                                        grid_steps)
    return interpolate_velocity_from_cell_bilinear(cells, rel_pos,
                grid_vel, grid_no_cells)

def update_grid_r_l_np(m_w, xi, cells, grid_r_l, grid_mass_dry_inv,
                       active_ids, id_list):
    """Update the atmospheric liquid water mixing ratio grid from particles
    
    Parameters
    ----------
    m_w: ndarray, dtype=float
        1D array holding the particle water masses (1E-18 kg)
        This array gets updated by the function.
    xi: ndarray, dtype=float
        1D array holding the particle multiplicities
    cells: ndarray, dtype=int
        2D array, holding the particle cell indices, i.e.
        cells[0] = 1D array of horizontal indices
        cells[1] = 1D array of vertical indices
        (cells[0,n], cells[1,n]) gives the cell of particle 'n' 
    grid_r_l: ndarray, dtype=float
        2D array holding the discretized atmos. liquid water mixing ratio
    grid_mass_dry_inv: ndarray, dtype=float
        2D array holding 1/m_dry, where m_dry is the inverse
        discretized atmos. dry air mass in each cell (kg)
    active_ids: ndarray, dtype=bool
        1D mask-array. Each particle gets a flag 'True' or 'False', defining
        if it still resides in the simulation domain or has already hit the
        ground and is thereby removed from the simulation
    id_list: ndarray, dtype=float
        1D array holding the ordered particle IDs.
        Other arrays, like 'm_w', 'm_s', 'xi' etc. refer to this list.
        I.e. 'm_w[n]' is the water mass of particle with ID 'id_list[n]'
    
    Returns
    -------
        tuple of floats
        [0]: Interpolated x-component of the velocity field
        [1]: Interpolated y-component of the velocity field
    
    """
    
    grid_r_l.fill(0.0)
    for ID in id_list[active_ids]:
        grid_r_l[cells[0,ID], cells[1,ID]] += m_w[ID] * xi[ID]
    grid_r_l *= 1.0E-18 * grid_mass_dry_inv
update_grid_r_l = njit()(update_grid_r_l_np)

#%% GRID CLASS

class Grid:
    """Grid class with discretization parameters and atmospheric fields

    For spatial discretization, we use a rectangular C-staggered
    Arakawa grid with cells [i,j] in spatial directions (x, z). The extend
    in the third dimension (y) has fixed size. The system is
    two-dimensional in the sense that all field properties are
    invariant under translation in y-direction.
    
    c ----- w ----- c ----- w ----- c
    |               |               |
    |               |               |
    u       x       u       x       u
    |     [0,1]     |     [1,1]     |
    |               |               |
    c ----- w ----- c ----- w ----- c
    |               |               |
    |               |               |
    u       x       u       x       u
    |     [0,0]     |     [1,0]     |
    |               |               |    
    c ----- w ----- c ----- w ----- c
    
    cells [i,j], c: corners, x: centers, (u, w): velocity components
    
    In the default case (ICMW 2012, Test Case 1, Mulhbauer et al. 2013),
    the domain has periodic boundary conditions in x and solid boundary
    conditions in z.
    Since the applied Numba package is not (yet) able to deal with
    class objects, the atmospheric fields are sampled in
    'grid_scalar_fields' and 'grid_mat_prop', when passed to the
    simulation algorithm.

    Attributes
    ----------
    no_cells: ndarray, dtype=int
        no_cells[0] = number of grid cells in x (horizontal)
        no_cells[1] = number of grid cells in z (vertical)
    no_cells_tot : int
        Total number of grid cells
    steps: ndarray, dtype=float
        steps[0] = horizontal grid step size in x (m)
        steps[1] = vertical grid step size in z (m)
    step_y: float
        Horizontal grid step size in y (m)
    volume_cell: float
        Grid cell volume (m^3). All grid cells have the same volume.
    ranges: ndarray, dtype=float
        2D array holding the coordinates of the domain box in two dimensions
        ranges[0] = [x_min, x_max]
        ranges[1] = [z_min, z_max]
    sizes: ndarray, dtype=float
        1D array holding the domain sizes (m)
        sizes[0] = [x_max - x_min]
        sizes[1] = [z_max - z_min]
    corners: ndarray, dtype=float
        Positions of the grid cell corners (marked 'c' in the sketch, s.a.)
        corners[0] is a 2D array holding the x-components, such that
        corners[0][i,j] is the x-component of the bottom left corner of
        cell [i,j]
        corners[1][i,j] is the z-component of the bottom left corner of
        cell [i,j]
    centers: ndarray, dtype=float
        Positions of the grid cell centers (marked 'x' in the sketch, s.a.)
        centers[0] is a 2D array holding the x-components, such that
        centers[0][i,j] is the x-component of the center of cell [i,j]
        centers[1][i,j] is the z-component of the center of cell [i,j]
    surface_centers: list of ndarray, dtype=float
        Positions of the grid cell surface centers.
        (marked 'u' and 'w' in the sketch, s.a.)
        surface_centers[0][0][i,j] is the x-component of the center
        of the left surface of cell [i,j] (marked by 'u')
        surface_centers[0][1][i,j] is the z-component of the center
        of the left surface of cell [i,j] (marked by 'u')
        surface_centers[1][0][i,j] is the x-component of the center
        of the bottom surface of cell [i,j] (marked by 'w')
        surface_centers[1][1][i,j] is the z-component of the center
        of the bottom surface of cell [i,j] (marked by 'w')
    pressure: ndarray, dtype=float
        2D array of the discretized atmos. pressure field (Pa)
    temperature: ndarray, dtype=float
        2D array of the discretized atmos. temperature field (K)
    potential_temperature: ndarray, dtype=float
        2D array of the discretized atmos. dry potential temperature
        field (K) := T (p_ref / p_dry)^(kappa_dry), where
        kappa_dry = R_dry / c_dry, where R_dry = specific gas constant
        of dry air, c_dry = specific isobaric heat capacity of dry air.
    mass_density_air_dry: ndarray, dtype=float
        2D array of the discretized atmos. dry mass density field (kg/m^3)
    mass_density_fluid: ndarray, dtype=float
        2D array of the discretized fluid mass density in each cell (kg/m^3)
    rho_dry_inv: ndarray, dtype=float
        2D array: 1 / mass_density_air_dry
    mass_dry_inv: ndarray, dtype=float
        2D array: 1 / mass_dry,
        where mass_dry = mass_density_air_dry * volume_cell
    mixing_ratio_water_vapor: ndarray, dtype=float
        2D array of the discretized water vapor mixing ratio field (-)
    mixing_ratio_water_liquid: ndarray, dtype=float
        2D array of the discretized liquid water mixing ratio field (-)
    saturation_pressure: ndarray, dtype=float
        2D array of the discretized saturation pressure (vapor-liquid) (Pa)
    saturation: ndarray, dtype=float
        2D array of the discretized saturation field (vapor-liquid) (-)
        saturation = partial pressure of water vapor / saturation pressure
    velocity: ndarray, dtype=float
        Discretized velocity field (m/s) of dry air.
        Positions for the projection marked by 'u' and 'w' above.
        velocity[0] is a 2D array holding the x-components of the vel. field
        projected onto the grid cell surface centers 'u'.
        velocity[1] is a 2D array holding the z-components of the vel. field
        projected onto the grid cell surface centers 'w'.
    mass_flux_air_dry: ndarray, dtype=float
        Discretized mass flux density field (m/s) of dry air.
        Positions for the projection marked by 'u' and 'w' above.
        mass_flux_air_dry[0] is a 2D array holding the x-components of
        the flux field projected onto the grid cell surface centers 'u'.
        mass_flux_air_dry[1] is a 2D array holding the z-components of
        the flux field projected onto the grid cell surface centers 'w'.
    heat_of_vaporization: ndarray, dtype=float
        2D array of the discretized heat of vaporization (J/kg)
    thermal_conductivity: ndarray, dtype=float
        2D array of the discretized thermal conductivity of air (W/(m K))
    diffusion_constant: ndarray, dtype=float
        2D array of the discretized diffusion coefficent of water vapor
        in air (m^2/s)
    surface_tension: ndarray, dtype=float
        2D array of the discretized surface tension of water (N/m)
    specific_heat_capacity: ndarray, dtype=float
        2D array of the specific heat capacity of moist air (J/(kg K))
    viscosity: ndarray, dtype=float
        2D array of the dynamic viscosity in air (Pa s)
    p_ref: float
        Reference pressure for the potential temperature.
    p_ref_inv: float
        1 / p_ref

    """
    
    # initialize with arguments in paranthesis of __init__
    def __init__(self,
                 grid_ranges, # (m), as list [ [x_min, x_max], [z_min, z_max]] 
                 grid_steps, # in meter as list [dx, dz]
                 dy, # in meter
                 u_field = u_rot_field, v_field = v_rot_field,
                 temperature_field = temperature_field_linear,
                 pressure_field = pressure_field_exponential): # m/s
        self.no_cells =\
            np.array( compute_no_grid_cells_from_step_sizes(grid_ranges,
                                                            grid_steps) )
        self.no_cells_tot = self.no_cells[0] * self.no_cells[1]
        self.steps = np.array( grid_steps )
        self.step_y = dy
        self.volume_cell = grid_steps[0] * grid_steps[1] * dy
        self.ranges = np.array( grid_ranges )
        self.ranges[:,1] = self.ranges[:,0] + self.steps * self.no_cells
        self.sizes = np.array( [ self.ranges[0,1] - self.ranges[0,0],
                                 self.ranges[1,1] - self.ranges[1,0] ]  )
        corners_x = np.linspace(0.0, self.sizes[0], self.no_cells[0] + 1)\
                    + self.ranges[0,0]
        corners_y = np.linspace(0.0, self.sizes[1], self.no_cells[1] + 1)\
                    + self.ranges[1,0]
        self.corners = np.array(
                           np.meshgrid(corners_x, corners_y, indexing = 'ij'))
        # get the grid centers (in 2D)
        self.centers = [self.corners[0][:-1,:-1] + 0.5 * self.steps[0],
                        self.corners[1][:-1,:-1] + 0.5 * self.steps[1]]
        self.pressure = np.zeros_like(self.centers[0])
        self.temperature = np.zeros_like(self.centers[0])
        self.potential_temperature = np.zeros_like(self.centers[0])
        self.mass_density_air_dry = np.zeros_like(self.centers[0])
        self.mixing_ratio_water_vapor = np.zeros_like(self.centers[0])
        self.mixing_ratio_water_liquid = np.zeros_like(self.centers[0])
        self.saturation_pressure = np.zeros_like(self.centers[0])
        self.saturation = np.zeros_like(self.centers[0])
        # for the normal velocities in u-direction, 
        # take the x-positions and shift the y-positions by half a y-step etc.
        pos_vel_u = [self.corners[0], self.corners[1] + 0.5 * self.steps[1]]
        pos_vel_w = [self.corners[0] + 0.5 * self.steps[0], self.corners[1]]
        # self.surface_centers[0] =
        # position where the normal velocity in x is projected onto the cell
        # self.surface_centers[1] =
        # position where of normal velocity in z is projected onto the cell
        self.surface_centers = [ pos_vel_u, pos_vel_w ]
        self.set_analytic_velocity_field_and_discretize(u_field, v_field)
        self.mass_flux_air_dry = np.zeros_like(self.velocity)
        # if the temperature field is given as discrete grid,
        # set default field first and change grid.pressure manually later
        self.set_analytic_temperature_field_and_discretize(temperature_field)
        # if the pressure field is given as discrete grid,
        # set default field first and change grid.pressure manually later
        self.set_analytic_pressure_field_and_discretize(pressure_field)
        
        # material properties
        self.heat_of_vaporization = np.zeros_like(self.centers[0])
        self.thermal_conductivity = np.zeros_like(self.centers[0])
        self.diffusion_constant = np.zeros_like(self.centers[0])
        self.surface_tension = np.zeros_like(self.centers[0])
        self.specific_heat_capacity = np.zeros_like(self.centers[0])
        self.viscosity = np.zeros_like(self.centers[0])
        self.mass_density_fluid = np.zeros_like(self.centers[0])
        self.rho_dry_inv = np.zeros_like(self.centers[0])
        self.mass_dry_inv = np.zeros_like(self.centers[0])
        
        self.p_ref = 1.0E5
        self.p_ref_inv = 1.0E-5
        
        ### CONVERSIONS cell <-> location
        #   For now, we have a rect. grid with constant steps step_x, step_y
        #   for all cells, i.e. the cell number can be calc. from a posi. (x,y)
    def compute_cell(self, x, y):
        # gridranges = arr [[x_min, x_max], [y_min, y_max]]
        x = x - self.ranges[0,0]
        y = y - self.ranges[1,0]

        return np.array(
            [math.floor(x/self.steps[0]) , math.floor(y/self.steps[1])])
    
    def compute_cell_and_relative_location(self, x, y):
        # gridranges = arr [[x_min, x_max], [y_min, y_max]]
        x = x - self.ranges[0,0] 
        y = y - self.ranges[1,0]
        i = np.floor(x/self.steps[0]).astype(int)
        j = np.floor(y/self.steps[1]).astype(int)

        return np.array( [i, j] ) , np.array( [ x / self.steps[0] - i,
                                                y / self.steps[1] - j] )
    
    # function to get the particle location from cell number and rel. loc.
    def compute_location(self, i, j, rloc_x, rloc_y):
        x = (i + rloc_x) * self.steps[0] + self.ranges[0][0]
        y = (j + rloc_y) * self.steps[1] + self.ranges[1][0]
        return np.array( [x, y] )
    
    ### VELOCITY INTERPOLATION
    #     'Standard field'
    def analytic_velocity_field_u(self, x, y):
        omega = 0.3
        return -omega * y
    
    def analytic_velocity_field_v(self, x, y):
        omega = 0.3
        return omega * x
    
    def set_analytic_temperature_field_and_discretize(self, T_field_):
        self.analytic_temperature_field = T_field_
        self.temperature = T_field_( *self.centers )
        
    def set_analytic_pressure_field_and_discretize(self, p_field_):
        self.analytic_temperature_field = p_field_
        self.pressure = p_field_( *self.centers )
    
    #     u_field and v_field have to be functions of (x,y)
    def set_analytic_velocity_field_and_discretize(self, u_field_, v_field_):
        self.analytic_velocity_field = [u_field_, v_field_]
        self.velocity =\
            np.array(
                    [self.analytic_velocity_field[0](*self.surface_centers[0]),
                     self.analytic_velocity_field[1](*self.surface_centers[1])]
                    )
    
    def interpolate_velocity_from_location_linear(self, x, y):
        n, rloc = self.compute_cell_and_relative_location(x, y)
        u, v = weight_velocities_linear(*n, *rloc, *self.velocity)
        return u, v    
    
    def interpolate_velocity_from_cell_linear(self,i,j,rloc_x,rloc_y):
        return weight_velocities_linear(i, j, rloc_x, rloc_y, *self.velocity)

    # adjusted for period. bound cond. in x and solid BC in z (BC=PS)
    def interpolate_velocity_from_cell_bilinear(self, i, j,
                                                weight_x, weight_y):
        if ( j == 0 and weight_y <= 0.5):
            u, v = self.interpolate_velocity_from_cell_linear(
                            i, j, weight_x, weight_y)
        elif ( j == (self.no_cells[1] - 1) and weight_y >= 0.5):
            u, v = self.interpolate_velocity_from_cell_linear(
                            i, j, weight_x, weight_y)
        else:
            if weight_y > 0.5:
                u = bilinear_weight(i, j,
                                    weight_x, weight_y - 0.5, self.velocity[0])
            else:
                u = bilinear_weight(i, j - 1,
                                    weight_x, weight_y + 0.5, self.velocity[0])
        if weight_x > 0.5:
            v = bilinear_weight(i, j,
                                weight_x - 0.5, weight_y, self.velocity[1])
        else:
            v = bilinear_weight(i - 1, j,
                                weight_x + 0.5, weight_y, self.velocity[1])
        return u, v
        
    def interpolate_velocity_from_location_bilinear(self, x, y):
        n, rloc = self.compute_cell_and_relative_location(x, y)
        return self.interpolate_velocity_from_cell_bilinear(*n, *rloc)

    def update_material_properties(self):
        self.thermal_conductivity =\
            mat.compute_thermal_conductivity_air(self.temperature)
        self.diffusion_constant =\
            mat.compute_diffusion_constant(self.temperature, self.pressure)
        self.heat_of_vaporization =\
            mat.compute_heat_of_vaporization(self.temperature)
        self.surface_tension =\
            mat.compute_surface_tension_water(self.temperature)        
        self.specific_heat_capacity =\
            atm.compute_specific_heat_capacity_air_moist(
                                          self.mixing_ratio_water_vapor)
        self.viscosity = mat.compute_viscosity_air(self.temperature)
        self.mass_density_fluid = self.mass_density_air_dry\
                                  * (1 + self.mixing_ratio_water_vapor)

    ### PRINT GRID INFORMATION
    def print_info(self):
        print('')
        print('grid information:')
        print('grid ranges [x_min, x_max] [z_min, z_max]:')
        print(self.ranges)
        print('number of cells:', self.no_cells)
        print('grid steps:', self.steps)
    
    ### PLOTTING
        
    def plot_thermodynamic_scalar_profiles_vertical_average(self):
        fields = [self.pressure, self.temperature, self.mass_density_air_dry,
                  self.saturation,
                  self.mixing_ratio_water_vapor,
                  self.mixing_ratio_water_liquid]
                  
        field_names = ['pressure', 'temperature',
                       'mass_density_air_dry', 'saturation', 
                       'mixing_ratio_water_vapor', 'mixing_ratio_water_liquid']
        nfields = len(fields)
        ncols = 2
        nrows = int(np.ceil( nfields/ncols ))
        
        fields_avg = []
        
        for field in fields:
            fields_avg.append( field.mean(axis=0) )
        
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                               figsize = (10,5*nrows))
        
        n = 0
        for i in range(nrows):
            for j in range(ncols):
                field = fields_avg[n]
                ax[i,j].plot( field, self.centers[1][0] )
                ax[i,j].set_title( field_names[n] )
                ax[i,j].grid()
                n += 1
              
        fig.tight_layout()
    
    def plot_thermodynamic_scalar_fields(self, no_ticks_ = [5,5],
                                         t = 0, fig_dir = None):
        fields = [self.pressure * 0.01, self.temperature,
                  self.potential_temperature, self.mass_density_air_dry,
                  self.saturation, self.saturation_pressure * 0.01,
                  self.mixing_ratio_water_vapor*1000,
                  self.mixing_ratio_water_liquid*1000]
                  
        field_names = ['pressure', 'temperature', 'potential temperature',
                       'mass_density_air_dry',
                       'saturation', 'saturation pressure',
                       'mixing ratio water vapor', 'mixing ratio water liquid']
        unit_names = ['hPa', 'K', 'K', r'$\mathrm{kg/m^3}$', '-',
                      'hPa', 'g/kg', 'g/kg']
        nfields = len(fields)
        ncols = 2
        nrows = int(np.ceil( nfields/ncols ))
        
        tick_ranges_ = self.ranges
        
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                               figsize = (10,4*nrows))
        n = 0
        for i in range(nrows):
            for j in range(ncols):
                field = fields[n]
                if n == 7:
                    field_min = 0.001
                else:
                    field_min = field.min()
                field_max = field.max()
                if n in [0,1,2,3,5]:
                    cmap = 'coolwarm'
                    alpha = None
                else:
                    cmap = 'rainbow'
                    alpha = 0.7
                    
                CS = ax[i,j].pcolorfast(*self.corners, field, cmap=cmap,
                                        alpha=alpha, edgecolor='face',
                                        vmin=field_min, vmax=field_max)
                CS.cmap.set_under('white')
                ax[i,j].set_title( field_names[n] + ' (' + unit_names[n] + ')')
                ax[i,j].set_xticks( np.linspace( tick_ranges_[0,0],
                                                 tick_ranges_[0,1],
                                                 no_ticks_[0] ) )
                ax[i,j].set_yticks( np.linspace( tick_ranges_[1,0],
                                                 tick_ranges_[1,1],
                                                 no_ticks_[1] ) )
                if n == 7:
                    cbar = fig.colorbar(CS, ax=ax[i,j], extend = 'min')
                else: cbar = fig.colorbar(CS, ax=ax[i,j])
                n += 1
              
        fig.tight_layout()
        if fig_dir is not None:
            fig.savefig(fig_dir + f'scalar_fields_grid_t_{int(t)}.png')
    
    def plot_scalar_field_2D(self, field_,
                             no_ticks_ = [5,5],
                             no_contour_colors_ = 10, no_contour_lines_ = 5,
                             colorbar_fraction_=0.046, colorbar_pad_ = 0.02):
        
        tick_ranges_ = self.ranges
        plot_scalar_field_2D( *self.centers, field_,
                             tick_ranges_, no_ticks_,
                             no_contour_colors_, no_contour_lines_,
                             colorbar_fraction_, colorbar_pad_)
        
    # velocity = [ velocity_x[i,j], velocity_z[i,j] ] for 2D
    def plot_velocity_field_at_cell_surface(
            self, no_major_xticks=10, no_major_yticks=10, 
            no_arrows_u=10, no_arrows_v=10, ARROW_SCALE = 40.0,
            ARROW_WIDTH= 0.002, gridopt = 'minor'):

        # assume we have 21 cells and we want about 10 labeled x-ticks
        # i.e. we will label the left corner
        # of cells 0,2,4,6,8,10,12,14,16,18,20,22
        # for 20 cells, we will label left corn. of 0,2,4,6,8,10,12,14,16,18,20

        if no_major_xticks < self.no_cells[0]:
            # take no_major_xticks - 1 to get the right spacing
            # in dimension of full cells widths
            tick_every_x = self.no_cells[0] // (no_major_xticks - 1)
        else:
            tick_every_x = 1

        if no_major_yticks < self.no_cells[1]:
            tick_every_y = self.no_cells[1] // (no_major_yticks - 1)
        else:
            tick_every_y = 1

        if no_arrows_u < self.no_cells[0]:
            arrow_every_x = self.no_cells[0] // (no_arrows_u - 1)
        else:
            arrow_every_x = 1

        if no_arrows_v < self.no_cells[1]:
            arrow_every_y = self.no_cells[1] // (no_arrows_v - 1)
        else:
            arrow_every_y = 1

        vel_pos_u = self.surface_centers[0]
        vel_pos_w = self.surface_centers[1]
        u_n = self.velocity[0]
        w_n = self.velocity[1]
        
        fig = plt.figure(figsize=(8,8), dpi = 81)
        ax = plt.gca()
        ax.quiver(vel_pos_u[0][::arrow_every_y,::arrow_every_x],
                  vel_pos_u[1][::arrow_every_y,::arrow_every_x],          
                  u_n[::arrow_every_y,::arrow_every_x],
                  np.zeros_like(u_n[::arrow_every_y,::arrow_every_x]),
                  pivot = 'mid',
                  width = ARROW_WIDTH, scale = ARROW_SCALE,
                  zorder = 3)
        ax.quiver(vel_pos_w[0][::arrow_every_y,::arrow_every_x],
                  vel_pos_w[1][::arrow_every_y,::arrow_every_x],
                  np.zeros_like(w_n[::arrow_every_y,::arrow_every_x]),
                  w_n[::arrow_every_y,::arrow_every_x], pivot = 'mid',
                  width = ARROW_WIDTH, scale = ARROW_SCALE,
                  zorder = 3)
        
        ax.set_xticks(self.corners[0][::tick_every_x,0])
        ax.set_yticks(self.corners[1][0,::tick_every_y])
        ax.set_xticks(self.corners[0][:,0], minor = True)
        ax.set_yticks(self.corners[1][0,:], minor = True)

        if gridopt == 'minor':
            ax.grid(which='minor', zorder=0)
        else:
            ax.grid(which='major', zorder=0)

        ax.set_xlabel('horiz. pos. [m]')
        ax.set_ylabel('vert. pos. [m]')

    def plot_velocity_field_centered(
            self, no_major_xticks=10, no_major_yticks=10, 
            no_arrows_u=10, no_arrows_v=10, ARROW_SCALE = 40.0,
            ARROW_WIDTH= 0.002, gridopt = 'minor'):
        
        centered_u_field = ( self.velocity[0][0:-1,0:-1]\
                             + self.velocity[0][1:,0:-1] ) * 0.5
        centered_w_field = ( self.velocity[1][0:-1,0:-1]\
                             + self.velocity[1][0:-1,1:] ) * 0.5
        
        self.plot_external_field_list_output_centered(
                 [centered_u_field, centered_w_field],
                 no_major_xticks, no_major_yticks,
                 no_arrows_u, no_arrows_v, ARROW_SCALE, ARROW_WIDTH, gridopt)       

    def plot_mass_flux_field_centered(self, no_major_xticks=10,
                                      no_major_yticks=10, 
                                      no_arrows_u=10, no_arrows_v=10,
                                      ARROW_SCALE = 40.0, ARROW_WIDTH= 0.002,
                                      gridopt = 'minor'):
        
        centered_u_field = ( self.mass_flux_air_dry[0][0:-1,0:-1]\
                             + self.mass_flux_air_dry[0][1:,0:-1] ) * 0.5
        centered_w_field = ( self.mass_flux_air_dry[1][0:-1,0:-1]\
                             + self.mass_flux_air_dry[1][0:-1,1:] ) * 0.5
        
        self.plot_external_field_list_output_centered( [centered_u_field,
                                                        centered_w_field],
                                                       no_major_xticks,
                                                       no_major_yticks,
                                                       no_arrows_u,
                                                       no_arrows_v,
                                                       ARROW_SCALE,
                                                       ARROW_WIDTH,
                                                       gridopt)     

    # field f returns f_x, f_y
    def plot_external_field_function_list_output(self, f,
                                     no_major_xticks=10, no_major_yticks=10, 
                                     no_arrows_u=10, no_arrows_v=10,
                                     ARROW_SCALE=40, ARROW_WIDTH=0.002,
                                     gridopt = 'minor'):

        # assume we have 21 cells and we want about 10 labeled x-ticks
        # i.e. we will label cell the left corner
        # of cell 0,2,4,6,8,10,12,14,16,18,20,22
        # for 20 cells, we will label left corn. of 0,2,4,6,8,10,12,14,16,18,20

        if no_major_xticks < self.no_cells[0]:
            # take no_major_xticks - 1 to get the right spacing
            # in dimension of full cells widths
            tick_every_x = self.no_cells[0] // (no_major_xticks - 1)
        else:
            tick_every_x = 1

        if no_major_yticks < self.no_cells[1]:
            tick_every_y = self.no_cells[1] // (no_major_yticks - 1)
        else:
            tick_every_y = 1

        if no_arrows_u < self.no_cells[0]:
            arrow_every_x = self.no_cells[0] // (no_arrows_u - 1)
        else:
            arrow_every_x = 1

        if no_arrows_v < self.no_cells[1]:
            arrow_every_y = self.no_cells[1] // (no_arrows_v - 1)
        else:
            arrow_every_y = 1

        fig = plt.figure(figsize=(8,8), dpi = 92)
        ax = plt.gca()
        ax.quiver(
            self.corners[0][::arrow_every_y,::arrow_every_x],
            self.corners[1][::arrow_every_y,::arrow_every_x],
            *f(self.corners[0][::arrow_every_y,::arrow_every_x],
               self.corners[1][::arrow_every_y,::arrow_every_x]),
                  pivot = 'mid',
                  width = ARROW_WIDTH, scale = ARROW_SCALE, zorder=3 )
        ax.set_xticks(self.corners[0][::tick_every_x,0])
        ax.set_yticks(self.corners[1][0,::tick_every_y])
        ax.set_xticks(self.corners[0][:,0], minor = True)
        ax.set_yticks(self.corners[1][0,:], minor = True)

        if gridopt == 'minor':
            ax.grid(which='minor', zorder=0)
        else:
            ax.grid(which='major', zorder=0)
        ax.set_xlabel('horiz. pos. [m]')
        ax.set_ylabel('vert. pos. [m]')
        
    # field f returns ARRAYS f_x[i,j], f_y[i,j]
    def plot_external_field_list_output_centered(self, f,
                                     no_major_xticks=10, no_major_yticks=10, 
                                     no_arrows_u=10, no_arrows_v=10,
                                     ARROW_SCALE=40, ARROW_WIDTH=0.002,
                                     gridopt = 'minor'):

        # assume we have 21 cells and we want about 10 labeled x-ticks
        # i.e. we will label cell the left corner
        # of cell 0,2,4,6,8,10,12,14,16,18,20,22
        # for 20 cells, we will label left corn. of 0,2,4,6,8,10,12,14,16,18,20

        if no_major_xticks < self.no_cells[0]:
            # take no_major_xticks - 1 to get the right spacing
            # in dimension of full cells widths
            tick_every_x = self.no_cells[0] // (no_major_xticks - 1)
        else:
            tick_every_x = 1

        if no_major_yticks < self.no_cells[1]:
            tick_every_y = self.no_cells[1] // (no_major_yticks - 1)
        else:
            tick_every_y = 1

        if no_arrows_u < self.no_cells[0]:
            arrow_every_x = self.no_cells[0] // (no_arrows_u - 1)
        else:
            arrow_every_x = 1

        if no_arrows_v < self.no_cells[1]:
            arrow_every_y = self.no_cells[1] // (no_arrows_v - 1)
        else:
            arrow_every_y = 1

        fig = plt.figure(figsize=(8,8), dpi = 92)
        ax = plt.gca()
        ax.quiver(
            self.centers[0][::arrow_every_y,::arrow_every_x],
            self.centers[1][::arrow_every_y,::arrow_every_x],
            f[0][::arrow_every_y,::arrow_every_x],
            f[1][::arrow_every_y,::arrow_every_x],
                  pivot = 'mid',
                  width = ARROW_WIDTH, scale = ARROW_SCALE, zorder=3 )
        ax.set_xticks(self.corners[0][::tick_every_x,0])
        ax.set_yticks(self.corners[1][0,::tick_every_y])
        ax.set_xticks(self.corners[0][:,0], minor = True)
        ax.set_yticks(self.corners[1][0,:], minor = True)
        ax.set_xlim(self.corners[0][0,0], self.corners[0][-1,0])
        ax.set_ylim(self.corners[1][0,0], self.corners[1][0,-1])

        if gridopt == 'minor':
            ax.grid(which='minor', zorder=0)
        else:
            ax.grid(which='major', zorder=0)
        ax.set_xlabel('horiz. pos. [m]')
        ax.set_ylabel('vert. pos. [m]')
        
    # the components of the external field f = (f_x, f_y) 
    # must be defined python functions
    # of the spatial variables (x,y): f_x(x,y), f_y (x,y)
    def plot_external_field_function(self,f_x, f_y, 
                                     no_major_xticks=10, no_major_yticks=10, 
                                     no_arrows_u=10, no_arrows_v=10,
                                     ARROW_SCALE=40, ARROW_WIDTH=0.002,
                                     gridopt = 'minor'):

        # assume we have 21 cells and we want about 10 labeled x-ticks
        # i.e. we will label cell the left corner
        # of cell 0,2,4,6,8,10,12,14,16,18,20,22
        # for 20 cells, we will label left corn. of 0,2,4,6,8,10,12,14,16,18,20

        if no_major_xticks < self.no_cells[0]:
            # take no_major_xticks - 1 to get the right spacing
            # in dimension of full cells widths
            tick_every_x = self.no_cells[0] // (no_major_xticks - 1)
        else:
            tick_every_x = 1

        if no_major_yticks < self.no_cells[1]:
            tick_every_y = self.no_cells[1] // (no_major_yticks - 1)
        else:
            tick_every_y = 1

        if no_arrows_u < self.no_cells[0]:
            arrow_every_x = self.no_cells[0] // (no_arrows_u - 1)
        else:
            arrow_every_x = 1

        if no_arrows_v < self.no_cells[1]:
            arrow_every_y = self.no_cells[1] // (no_arrows_v - 1)
        else:
            arrow_every_y = 1

        fig = plt.figure(figsize=(8,8), dpi = 92)
        ax = plt.gca()
        ax.quiver(
            self.corners[0][::arrow_every_y,::arrow_every_x],
            self.corners[1][::arrow_every_y,::arrow_every_x],
            f_x(self.corners[0][::arrow_every_y,::arrow_every_x],
                self.corners[1][::arrow_every_y,::arrow_every_x]),
            f_y(self.corners[0][::arrow_every_y,::arrow_every_x],
                self.corners[1][::arrow_every_y,::arrow_every_x]),
                  pivot = 'mid',
                  width = ARROW_WIDTH, scale = ARROW_SCALE, zorder=3 )
        ax.set_xticks(self.corners[0][::tick_every_x,0])
        ax.set_yticks(self.corners[1][0,::tick_every_y])
        ax.set_xticks(self.corners[0][:,0], minor = True)
        ax.set_yticks(self.corners[1][0,:], minor = True)
        if gridopt == 'minor':
            ax.grid(which='minor', zorder=0)
        else:
            ax.grid(which='major', zorder=0)
        ax.set_xlabel('horiz. pos. [m]')
        ax.set_ylabel('vert. pos. [m]')

    # trajectories must be a list or array of arrays:
    # trajectories[ [x1, y1  ] , [x2, y2 ] , [...] ]
    def plot_field_and_trajectories(self,f_x, f_y, trajectories,
                                     no_major_xticks=10, no_major_yticks=10, 
                                     no_arrows_u=10, no_arrows_v=10, 
                                    fig_n = 1, LW = 0.9, ARROW_SCALE = 80.0):
        
        if no_major_xticks < self.no_cells[0]:
            tick_every_x = self.no_cells[0] // no_major_xticks
        else:
            tick_every_x = 1

        if no_major_yticks < self.no_cells[1]:
            tick_every_y = self.no_cells[1] // no_major_yticks
        else:
            tick_every_y = 1

        if no_arrows_u < self.no_cells[0]:
            arrow_every_x = self.no_cells[0] // no_arrows_u
        else:
            arrow_every_x = 1

        if no_arrows_v < self.no_cells[1]:
            arrow_every_y = self.no_cells[1] // no_arrows_v
        else:
            arrow_every_y = 1

        if (self.sizes[0] == self.sizes[1]):
            figsize_x = 8
            figsize_y = 8
        else:
            size_ratio = self.sizes[1] / self.sizes[0]
            figsize_x = 8
            figsize_y = figsize_x * size_ratio
        fig = plt.figure(figsize=(figsize_x, figsize_y), dpi = 92)
        ax = plt.gca()
        ax.quiver(
            self.corners[0][::arrow_every_y,::arrow_every_x],
            self.corners[1][::arrow_every_y,::arrow_every_x],
            f_x(self.corners[0][::arrow_every_y,::arrow_every_x],
                self.corners[1][::arrow_every_y,::arrow_every_x]),
            f_y(self.corners[0][::arrow_every_y,::arrow_every_x],
                self.corners[1][::arrow_every_y,::arrow_every_x]),
                  pivot = 'mid',
                  width = 0.002, scale = ARROW_SCALE )
        for data_x_y in trajectories:
            ax.plot(data_x_y[0], data_x_y[1], linewidth = LW, linestyle='--')
        ax.set_xticks(self.corners[0][0,::tick_every_x])
        ax.set_yticks(self.corners[1][::tick_every_y,0])
        ax.set_xticks(self.corners[0][0], minor = True)
        ax.set_yticks(self.corners[1][:,0], minor = True)
        ax.grid(which='minor', zorder=0)
        ax.set_xlabel('horiz. pos. [m]')
        ax.set_ylabel('vert. pos. [m]')
