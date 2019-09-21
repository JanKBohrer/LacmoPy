import os
import numpy as np
# import os
# from datetime import datetime
# import timeit

import constants as c
from microphysics import compute_R_p_w_s_rho_p_AS
from microphysics import compute_R_p_w_s_rho_p_NaCl
from microphysics import compute_radius_from_mass_vec
from file_handling import load_grid_and_particles_full,\
                          load_grid_scalar_fields\
                          

#                     plot_particle_size_spectra
# from integration import compute_dt_max_from_CFL
#from grid import compute_no_grid_cells_from_step_sizes

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
mpl.rcParams.update(plt.rcParamsDefault)
#mpl.use("pdf")
mpl.use("pgf")

import matplotlib.ticker as mticker


#import numpy as np

from microphysics import compute_mass_from_radius_vec
import constants as c

#import numba as 

from plotting import cm2inch
from plotting import generate_rcParams_dict
from plotting import pgf_dict, pdf_dict
plt.rcParams.update(pgf_dict)
#plt.rcParams.update(pdf_dict)

from file_handling import load_grid_and_particles_full,\
                          load_grid_scalar_fields\

from analysis import sample_masses, sample_radii
from analysis import sample_masses_per_m_dry , sample_radii_per_m_dry
from analysis import plot_size_spectra_R_Arabas, generate_size_spectra_R_Arabas

from plotting_fcts_MA import plot_size_spectra_R_Arabas_MA

#%%

x = np.arange(10)
fig, ax = plt.subplots()

ax.plot(x, np.sin(x))
ax.set_title(r"1000.222 \si{\micro\meter}")
fig.savefig("test1000_siunitx_error.pdf")

