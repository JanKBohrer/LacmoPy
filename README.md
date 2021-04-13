# CloudMP
Simulation program for warm cloud microphysics using Lagrangian (discrete) particle methods.

The program enables the simulation of a drizzling stratocumulus cloud in a two-dimensional kinematic framework, built for the 8th International Cloud Modeling Workshop 2012 (Test case 1 in Muhlbauer et al., Bulletin of the American Meteorological Society 94, 45 (2013))

Requires Python V3.7 and Numba package V0.47 or more recent.

To perform and evaluate a simulation:

1. Set required parameters in the config file "config.py". It is possible (and recommended) to include a spin-up period of 2h = 7200s.
2. Execute the simulation: Set the number of independent simulation runs (starting from independent grids) and the first random seeds for grid generation and simulation in "run_lacmo.sh". Then execute the script "run_lacmo.sh" in the terminal (may require to be set as executable via chmod +x).
3. If required, evaluate data, using "run_gen_data.sh". This will include a statistical analysis over the independent simulation runs.
4. Data can be plotted with the python scripts "plot_results_paper.py" See comments in there.

In case of any questions, please contact the repository admin (Jan Bohrer).
