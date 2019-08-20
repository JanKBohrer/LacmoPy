#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 15:45:21 2019

@author: bohrer
"""

# INWORK: add title and ax labels
def plot_scalar_field_2D( grid_centers_x_, grid_centers_y_, field_,
                         tick_ranges_, no_ticks_=[5,5],
                         no_contour_colors_ = 10, no_contour_lines_ = 5,
                         colorbar_fraction_=0.046, colorbar_pad_ = 0.02):
    fig, ax = plt.subplots(figsize=(8,8))

    contours = plt.contour(grid_centers_x_, grid_centers_y_,
                           field_, no_contour_lines_, colors = 'black')
    ax.clabel(contours, inline=True, fontsize=8)
    CS = ax.contourf( grid_centers_x_, grid_centers_y_,
                     field_,
                     levels = no_contour_colors_,
                     vmax = field_.max(),
                     vmin = field_.min(),
                    cmap = plt.cm.coolwarm)
    ax.set_xticks( np.linspace( tick_ranges_[0,0], tick_ranges_[0,1],
                                no_ticks_[0] ) )
    ax.set_yticks( np.linspace( tick_ranges_[1,0], tick_ranges_[1,1],
                                no_ticks_[1] ) )
    plt.colorbar(CS, fraction=colorbar_fraction_ , pad=colorbar_pad_)