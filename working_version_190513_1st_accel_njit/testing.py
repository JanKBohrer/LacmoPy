#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 16:20:21 2019

@author: jdesk
"""
import numpy as np
import matplotlib.pyplot as plt

from init import dst_log_normal

def convert_sigma_star_to_sigma(mu_star, sigma_star):
    mu = mu_star * sigma_star**(0.5 * np.log(sigma_star))
    sigma = mu * np.sqrt( sigma_star**( np.log(sigma_star) ) - 1.0)
    return mu, sigma

def test_initial_size_distribution(R_s, xi, no_spcm, dst_par, fig_path = None):
    R_ = []
    f = []
    mu = dst_par[:,0]
    sigma = np.exp(dst_par[:,1])
    
    mus,sigma = convert_sigma_star_to_sigma(mu, sigma)
    xmin = np.maximum(4E-3, mu - np.sqrt(sigma/10))
    xmax = np.maximum(1E-3, mu + sigma*5)
    print("no super particles per cell and mode: ")
    print(no_spcm)
    print("no cells, no SP total:")
    print(len(R_s)//np.sum(no_spcm), len(R_s))
    print()
    
    unique, count = np.unique(xi, return_counts=True)
    # print(unique)
    # print(count)
    cnt = 0
    for i,c_ in enumerate(count):
        if c_>1:
            print("xi = ", unique[i], " occurs ", c_, " times")
            cnt += 1
    if cnt == 0:
        print("no xi values occur multiple times")
    
    print()
    print("mu, sigma, xmin, xmax")
    print(mu, sigma, xmin, xmax)
    print()
    for i,xmin_ in enumerate(xmin):
        R_.append( np.linspace(xmin_,xmax[i],1000) )
        f.append( dst_log_normal(R_[i], dst_par[i]) )
    
    idx = []
    xi1 = []
    Rs1 = []
    for n,spc_N in enumerate(no_spcm):
        if n == 0:
            idx1 = [ x for x in range(len(xi)) if x % np.sum(no_spcm) in range(no_spcm[n]) ]
        else:
            idx1 = [ x for x in range(len(xi)) if x % np.sum(no_spcm) in\
                        range( np.sum(no_spcm[0:n]), np.sum(no_spcm[0:n]) + no_spcm[n]) ]
        # print(idx1)
        idx.append(idx1)
        xi1.append(xi[idx1])
        Rs1.append(R_s[idx1])
        
    # print(l_)
    # print (R_s)
    # print (R_s[l_])
    # xi1 = xi[idx]
    # Rs1 = R_s[idx]
    
    # print(len(xi))
    # for xi1_ in xi1:
    #     print(len(xi1_))
    # for Rs1_ in Rs1:
    #     print(len(Rs1_))
    # print(len(R_s))
    # print(len(Rs1))
    
    # xi1 = np.hstack( (xi[::4], xi[1::4]) )
    # xi2 = np.hstack( (xi[2::4], xi[3::4]) )
    # Rs1 = np.hstack( (R_s[::2], R_s[1::2]) )
    # Rs2 = np.hstack( (R_s[2::2], R_s[3::2]) )
    
    # print (R_s)
    # print (Rs1)
    # print (Rs2)
    
    # print (xi)
    # print (xi1)
    # print (xi2)
    dst_par = np.array(dst_par)
    fig, axes = plt.subplots(ncols=len(no_spcm), nrows=2, figsize=(6*len(no_spcm),6) )
    for i,ax in enumerate(axes[0]):
        ax.plot(R_[i],f[i])
        ax.hist(Rs1[i], bins= 20, weights=xi1[i], density=True)
        ax.grid()
        ax.set_xscale("log")
        # ax.set_yscale("log")
        ax.set_xlim( [xmin[i],xmax[i]] )
        ax.set_xlabel("dry radius (mu)")
        ax.set_ylabel("PDF (-)")
        ax.set_title("mode " + str(i+1))
        # ax.set_xlim( [dst_par[i,0]-dst_par[i,1]/10, dst_par[i,0]+dst_par[i,1]*2] )
        # ax.set_ylim( [0.01,100] )
    ax = axes[1,0]
    ax.set_xlabel("multiplicity (-)")
    ax.set_ylabel("frequency")
    ax.set_title("occurance of xi")
    ax.hist(xi, bins = len(R_s))
    fig.suptitle("compare init. dry size distr. with the generating PDF")
    fig.tight_layout()
    if fig_path is not None:
        fig.savefig(fig_path)
