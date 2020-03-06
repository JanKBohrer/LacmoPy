#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 16:38:22 2019

@author: bohrer
"""


import sys

print ('Number of arguments:', len(sys.argv), 'arguments.')
print ('Argument List:', str(sys.argv))


if len(sys.argv) > 1:
    seed_SIP_gen = int(sys.argv[1])
    print("seed SIP gen entered = ", seed_SIP_gen)
    
print("seed_SIP_gen")    
print(seed_SIP_gen, type(seed_SIP_gen))    

print(50*seed_SIP_gen)