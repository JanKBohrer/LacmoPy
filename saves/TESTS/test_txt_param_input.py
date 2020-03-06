#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 11:36:13 2019

@author: bohrer
"""

# https://stackoverflow.com/questions/42644025/read-parameters-from-txt

import json

with open('input_params_test.dat') as file:
    inparams = json.load(file)

print(type(inparams))
print(inparams)

