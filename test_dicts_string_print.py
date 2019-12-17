#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 11:57:05 2019

@author: bohrer
"""

dict1 = {
        'a'         : 15.7,
        'str'       : 'solute_type'
        }


string1 = "a " + dict1['str']
string2 = f"a {dict1['str']}"

print(string1)
print(string2)

