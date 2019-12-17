#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 14:48:42 2019

@author: bohrer
"""


def change_dict(dic):
    dic["a"] = 6
    dic["zeta"] = "new entry"
    
dict1 = {"a" : 5, "b" : 7}

print(dict1)

change_dict(dict1)

print(dict1)


dict1["alpha"] = 15.4


print(dict1)


A = dict1['a']


keys = ['a', 'b', 'alpha']

#values = map(dict1.get, keys)

list1 = [dict1.get(key) for key in keys]

A,B,C = list1