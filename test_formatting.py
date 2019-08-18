#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 15:12:07 2019

@author: jdesk
"""

field_max = 1.2345
field_max *= 10
#field_max *= 1E-2

strings = []
strings.append(r"{0:.8g}".format(field_max))

strings.append(r"{0:2.2f}".format(field_max))
strings.append(r"{0:1.2f}".format(field_max))
strings.append(r"{0:0.2f}".format(field_max))
strings.append(r"{0:.2f}".format(field_max))

strings.append(r"{0:2.2g}".format(field_max))
strings.append(r"{0:1.2g}".format(field_max))
strings.append(r"{0:0.2g}".format(field_max))
strings.append(r"{0:.2g}".format(field_max))

strings.append(r"{0:2.2e}".format(field_max))
strings.append(r"{0:1.2e}".format(field_max))
strings.append(r"{0:0.2e}".format(field_max))
strings.append(r"{0:.2e}".format(field_max))

strings.append(r"{0:2.0e}".format(field_max))
strings.append(r"{0:1.0e}".format(field_max))
strings.append(r"{0:0.0e}".format(field_max))
strings.append(r"{0:.0e}".format(field_max))

#strings.append(r"{0:2.1f}".format(field_max))
#strings.append(r"{0:1.1f}".format(field_max))
#strings.append(r"{0:0.1f}".format(field_max))
#strings.append(r"{0:.1f}".format(field_max))
#
#strings.append(r"{0:2.1g}".format(field_max))
#strings.append(r"{0:1.1g}".format(field_max))
#strings.append(r"{0:0.1g}".format(field_max))
#strings.append(r"{0:.1g}".format(field_max))
#
#strings.append(r"{0:2.1e}".format(field_max))
#strings.append(r"{0:1.1e}".format(field_max))
#strings.append(r"{0:0.1e}".format(field_max))
#strings.append(r"{0:.1e}".format(field_max))

#str1 = 
#str2 = r"{0:0.1f}".format(field_max)
#str3 = r"{0:0.1f}".format(field_max)

print(strings)


#%%

import math
import numpy as np

field = 2.786*np.logspace(-3,3,20)

print(field)

field_max = field.max()
#field_max = 2.7593E6

oom = int(math.log10(field_max))

oom_factor = 10**(-oom)

#field_max_int_s = str( int(field_max) )

#oom2 = ( len(field_max) )

print(f"{field_max:.3g}", oom, f"{oom_factor:.3g}")

field_scal = field * oom_factor

print(field_scal)

field_string = []

for i,f in enumerate(field_scal):
    print(i, r"{0:.2f}".format(f))
    field_string.append(r"{0:.32g}".format(f))
    
print()    
print("field_string")
print(field_string)
print()    

field_rec = np.array(field_string, dtype = np.float64) * 10**(oom)    
print("field_rec")
print(field_rec)

print(field_rec - field)


