__author__ = 'DavidKMYang'

import math


def DewPoint(temp, rh):
    temp = temp - 273.15
    if rh != 0:
        x = 243.04*(math.log(rh/100.0)+((17.625*temp)/(243.04+temp)))/(17.625-math.log(rh/100.0)-((17.625*temp)/(243.04+temp)))
    else:
        x = float('nan')
    return x

print (DewPoint(0, 0))