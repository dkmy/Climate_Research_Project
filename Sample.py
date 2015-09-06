__author__ = 'DavidKMYang'

import math

import scipy
import os
import scipy.io
import csv
import numpy as np
import glob
import pandas as pd
from numpy import genfromtxt
import math
import os
from sympy import *

#
# x = symbols('x')
#
# from sympy import roots, solve_poly_system
#
# print (solve(4*x**3 - x +3, x))
#
# print (solve(4*x**4 + 3*x**2 - 4*x + 42, rational = False))
#
# # coeff = [3, -1, 0, 4]
# # print (np.roots(coeff))

path_temp = '/Users/DavidKMYang/ClimateResearch/WBGT/corrected_gfdl_tasmax_nh/'
os.chdir(path_temp)
file_names_temp = glob.glob("*.mat")

print (len(file_names_temp))

path_rh = '/Users/DavidKMYang/ClimateResearch/WBGT/gfdl-esm2m-rh-nh/'
os.chdir(path_rh)
file_names_rh = glob.glob("*.mat")

print (len(file_names_rh))