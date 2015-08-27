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

def DewPoint(temp, rh):
    temp = temp - 273.15
    if rh != 0:
        x = 243.04*(math.log(rh/100.0)+((17.625*temp)/(243.04+temp)))/(17.625-math.log(rh/100.0)-((17.625*temp)/(243.04+temp)))
    else:
        x = float('nan')
    return x

print ("rh_2047_01_01.mat"[2:][:-4])




path_dew = '/Users/DavidKMYang/ClimateResearch/WBGT/gfdl_dewpoint/'
os.chdir(path_dew)
file_names_dew = glob.glob("*.mat")


for i in range(len(file_names_dew)):
    print (i)
    tempData_temp = scipy.io.loadmat(path_dew + file_names_dew[i])
    tempData_Lat = tempData_temp[file_names_dew[i][:-4]+"_Lat"]
    tempData_Long = tempData_temp[file_names_dew[i][:-4] + "_Long"]
    tempData_Val = tempData_temp[file_names_dew[i][:-4] + "_Val"]

    print (tempData_Val)

    break
