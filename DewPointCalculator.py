__author__ = 'DavidKMYang'

#given a grid and time, average results of multiple models
import h5py
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import scipy
import os
import scipy.io
import csv
import numpy as np
import glob
import pandas as pd
from numpy import genfromtxt
import math


def DewPoint(temp, rh):
    temp = temp - 273.15
    if rh != 0:
        x = 243.04*(math.log(rh/100.0)+((17.625*temp)/(243.04+temp)))/(17.625-math.log(rh/100.0)-((17.625*temp)/(243.04+temp)))
    else:
        x = float('nan')
    return x


def BoundSearch():

    path_temp = '/Users/DavidKMYang/ClimateResearch/WBGT/corrected_gfdl_tasmax_nh/'
    os.chdir(path_temp)
    file_names_temp = glob.glob("*.mat")

    path_rh = '/Users/DavidKMYang/ClimateResearch/WBGT/gfdl-esm2m-rh-nh/'
    os.chdir(path_rh)
    file_names_rh = glob.glob("*.mat")


    for i in range(len(file_names_temp)):

        tempData_temp = scipy.io.loadmat(path_temp + file_names_temp[i])

        tempData_Lat = tempData_temp[file_names_temp[i][:-4]+"_Lat"]
        tempData_Long = tempData_temp[file_names_temp[i][:-4] + "_Long"]
        tempData_Val = tempData_temp[file_names_temp[i][:-4] + "_Val"]

        tempData_rh = scipy.io.loadmat(path_rh + file_names_rh[i])
        tempData_rh = tempData_rh[file_names_rh[i][:-4]][0]

        print (tempData_rh[2])
        break


BoundSearch()

