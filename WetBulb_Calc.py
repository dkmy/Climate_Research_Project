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


def WetBulb(temp, rh):
    temp = temp - 273.15
    if rh != 0:
        x = (-5.806 + 0.672 * temp - 0.006 * math.pow(temp, 2)+(0.061 + 0.004 * temp + 99*10-6 * math.pow(temp, 2)) * rh + (-33*10-6 - 5*10-6 *temp - 1*10-7 * math.pow(temp, 2)) * math.pow(rh, 2))
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
        print (i)
        tempData_temp = scipy.io.loadmat(path_temp + file_names_temp[i])

        tempData_Lat = tempData_temp[file_names_temp[i][:-4]+"_Lat"]
        tempData_Long = tempData_temp[file_names_temp[i][:-4] + "_Long"]
        tempData_Val = tempData_temp[file_names_temp[i][:-4] + "_Val"]

        tempData_rh = scipy.io.loadmat(path_rh + file_names_rh[i])
        tempData_rh = tempData_rh[file_names_rh[i][:-4]][0]

        dewpoint = tempData_rh[2]

        for k in range(len(tempData_rh[2])): #lat
            for j in range(len(tempData_rh[2][0])): #long
                for s in range(len(tempData_rh[2][0][0])): #day
                    dewpoint[k][j][s] = WetBulb(tempData_Val[k][j][s], tempData_rh[2][k][j][s])

        final_Lat_List = np.asarray(tempData_Lat)
        final_Long_List = np.asarray(tempData_Long)
        final_Val_List = np.asarray(dewpoint)

        # final_Total_List = np.asarray(final_Lat_List, final_Long_List, final_Val_List)
        scipy.io.savemat('/Users/DavidKMYang/ClimateResearch/WBGT/gfdl_wetbulb/' + "wetbulb" + file_names_rh[i][2:], mdict={"wetbulb" + file_names_rh[i][2:][:-4] + "_Lat" : final_Lat_List, "wetbulb" + file_names_rh[i][2:][:-4] + "_Long" : final_Long_List, "wetbulb" + file_names_rh[i][2:][:-4] + "_Val" : final_Val_List})



BoundSearch()

