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





def BoundSearch():

    path_temp = '/Users/DavidKMYang/ClimateResearch/WBGT/corrected_gfdl_tasmax_nh/'
    os.chdir(path_temp)
    file_names_temp = glob.glob("*.mat")

    path_wetbulb = '/Users/DavidKMYang/ClimateResearch/WBGT/gfdl_wetbulb/'
    os.chdir(path_wetbulb)
    file_names_wetbulb = glob.glob("*.mat")

    path_globe = '/Users/DavidKMYang/ClimateResearch/WBGT/gfdl_globe/'
    os.chdir(path_globe)
    file_names_globe = glob.glob("*.mat")



    for i in range(len(file_names_temp)):
        print (i)
        tempData_temp = scipy.io.loadmat(path_temp + file_names_temp[i])

        tempData_Lat_temp = tempData_temp[file_names_temp[i][:-4]+"_Lat"]
        tempData_Long_temp = tempData_temp[file_names_temp[i][:-4] + "_Long"]
        tempData_Val_temp = tempData_temp[file_names_temp[i][:-4] + "_Val"]

        tempData_globe = scipy.io.loadmat(path_globe + file_names_globe[i])

        tempData_Lat_globe = tempData_globe[file_names_globe[i][:-4]+"_Lat"]
        tempData_Long_globe = tempData_globe[file_names_globe[i][:-4] + "_Long"]
        tempData_Val_globe = tempData_globe[file_names_globe[i][:-4] + "_Val"]

        tempData_wetbulb = scipy.io.loadmat(path_wetbulb + file_names_wetbulb[i])

        tempData_Lat_wetbulb = tempData_wetbulb[file_names_wetbulb[i][:-4]+"_Lat"]
        tempData_Long_wetbulb = tempData_wetbulb[file_names_wetbulb[i][:-4] + "_Long"]
        tempData_Val_wetbulb = tempData_wetbulb[file_names_wetbulb[i][:-4] + "_Val"]



        globeT = tempData_Val_wetbulb

        for k in range(len(tempData_Val_wetbulb)): #lat
            for j in range(len(tempData_Val_wetbulb[0])): #long
                for s in range(len(tempData_Val_wetbulb[0][0])): #day
                    # print ("hi")
                    # print (tempData_Val_temp[k][j][s])
                    # print (tempData_rh[2][k][j][s])
                    # print (tempData_Val_windspeed[k][j][s])
                    # print (tempData_Val_dewpoint[k][j][s])
                    globeT[k][j][s] = 0.1* tempData_Val_temp[k][j][s] + 0.7* tempData_Val_wetbulb[k][j][s] + 0.2 * tempData_Val_globe[k][j][s]

        final_Lat_List = np.asarray(tempData_Lat_temp)
        final_Long_List = np.asarray(tempData_Long_temp)
        final_Val_List = np.asarray(globeT)

        # final_Total_List = np.asarray(final_Lat_List, final_Long_List, final_Val_List)
        scipy.io.savemat('/Users/DavidKMYang/ClimateResearch/WBGT/gfdl_wbgt/' + "wbgt" + file_names_globe[i][2:], mdict={"wbgt" + file_names_globe[i][2:][:-4] + "_Lat" : final_Lat_List, "wbgt" + file_names_globe[i][2:][:-4] + "_Long" : final_Long_List, "wbgt" + file_names_globe[i][2:][:-4] + "_Val" : final_Val_List})



BoundSearch()

