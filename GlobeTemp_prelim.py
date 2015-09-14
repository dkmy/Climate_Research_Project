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

# u is windspeed

def Globetemp(temp, rh, windspeed, dewpoint):
    temp = temp - 273.15

    C = math.pow((.315 * windspeed), 0.58)/ (5.3865 * math.pow(10, -8))
    S = 1366
    fdb = 1000
    fdif = 60
    zenith = math.radians(50)
    p = 1000
    ea = math.exp((17.67*(dewpoint - temp)/(dewpoint+243.5))) * (1.0007 + 0.00000346 * p) * 6.112 * math.exp((17.502*temp)/(240.97+temp))
    realEa = 0.575 * math.pow(ea, 1/7)

    B = S*(fdb/(2.268*math.pow(10,-8)*math.cos(zenith)) + (1.2/(5.67 * math.pow(10,-8))) * fdif) + realEa * math.pow(temp, 4)
    print (B)
    print ("zentith ", math.cos(zenith))
    print (C)

    Tg = (B * C * temp + 7680000)/(C+256000)
    return Tg



def BoundSearch():

    path_temp = '/Users/DavidKMYang/ClimateResearch/WBGT/corrected_gfdl_tasmax_nh/'
    os.chdir(path_temp)
    file_names_temp = glob.glob("*.mat")

    path_dewpoint = '/Users/DavidKMYang/ClimateResearch/WBGT/gfdl_dewpoint/'
    os.chdir(path_dewpoint)
    file_names_dewpoint = glob.glob("*.mat")

    path_rh = '/Users/DavidKMYang/ClimateResearch/WBGT/gfdl-esm2m-rh-nh/'
    os.chdir(path_rh)
    file_names_rh = glob.glob("*.mat")

    path_windspeed = '/Users/DavidKMYang/ClimateResearch/WBGT/gfdl_windspeed/'
    os.chdir(path_windspeed)
    file_names_windspeed = glob.glob("*.mat")


    for i in range(len(file_names_temp)):
        print (i)
        tempData_temp = scipy.io.loadmat(path_temp + file_names_temp[i])

        tempData_Lat_temp = tempData_temp[file_names_temp[i][:-4]+"_Lat"]
        tempData_Long_temp = tempData_temp[file_names_temp[i][:-4] + "_Long"]
        tempData_Val_temp = tempData_temp[file_names_temp[i][:-4] + "_Val"]

        tempData_dewpoint = scipy.io.loadmat(path_dewpoint + file_names_dewpoint[i])

        tempData_Lat_dewpoint = tempData_dewpoint[file_names_dewpoint[i][:-4]+"_Lat"]
        tempData_Long_dewpoint = tempData_dewpoint[file_names_dewpoint[i][:-4] + "_Long"]
        tempData_Val_dewpoint = tempData_dewpoint[file_names_dewpoint[i][:-4] + "_Val"]

        tempData_windspeed = scipy.io.loadmat(path_windspeed + file_names_windspeed[i])

        tempData_Lat_windspeed = tempData_windspeed[file_names_windspeed[i][:-4]+"_Lat"]
        tempData_Long_windspeed = tempData_windspeed[file_names_windspeed[i][:-4] + "_Long"]
        tempData_Val_windspeed = tempData_windspeed[file_names_windspeed[i][:-4] + "_Val"]

        tempData_rh = scipy.io.loadmat(path_rh + file_names_rh[i])
        tempData_rh = tempData_rh[file_names_rh[i][:-4]][0]

        globeT = tempData_rh[2]



        for k in range(len(tempData_rh[2])): #lat
            for j in range(len(tempData_rh[2][0])): #long
                for s in range(len(tempData_rh[2][0][0])): #day
                    # print ("hi")
                    # print (tempData_Val_temp[k][j][s])
                    # print (tempData_rh[2][k][j][s])
                    # print (tempData_Val_windspeed[k][j][s])
                    # print (tempData_Val_dewpoint[k][j][s])
                    globeT[k][j][s] = Globetemp(tempData_Val_temp[k][j][s], tempData_rh[2][k][j][s], tempData_Val_windspeed[k][j][s], tempData_Val_dewpoint[k][j][s])
                    break
            break
        break
        final_Lat_List = np.asarray(tempData_Lat_temp)
        final_Long_List = np.asarray(tempData_Long_temp)
        final_Val_List = np.asarray(globeT)

        # final_Total_List = np.asarray(final_Lat_List, final_Long_List, final_Val_List)
        scipy.io.savemat('/Users/DavidKMYang/ClimateResearch/WBGT/gfdl_globe/' + "globe" + file_names_rh[i][2:], mdict={"globe" + file_names_rh[i][2:][:-4] + "_Lat" : final_Lat_List, "globe" + file_names_rh[i][2:][:-4] + "_Long" : final_Long_List, "globe" + file_names_rh[i][2:][:-4] + "_Val" : final_Val_List})
        break


BoundSearch()

