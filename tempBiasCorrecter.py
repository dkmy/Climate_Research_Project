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



def find_nearest(arr,v):
    return (np.abs(arr - v)).argmin()

def BoundSearch(month):

    path_bias = '/Users/DavidKMYang/ClimateResearch/WBGT/BiasCorrections_nh/Temp/'
    correction = scipy.io.loadmat(path_bias + "biasCorrection_" + str(month) + ".mat")


    path_gfdl = '/Users/DavidKMYang/ClimateResearch/WBGT/gfdl_tasmax_nh/'
    os.chdir(path_gfdl)
    if month < 10:
        file_names_gfdl = glob.glob("*" + "0" + str(month) + "_01.mat")
    else:
        file_names_gfdl = glob.glob("*" + str(month) + "_01.mat")

    correction = correction['biasCorrection_' + str(month)]


    for i in range(len(file_names_gfdl)):
        print (i)

        tempData_gfdl = scipy.io.loadmat(path_gfdl + file_names_gfdl[i])
        tempData_gfdl = tempData_gfdl[file_names_gfdl[i][:-4]][0]


        corrected_Val = tempData_gfdl[2]

        # print ("correction: " , correction[0][0])

        # print (len(corrected_Val[17][0]))


        for k in range(len(corrected_Val)):
            for j in range(len(corrected_Val[0])):
                for s in range(len(corrected_Val[0][0])):
                    corrected_Val[k][j][s] = corrected_Val[k][j][s] + correction[k][j]


        final_Lat_List = np.asarray(tempData_gfdl[0])
        final_Long_List = np.asarray(tempData_gfdl[1])
        final_Val_List = np.asarray(corrected_Val)

        print (final_Val_List)
        final_Total_List = np.asarray(final_Lat_List, final_Long_List, final_Val_List)
        scipy.io.savemat('/Users/DavidKMYang/ClimateResearch/WBGT/corrected_gfdl_tasmax_nh/' + file_names_gfdl[i] + "_corrected.mat", mdict={file_names_gfdl[i] + "_corrected_Lat" : final_Lat_List, file_names_gfdl[i] + "_corrected_Long" : final_Long_List, file_names_gfdl[i] + "_corrected_Val" : final_Val_List})
        break
    return

for i in range(12):
    BoundSearch(i+1)
    break
