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
    path_uas = '/Users/DavidKMYang/ClimateResearch/WBGT/gfdl-esm2m-uas-nh/'
    os.chdir(path_uas)
    file_names_uas = glob.glob("*.mat")

    path_vas = '/Users/DavidKMYang/ClimateResearch/WBGT/gfdl-esm2m-vas-nh/'
    os.chdir(path_vas)
    file_names_vas = glob.glob("*.mat")



    for i in range(len(file_names_uas)):
        print (i)
        tempData_uas = scipy.io.loadmat(path_uas + file_names_uas[i])
        tempData_uas = tempData_uas[file_names_uas[i][:-4]][0]
        # print (file_names_uas[i][:-4])
        # print (tempData_uas[2])
        # break


        tempData_vas = scipy.io.loadmat(path_vas + file_names_vas[i])
        tempData_vas = tempData_vas[file_names_vas[i][:-4]][0]


        windspeed = tempData_uas[2]

        for k in range(len(tempData_vas[2])): #lat
            for j in range(len(tempData_vas[2][0])): #long
                for s in range(len(tempData_vas[2][0][0])): #day
                    windspeed[k][j][s] = math.sqrt(math.pow(tempData_uas[2][k][j][s], 2) + math.pow(tempData_vas[2][k][j][s], 2))

        final_Lat_List = np.asarray(tempData_vas[0])
        final_Long_List = np.asarray(tempData_vas[1])
        final_Val_List = np.asarray(windspeed)

        # final_Total_List = np.asarray(final_Lat_List, final_Long_List, final_Val_List)
        scipy.io.savemat('/Users/DavidKMYang/ClimateResearch/WBGT/gfdl_windspeed/' + "windspeed" + file_names_uas[i][2:], mdict={"windspeed" + file_names_uas[i][2:][:-4] + "_Lat" : final_Lat_List, "windspeed" + file_names_uas[i][2:][:-4] + "_Long" : final_Long_List, "windspeed" + file_names_uas[i][2:][:-4] + "_Val" : final_Val_List})

BoundSearch()






