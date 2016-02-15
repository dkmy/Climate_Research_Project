__author__ = 'DavidKMYang'

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

    path_gfdl_cm3 = '/Users/DavidKMYang/ClimateResearch/WBGT/gfdl_tasmax_nh/'
    os.chdir(path_gfdl_cm3)

    if month < 10:
        file_names_gfdl_cm3 = glob.glob("*" + "0" + str(month) + "_01.mat")
    else:
        file_names_gfdl_cm3 = glob.glob("*" + str(month) + "_01.mat")

    path_ncep = '/Users/DavidKMYang/ClimateResearch/WBGT/ncep_tasmax_nh/'
    os.chdir(path_ncep)

    if month < 10:
        file_names_ncep = glob.glob("*" + "0" + str(month) + "_01.mat")
    else:
        file_names_ncep = glob.glob("*" + str(month) + "_01.mat")

    tot_Diff_List = [] #contains all the values in all the grid over the course of X years of a particular month

    for i in range(len(file_names_ncep)):
        print (i)

        tempData_gfdl = scipy.io.loadmat(path_gfdl_cm3 + file_names_gfdl_cm3[i])
        tempData_gfdl = tempData_gfdl[file_names_gfdl_cm3[i][:-4]][0]

        tempData_ncep = scipy.io.loadmat(path_ncep + file_names_ncep[i])
        tempData_ncep = tempData_ncep[file_names_ncep[i][:-4]][0]

        DiffList = [] #should contain values in all the grid over the course of a particular year of a particualr month

        for k in range(len(tempData_ncep[0])):
            tempDiffList = [] # should contain values in one grid of one month of a particualr lyear
            for j in range(len(tempData_ncep[0][0])):
                modelMean = np.mean(tempData_gfdl[2][k][j])
                actualMean = np.mean(tempData_ncep[2][k][j])

                tempDiffList.append(actualMean - modelMean) #actual  - model
            DiffList.append(tempDiffList) #
            sum_List = DiffList

        tot_Diff_List.append(DiffList)

    # sum_List = [tot_Diff_List[0]]

    for i in range(len(sum_List)):
        for k in range(len(sum_List[0])):
            sum_List[i][k] = 0


    for i in range(len(file_names_ncep)):
        for k in range(len(tot_Diff_List[0])):
            for j in range(len(tot_Diff_List[0][0])):
                sum_List[k][j] = sum_List[k][j] + tot_Diff_List[i][k][j]

    print (len(sum_List))
    print (len(sum_List[0]))
    for i in range(len(sum_List)):
        for k in range(len(sum_List[0])):
            sum_List[i][k] = sum_List[i][k]/len(tot_Diff_List)
    # print (sum_List)
    # scipy.io.savemat('/Users/DavidKMYang/ClimateResearch/WBGT/BiasCorrections/Temp/' + 'biasCorrection_'+ str(month) + ".mat", mdict={'biasCorrection_'+ str(month): sum_List})

# for i in range(12):
#     BoundSearch(i+1)
BoundSearch(1)