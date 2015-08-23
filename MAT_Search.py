__author__ = 'DavidKMYang'
import os
import scipy.io
import csv
import numpy as np
import glob
import pandas as pd
from numpy import genfromtxt


def find_nearest(arr,v):
    return (np.abs(arr - v)).argmin()

def access_Measurement(lat, long, year):
    path_ccsm4 = '/Users/DavidKMYang/ClimateResearch/WBGT/ccsm4_tasmax_nepal/'

    os.chdir(path_ccsm4)
    file_names_ccsm4 = glob.glob("tasmax_" + str(year)+"*.mat")

    for i in range(len(file_names_ccsm4)):
        lat_index = 0
        long_index = 0
        print (file_names_ccsm4[i])
        tempData = scipy.io.loadmat(path_ccsm4 + file_names_ccsm4[i])
        tempData = tempData[file_names_ccsm4[i][:-4]][0]

        tempLatList = []
        for k in range(len(tempData[0])):
            tempLatList.append(tempData[0][k][0])
        tempLatList = np.asarray(tempLatList)
        lat_index = find_nearest(tempLatList, lat)

        tempLongList = tempData[1][0]
        tempLongList = np.asarray(tempLongList)

        long_index = find_nearest(tempLongList, long)
        print (tempLatList[lat_index])
        print (tempLongList[long_index])
        print (tempData[2][lat_index][long_index])


        access_Measurement(25, 30, 2001)

def BoundSearch(Lat_Low, Lat_High, Long_Low, Long_High):
    path_ccsm4 = '/Users/DavidKMYang/ClimateResearch/WBGT/ccsm4_tasmax_nepal/'

    os.chdir(path_ccsm4)
    file_names_ccsm4 = glob.glob("*.mat")
    # file_names_ccsm4 = glob.glob("tasmax_" + str(year)+"*.mat")

    for i in range(len(file_names_ccsm4)):
        lat_index = 0
        long_index = 0
        print (file_names_ccsm4[i])
        tempData = scipy.io.loadmat(path_ccsm4 + file_names_ccsm4[i])
        tempData = tempData[file_names_ccsm4[i][:-4]][0]

        tempLatValueList = []
        tempBoundLatIndexList = []  # we just want the indices of the latitude that are in the bound
        for k in range(len(tempData[0])):
            tempLatValueList.append(tempData[0][k][0])
            if tempData[0][k][0] >= Lat_Low and tempData[0][k][0] <= Lat_High:
                tempBoundLatIndexList.append(k)


        tempLongValueList = tempData[1][0]
        tempBoundLongIndexList = []

        for k in range(len(tempLongValueList)):
            if tempData[1][0][k] >= Long_Low and tempData[1][0][k] <= Long_High:
                tempBoundLongIndexList.append(k)

        final_Lat_List = []
        final_Long_List = []
        final_Val_List = []

        for i in range(len(tempBoundLatIndexList)):
            final_Lat_List.append([])
            for k in range(len(tempBoundLongIndexList)):
                final_Lat_List[i].append(tempLatValueList[tempBoundLatIndexList[i]])

        for i in range(len(tempBoundLatIndexList)):
            final_Long_List.append([])
            for k in range(len(tempBoundLongIndexList)):
                final_Long_List[i].append(tempLongValueList[tempBoundLongIndexList[k]])

        for i in range(len(tempBoundLatIndexList)):
            final_Val_List.append([])
            for k in range(len(tempBoundLongIndexList)):
                final_Val_List[i].append(tempData[2][i][k][0]) #firstday

        final_Lat_List = np.asarray(final_Lat_List)
        final_Long_List = np.asarray(final_Long_List)
        final_Val_List = np.asarray(final_Val_List)

        final_Total_List = np.asarray([final_Lat_List, final_Long_List, final_Val_List])
        scipy.io.savemat('testFile.mat', mdict={'final_Total_List': final_Total_List})


        # for i in range(len(tempBoundLatIndexList)):
        #     for k in range(len(tempBoundLongIndexList)):
        #         final_List.append([tempLatValueList[tempBoundLatIndexList[i]], tempLongList[tempBoundLongIndexList[k]], tempData[2][i][k][0]]) #get first day
        #
        # print (len(tempBoundLatIndexList), len(tempBoundLongIndexList))
        # print (final_List)
        break

BoundSearch(29, 50, 50, 90)







