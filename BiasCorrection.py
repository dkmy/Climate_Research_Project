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

        tempData_gfdl = scipy.io.loadmat(path_gfdl_cm3 + file_names_gfdl_cm3[i]) #historical
        tempData_gfdl = tempData_gfdl[file_names_gfdl_cm3[i][:-4]][0]

        tempData_ncep = scipy.io.loadmat(path_ncep + file_names_ncep[i]) #model
        tempData_ncep = tempData_ncep[file_names_ncep[i][:-4]][0]

        DiffList = [] #should contain values in all the grid over the course of a particular year of a particualr month

        for k in range(len(tempData_ncep[0])):
            tempDiffList = [] # should contain values in one grid of one month of a particualr lyear
            for j in range(len(tempData_ncep[0][0])):
                modelMean = np.mean(tempData_gfdl[2][k][j])
                actualMean = np.mean(tempData_ncep[2][k][j])
                tempDiffList.append(actualMean - modelMean) #actual  - model
                # tempDiffList.append(actualMean) #actual  - model

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


    for i in range(len(sum_List)):
        for k in range(len(sum_List[0])):
            sum_List[i][k] = sum_List[i][k]/len(tot_Diff_List)

    # for i in range(len(sum_List)):
    #     for k in range(len(sum_List[0])):
    #         if abs(sum_List[i][k]) > 20:
    #             print (sum_List[i])
    #             print (i)
    #             print (k)

    array = sum_List
    # print (sum_List[1])

    # scipy.io.savemat('/Users/DavidKMYang/ClimateResearch/WBGT/BiasCorrections/Temp/' + 'biasCorrection_'+ str(month) + ".mat", mdict={'biasCorrection_'+ str(month): sum_List})
    # def Plot(array, correct_Bool):
    # path_wbgt = '/Users/DavidKMYang/ClimateResearch/WBGT/gfdl_tasmax_nh/'
    # path_wbgt_corrected = '/Users/DavidKMYang/ClimateResearch/WBGT/corrected_gfdl_tasmax_nh/'
    #
    # tempData_wbgt = scipy.io.loadmat(path_wbgt + 'tasmax_2005_07_01.mat')
    # tempData_wbgt = tempData_wbgt['tasmax_2005_07_01'][0]
    #
    # tempData_wbgt_corrected = scipy.io.loadmat(path_wbgt_corrected + 'tasmax_2005_07_01.mat_corrected.mat')
    # # tempData_wbgt_corrected = tempData_wbgt_corrected['tasmax_2005_07_01.mat_corrected'][0]
    #
    # print (tempData_wbgt_corrected)
    #
    flatTLat = np.array(tempData_ncep[0])
    flatTLon = np.array(tempData_ncep[1])
    flatTData = np.array(array)
    #
    # if correct_Bool:
    #     flatTLat = np.array(tempData_wbgt_corrected['tasmax_2005_07_01.mat_corrected_Lat'])
    #     flatTLon = np.array(tempData_wbgt_corrected['tasmax_2005_07_01.mat_corrected_Long'])
    #     # flatTData = np.array(tempData_wbgt_corrected['tasmax_2005_07_01.mat_corrected_Val'])


    for i in range(len(flatTLon)):
        for j in range(len(flatTLon[0])):
            flatTLon[i][j] -= 180



    # m = Basemap(width=10000000,height=10000000,
    #             resolution='l',projection='kav7',
    #             lat_ts = 10, lat_0=30, lon_0 = 30)
    m = Basemap(projection = 'kav7', lon_0 = 0, resolution = 'l')


    lon, lat = np.meshgrid(flatTLon[0,:], flatTLat[:,0])
    x, y = m(lon,lat)
    cs = m.pcolor(x,y,np.squeeze(flatTData), vmin=250, vmax=350)
    m.drawmapboundary(fill_color='0.3')
    # Add Grid Lines
    m.drawparallels(np.arange(-80., 81., 10.), labels=[1,0,0,0], fontsize=10)
    m.drawmeridians(np.arange(-180., 181., 10.), labels=[0,0,0,1], fontsize=10)

    # Add Coastlines, States, and Country Boundaries
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()

    # Add Colorbar
    cbar = m.colorbar(cs, location='bottom', pad="10%")

    # Add Title
    plt.title('n')

    plt.show()

# for i in range(12):
#     BoundSearch(i+1)
BoundSearch(1)