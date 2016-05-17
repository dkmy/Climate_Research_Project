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


def BiasCorrect(month):

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

    # tot_Diff_List = [] #contains all the values in all the grid over the course of X years of a particular month

    tempData = scipy.io.loadmat(path_gfdl_cm3 + file_names_gfdl_cm3[0])
    tempData = tempData[file_names_gfdl_cm3[0][:-4]][0]


    tot_Diff_List = np.zeros((len(tempData[0]), len(tempData[0][0])))


    for i in range(len(file_names_ncep)):

        tempData_gfdl = scipy.io.loadmat(path_gfdl_cm3 + file_names_gfdl_cm3[i]) #historical
        tempData_gfdl = tempData_gfdl[file_names_gfdl_cm3[i][:-4]][0]

        tempData_ncep = scipy.io.loadmat(path_ncep + file_names_ncep[i]) #model
        tempData_ncep = tempData_ncep[file_names_ncep[i][:-4]][0]

        DiffList = [] #should contain values in all the grid over the course of a particular year of a particualr month

        for j in range(len(tempData_gfdl[0])):
            for k in range(len(tempData_gfdl[0][0])):
                diff = np.mean(tempData_ncep[2][j][k]) - np.mean(tempData_gfdl[2][j][k])
                tot_Diff_List[j][k] += diff



    tot_Diff_List = tot_Diff_List/len(file_names_ncep)

    # print (sum_List[1])

    scipy.io.savemat('/Users/DavidKMYang/ClimateResearch/WBGT/BiasCorrected_v2/' + 'biasCorrection_'+ str(month) + ".mat", mdict={'biasCorrection_'+ str(month): tot_Diff_List})


    # for i in range(len(flatTLon)):
    #     for j in range(len(flatTLon[0])):
    #         flatTLon[i][j] -= 180
    # # m = Basemap(width=10000000,height=10000000,
    # #             resolution='l',projection='kav7',
    # #             lat_ts = 10, lat_0=30, lon_0 = 30)
    # m = Basemap(projection = 'kav7', lon_0 = 0, resolution = 'l')
    #
    #
    # lon, lat = np.meshgrid(flatTLon[0,:], flatTLat[:,0])
    # x, y = m(lon,lat)
    # cs = m.pcolor(x,y,np.squeeze(flatTData), vmin=-15, vmax=15)
    # m.drawmapboundary(fill_color='0.3')
    # # Add Grid Lines
    # m.drawparallels(np.arange(-80., 81., 10.), labels=[1,0,0,0], fontsize=10)
    # m.drawmeridians(np.arange(-180., 181., 10.), labels=[0,0,0,1], fontsize=10)
    #
    # # Add Coastlines, States, and Country Boundaries
    # m.drawcoastlines()
    # m.drawstates()
    # m.drawcountries()
    #
    # # Add Colorbar
    # cbar = m.colorbar(cs, location='bottom', pad="10%")
    #
    # # Add Title
    # plt.title('n')
    #
    # plt.show()

for i in range(12):
    BiasCorrect(i+1)
# BiasCorrect(10)