__author__ = 'DavidKMYang'

#given a grid and time, average results of multiple models

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

def BoundSearch(Lat_Low, Lat_High, Long_Low, Long_High, year, month, day):
    path_ccsm4 = '/Users/DavidKMYang/ClimateResearch/WBGT/ccsm4_tasmax_nepal/'
    os.chdir(path_ccsm4)
    print ("tasmax_" + str(year) + "_0" + str(month) + "_01" +".mat")
    if month < 10:
        file_names_ccsm4 = glob.glob("tasmax_" + str(year) + "_0" + str(month) + "_01" +".mat")
    else:
        file_names_ccsm4 = glob.glob("tasmax_" + str(year) + "_" + str(month) + "_01"+".mat")

    path_gfdl_cm3 = '/Users/DavidKMYang/ClimateResearch/WBGT/cm3_tasmax_nepal/'
    os.chdir(path_gfdl_cm3)

    if month < 10:
        file_names_gfdl_cm3 = glob.glob("tasmax_" + str(year) + "_0" + str(month) + "*.mat")
    else:
        file_names_gfdl_cm3 = glob.glob("tasmax_" + str(year) + "_" + str(month) + "*.mat")

    print (file_names_ccsm4)
    for i in range(len(file_names_ccsm4)):
        lat_index = 0
        long_index = 0
        print (file_names_ccsm4[i])

        tempData_ccsm4 = scipy.io.loadmat(path_ccsm4 + file_names_ccsm4[i])
        tempData_ccsm4 = tempData_ccsm4[file_names_ccsm4[i][:-4]][0]

        tempData_gfdl_cm3 = scipy.io.loadmat(path_gfdl_cm3 + file_names_ccsm4[i])
        tempData_gfdl_cm3 = tempData_gfdl_cm3[file_names_ccsm4[i][:-4]][0]

        tempLatValueList = []
        tempBoundLatIndexList = []  # we just want the indices of the latitude that are in the bound
        for k in range(len(tempData_ccsm4[0])):
            tempLatValueList.append(tempData_ccsm4[0][k][0])
            if tempData_ccsm4[0][k][0] >= Lat_Low and tempData_ccsm4[0][k][0] <= Lat_High:
                tempBoundLatIndexList.append(k)

        tempLongValueList = tempData_ccsm4[1][0]
        tempBoundLongIndexList = []

        for k in range(len(tempLongValueList)):
            if tempData_ccsm4[1][0][k] >= Long_Low and tempData_ccsm4[1][0][k] <= Long_High:
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
                final_Val_List[i].append((tempData_ccsm4[2][i][k][day-1] + tempData_gfdl_cm3[2][i][k][day-1])/2) #firstday

        final_Lat_List = np.asarray(final_Lat_List)
        final_Long_List = np.asarray(final_Long_List)
        final_Val_List = np.asarray(final_Val_List)

        final_Total_List = np.asarray([final_Lat_List, final_Long_List, final_Val_List])
        scipy.io.savemat('testFile.mat', mdict={'final_Total_List': final_Total_List})

        flatTLat = np.array(final_Lat_List)
        flatTLon = np.array(final_Long_List)
        flatTData = np.array(final_Val_List)

        m = Basemap(width=10000000/2,height=7000000/2,
                    resolution='l',projection='stere',
                    lat_ts = 40, lat_0=(Lat_High+Lat_Low)/2, lon_0 = (Long_High+Long_Low)/2)

        lon, lat = np.meshgrid(flatTLon[0,:], flatTLat[:,0])
        x, y = m(lon,lat)
        cs = m.pcolor(x,y,np.squeeze(flatTData), vmin=250, vmax=350)

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
        plt.title('Mean Temperature in 2000-2004 Relative to 1990-1995')
        plt.show()
        print ("hello")
        break

print ("hi")
BoundSearch(29, 50, 50, 90, 1980, 1, 1)
print ("ey")




