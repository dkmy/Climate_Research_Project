__author__ = 'DavidKMYang'


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
        # tempData = scipy.io.loadmat(path_ccsm4 + "tasmax_1980_08_01")
        tempData = tempData[file_names_ccsm4[i][:-4]][0]
        # tempData =tempData['tasmax_1980_08_01'][0]

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
        break

BoundSearch(25, 50, 50, 90)





