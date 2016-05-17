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
import h5py
import hdf5storage as hdf

def find_nearest(arr,v):
    return (np.abs(arr - v)).argmin()

def BoundSearch():
    # path_wbgt = '/Users/DavidKMYang/ClimateResearch/WBGT/gfdl_tasmax_nh/'
    path_wbgt = '/Users/DavidKMYang/ClimateResearch/Middle_East_Data/gfdl-cm3-tasmax-historical-world-nbc-v7/20050101-20051231/'
    os.chdir(path_wbgt)
    file_names_wbgt = glob.glob("*.mat")

    for i in range(len(file_names_wbgt)):

        # tempData_wbgt = scipy.io.loadmat(path_wbgt + file_names_wbgt[i])
        tempData_wbgt = scipy.io.loadmat(path_wbgt + 'tasmax_2005_07_01.mat')
        tempData_wbgt = tempData_wbgt['tasmax_2005_07_01'][0]


        tempData_Val_wbgt = tempData_wbgt[2]

        ValForPlot = []


        tempData_Lat_wbgt = tempData_wbgt[0]

        for i in range(len(tempData_Val_wbgt)):
            tempList = []
            for k in range(len(tempData_Val_wbgt[0])):
                tempList.append(np.mean(tempData_Val_wbgt[i][k]))
            ValForPlot.append(tempList)

        print (len(ValForPlot[0]))
        flatTLat = np.array(tempData_wbgt[0])
        flatTLon = np.array(tempData_wbgt[1])
        flatTData = np.array(ValForPlot)

        # m = Basemap(width=50000000,height=50000000,
        #             resolution='l',projection='stere',
        #             lat_ts = 3, lat_0=28, lon_0 = 52.6)
        m = Basemap(projection = 'kav7', lon_0 = 0, resolution = 'l')

        lon, lat = np.meshgrid(flatTLon[0,:], flatTLat[:,0])
        x, y = m(lon,lat)
        cs = m.pcolor(x,y,np.squeeze(flatTData), vmin=-40, vmax=+40)

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
        break

BoundSearch()





