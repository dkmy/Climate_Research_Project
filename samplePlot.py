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

def BoundSearch():
    path_wbgt = '/Users/DavidKMYang/ClimateResearch/WBGT/gfdl_wbgt/'
    os.chdir(path_wbgt)
    file_names_wbgt = glob.glob("*.mat")


    for i in range(len(file_names_wbgt)):
        print (i)
        tempData_wbgt = scipy.io.loadmat(path_wbgt + file_names_wbgt[i])
        tempData_wbgt = scipy.io.loadmat(path_wbgt + 'wbgt_1981_07_01.mat')
        print (tempData_wbgt)
        break

        tempData_Lat_wbgt = tempData_wbgt[file_names_wbgt[i][:-4]+"_Lat"]
        tempData_Long_wbgt = tempData_wbgt[file_names_wbgt[i][:-4] + "_Long"]
        tempData_Val_wbgt = tempData_wbgt[file_names_wbgt[i][:-4] + "_Val"]

        ValForPlot = []



        for i in range(len(tempData_Val_wbgt)):
            tempList = []
            for k in range(len(tempData_Val_wbgt[0])):
                tempList.append(np.mean(tempData_Val_wbgt[i][k]))
            ValForPlot.append(tempList)

        print (len(ValForPlot[0]))

        flatTLat = np.array(tempData_Lat_wbgt)
        flatTLon = np.array(tempData_Long_wbgt)
        flatTData = np.array(ValForPlot)


        # print (ValForPlot[i][k])

        # break
        m = Basemap(width=10000000,height=9000000,
                    resolution='l',projection='stere',
                    lat_ts = 40, lat_0=(10000)/2, lon_0 = (10000)/2)

        lon, lat = np.meshgrid(flatTLon[0,:], flatTLat[:,0])
        x, y = m(lon,lat)
        cs = m.pcolor(x,y,np.squeeze(flatTData), vmin=0, vmax=30)

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
        plt.title('Mean Temperature in 1981, july')

        plt.show()
        break

BoundSearch()





