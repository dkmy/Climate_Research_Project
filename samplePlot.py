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
    path_corrected = '/Users/DavidKMYang/ClimateResearch/WBGT/corrected_gfdl_tasmax_nh/'

    os.chdir(path_corrected)
    file_names_corrected = glob.glob("*.mat")

    for i in range(len(file_names_corrected)):

        corrected = scipy.io.loadmat(path_corrected + file_names_corrected[i])
        correctedLat = corrected[file_names_corrected[i][:-4]+"_Lat"]
        correctedLong = corrected[file_names_corrected[i][:-4] + "_Long"]
        correctedVal = corrected[file_names_corrected[i][:-4] + "_Val"]


        ValForPlot = correctedVal

        for i in range(len(correctedVal)):
            for k in range(len(correctedVal)):
                print ValForPlot[i][k]

                ValForPlot[i][k] = np.mean(correctedVal[i][k])

        flatTLat = np.array(correctedLat)
        flatTLon = np.array(correctedLong)
        flatTData = np.array(ValForPlot)

        print (ValForPlot[i][k])

        break
        m = Basemap(width=10000000/2,height=7000000/2,
                    resolution='l',projection='stere',
                    lat_ts = 40, lat_0=(100)/2, lon_0 = (100)/2)

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

BoundSearch()





