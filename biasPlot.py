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
    k="tasmax_1985_06_01.mat_corrected.mat"
    corrected = scipy.io.loadmat(path_corrected + k)
    correctedLat = corrected[k[:-4]+"_Lat"]
    correctedLong = corrected[k[:-4] + "_Long"]

    path_bias = '/Users/DavidKMYang/ClimateResearch/WBGT/BiasCorrections_nh/Temp/'
    os.chdir(path_bias)
    file_names_bias = glob.glob("*.mat")

    for i in range(len(file_names_bias)):

        k = "biasCorrection_6.mat"
        bias = scipy.io.loadmat(path_bias + k)
        bias = bias[k[:-4]]

        ValForPlot = []
        print (bias)

        flatTLat = np.array(correctedLat)
        flatTLon = np.array(correctedLong)
        flatTData = np.array(bias)


        # print (ValForPlot[i][k])

        # break
        m = Basemap(width=10000000,height=9000000,
                    resolution='l',projection='stere',
                    lat_ts = 40, lat_0=(100)/2, lon_0 = (250))

        lon, lat = np.meshgrid(flatTLon[0,:], flatTLat[:,0])
        x, y = m(lon,lat)
        cs = m.pcolor(x,y,np.squeeze(flatTData), vmin=-20, vmax=20)

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
        plt.title('Mean Temperature in 1980, january')

        plt.show()
        break

BoundSearch()





