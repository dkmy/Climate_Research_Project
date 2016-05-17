__author__ = 'DavidKMYang'
#https://plot.ly/python/getting-started

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import scipy
import os
import scipy.io
import numpy as np
import plotly.plotly as py
import glob
import pandas as pd
from numpy import genfromtxt


def hist_plot(header, yearStart, yearEnd, monthBool, monthStart, monthEnd, customBool, stdevBool):

    if header == "tasmax":
        path_wbgt = '/Users/DavidKMYang/ClimateResearch/Middle_East_Data/gfdl-cm3-tasmax-historical-world-nbc-v7/combine/'
    elif customBool == 0:
        path_wbgt = '/Users/DavidKMYang/ClimateResearch/WBGT/gfdl_tasmax_nh/'
    else:
        path_wbgt = '/Users/DavidKMYang/ClimateResearch/WBGT/corrected_gfdl_tasmax_nh/'


    os.chdir(path_wbgt)

    file_names_wbgt = []

    for n in range(yearEnd - yearStart + 1):
        if (int(monthBool) == 0):
            file_names_wbgt = glob.glob(header + "_" + str(yearStart + n) + "*.mat")

        else:
            for i in range(monthEnd - monthStart + 1):
                if monthStart + i < 10:
                    file_names_temp = glob.glob(header + "_" + str(yearStart + n) + "_" + "0" + str(monthStart + i) + "*.mat")
                else:
                    file_names_temp = glob.glob(header + "_" + str(yearStart + n) + "_" + str(monthStart + 1) + "*.mat")
                file_names_wbgt.extend(file_names_temp)

    totalDays = 0
    actualTotalDays = 0
    Val2D = []
    Val3D_Stdev = []


    for n in range(len(file_names_wbgt)):
        if (customBool == 0):
            tempData_wbgt = scipy.io.loadmat(path_wbgt + file_names_wbgt[n])
            tempData_wbgt = tempData_wbgt[file_names_wbgt[n][:-4]][0]
            tempData_Val_wbgt = tempData_wbgt[2]
        else:
            tempData_wbgt = scipy.io.loadmat(path_wbgt + file_names_wbgt[n])

            tempData_Lat_temp = tempData_wbgt[file_names_wbgt[i][:-4]+"_Lat"]
            tempData_Long_temp = tempData_wbgt[file_names_wbgt[i][:-4] + "_Long"]
            tempData_Val_wbgt = tempData_wbgt[file_names_wbgt[n][:-4] + "_Val"]

        # print (len(tempData_Val_wbgt))
        # break
        actualTotalDays += len(tempData_Val_wbgt[0][0])

        if n == 0:
            for i in range(len(tempData_Val_wbgt)):
                tempLatList = []
                Val2D_stdev = []
                for k in range(len(tempData_Val_wbgt[0])):
                    # Val1D_stdev = []
                    tempLong = 0
                    for j in range(len(tempData_Val_wbgt[0][0])):
                        tempLong += (tempData_Val_wbgt[i][k][j])
                        # Val1D_stdev.append(tempData_Val_wbgt[i][k][j])
                        totalDays += 1
                    tempLatList.append(tempLong)
                    Val2D_stdev.append(tempData_Val_wbgt[i][k])
                Val2D.append(tempLatList)
                Val3D_Stdev.append(Val2D_stdev)

            break

        for i in range(len(tempData_Val_wbgt)):
            for k in range(len(tempData_Val_wbgt[0])):
                tempLong = 0
                for j in range(len(tempData_Val_wbgt[0][0])):
                    tempLong += (tempData_Val_wbgt[i][k][j])
                    Val3D_Stdev[i][k][j].append(tempData_Val_wbgt[i][k][j])
                    totalDays += 1
                Val2D[i][k] += tempLong

    for i in range(len(Val2D)):
        for k in range(len(Val2D[0])):
            Val2D[i][k] /= actualTotalDays

    Val2D = np.array(Val2D)
    if stdevBool:
        # print (len(Val3D_Stdev[0]))
        return Val3D_Stdev
    else:
        return Val2D

def Plot(array):
    path_wbgt = '/Users/DavidKMYang/ClimateResearch/Middle_East_Data/gfdl-cm3-tasmax-historical-world-nbc-v7/combine/'

    tempData_wbgt = scipy.io.loadmat(path_wbgt + 'tasmax_2005_07_01.mat')
    tempData_wbgt = tempData_wbgt['tasmax_2005_07_01'][0]

    flatTLat = np.array(tempData_wbgt[0])
    flatTLon = np.array(tempData_wbgt[1])
    flatTData = np.array(array)

    # m = Basemap(width=10000000,height=10000000,
    #             resolution='l',projection='kav7',
    #             lat_ts = 10, lat_0=30, lon_0 = 30)
    m = Basemap(projection = 'robin', lon_0 = 0, resolution = 'l')
    # m = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,\
    #         llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='l')
    # m = Basemap(projection='hammer',lon_0=0,resolution='l')
    # m = Basemap(width=100000000/4,height=70000000/4, resolution='l',projection='stere', lat_ts = 40, lat_0=50, lon_0 = 100)
    lon, lat = np.meshgrid(flatTLon[0,:], flatTLat[:,0])
    x, y = m(lon,lat)
    cs = m.pcolor(x,y,np.squeeze(flatTData), vmin=-6, vmax=6)
    m.drawmapboundary(fill_color='0.3')
    # Add Grid Lines
    m.drawparallels(np.arange(-80., 81., 10.), labels=[1,0,0,0], fontsize=10)
    m.drawmeridians(np.arange(0., 360., 10.), labels=[0,0,0,1], fontsize=10)

    # Add Coastlines, States, and Country Boundaries
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()

    # Add Colorbar
    cbar = m.colorbar(cs, location='bottom', pad="10%")

    # Add Title
    plt.title('n')

    plt.show()

# model_Data = hist_plot("tasmax", 1980, 2005, 1, 1, 2, 0)
# model_Data = hist_plot("tasmax", 1994, 2005, 0, 0, 0, 0)

stdev_Data = hist_plot("tasmax", 1980, 1980, 1, 1, 12, 0, 1)
# print (len(stdev_Data))
# print (len(stdev_Data[0]))
# print (len(stdev_Data[0][0]))
# print (len(stdev_Data[0][0][0]))

final_stdev = []

for i in range(len(stdev_Data)):
    temp_Arr = []
    for k in range(len(stdev_Data[0])):
        low_temp = np.array(stdev_Data[i][k])
        temp_Arr.append(np.std(low_temp, ddof = 1))
    final_stdev.append(temp_Arr)

Plot(final_stdev)


# Plot(model_Data)
# print (historical_Data)
# differences = []
# for i in range(len(model_Data)):
#     tempArray = []
#     for k in range(len(model_Data[0])):
#         tempArray.append(model_Data[i][k] - historical_Data[i][k]) #subtract model from historical , model - historical
#     differences.append(tempArray)
#
# print (len(differences[0]))
# Plot(differences)




