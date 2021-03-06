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
from mpl_toolkits.basemap import shiftgrid



def hist_plot(header, yearStart, yearEnd, monthBool, monthStart, monthEnd, customBool):

    if header == "tmax":
        path_wbgt = '/Users/DavidKMYang/ClimateResearch/WBGT/ncep_tasmax_nh/' #ncep is historical
    elif customBool == 0:
        path_wbgt = '/Users/DavidKMYang/ClimateResearch/WBGT/gfdl_tasmax_nh/' #gfdl is model
    else:
        path_wbgt = '/Users/DavidKMYang/ClimateResearch/WBGT/BiasCorrected_v2/BiasCorrectedVals/'


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

    for n in range(len(file_names_wbgt)):
        if (customBool == 0):
            tempData_wbgt = scipy.io.loadmat(path_wbgt + file_names_wbgt[n])
            tempData_wbgt = tempData_wbgt[file_names_wbgt[n][:-4]][0]
            tempData_Val_wbgt = tempData_wbgt[2]
        else:
            tempData_wbgt = scipy.io.loadmat(path_wbgt + file_names_wbgt[n])

            # tempData_Lat_temp = tempData_wbgt[file_names_wbgt[i][:-4]+"_Lat"]
            # tempData_Long_temp = tempData_wbgt[file_names_wbgt[i][:-4] + "_Long"]
            tempData_Val_wbgt = tempData_wbgt[file_names_wbgt[n][:-4] + "_Val"]

        # print (len(tempData_Val_wbgt))
        # break
        actualTotalDays += len(tempData_Val_wbgt[0][0])

        if n == 0:
            for i in range(len(tempData_Val_wbgt)):
                tempLatList = []
                for k in range(len(tempData_Val_wbgt[0])):
                    tempLong = 0
                    for j in range(len(tempData_Val_wbgt[0][0])):
                        tempLong += (tempData_Val_wbgt[i][k][j])
                        totalDays += 1
                    tempLatList.append(tempLong)
                Val2D.append(tempLatList)
            break

        for i in range(len(tempData_Val_wbgt)):
            for k in range(len(tempData_Val_wbgt[0])):
                tempLong = 0
                for j in range(len(tempData_Val_wbgt[0][0])):
                    tempLong += (tempData_Val_wbgt[i][k][j])
                    totalDays += 1
                Val2D[i][k] += tempLong

    for i in range(len(Val2D)):
        for k in range(len(Val2D[0])):
            Val2D[i][k] /= actualTotalDays

    Val2D = np.array(Val2D)

    flat_product = Val2D.flatten()

    numpy_hist = plt.figure()

    # plt.hist(flat_product, bins)
    n, bins, patches = plt.hist(flat_product, 20, normed = 1, facecolor='green', alpha=0.5)

    # plot_url = py.plot_mpl(numpy_hist, filename='docs/histogram-mpl-legend')
    plt.grid(True)
    plt.show()

    return Val2D

def Plot(array, correct_Bool):
    path_wbgt = '/Users/DavidKMYang/ClimateResearch/WBGT/gfdl_tasmax_nh/'
    path_wbgt_corrected = '/Users/DavidKMYang/ClimateResearch/WBGT/BiasCorrected_v2/BiasCorrectedVals/'

    tempData_wbgt = scipy.io.loadmat(path_wbgt + 'tasmax_2005_07_01.mat')
    tempData_wbgt = tempData_wbgt['tasmax_2005_07_01'][0]

    tempData_wbgt_corrected = scipy.io.loadmat(path_wbgt_corrected + 'tasmax_2005_07_01.mat_corrected.mat')

    flatTLat = np.array(tempData_wbgt[0])
    flatTLon = np.array(tempData_wbgt[1])
    flatTData = np.array(array)

    if correct_Bool:
        flatTLat = np.array(tempData_wbgt_corrected['tasmax_2005_07_01.mat_corrected_Lat'])
        flatTLon = np.array(tempData_wbgt_corrected['tasmax_2005_07_01.mat_corrected_Long'])
        # flatTData = np.array(tempData_wbgt_corrected['tasmax_2005_07_01.mat_corrected_Val'])


    for i in range(len(flatTLon)):
        for j in range(len(flatTLon[0])):
            flatTLon[i][j] -= 180



    # m = Basemap(width=10000000,height=10000000,
    #             resolution='l',projection='kav7',
    #             lat_ts = 10, lat_0=30, lon_0 = 30)
    m = Basemap(projection = 'kav7', lon_0 = 0, resolution = 'l')


    lon, lat = np.meshgrid(flatTLon[0,:], flatTLat[:,0])
    x, y = m(lon,lat)
    cs = m.pcolor(x,y,np.squeeze(flatTData), vmin=-15, vmax=15)
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

# model_Data = hist_plot("tasmax", 1994, 2005, 0, 0, 0, 0)
# historical_Data = hist_plot("tmax", 1994, 2005, 0, 0, 0, 0)
corrected_Data = hist_plot("tasmax", 1994, 2005, 0, 0, 0, 1)
#
# # print (historical_Data)
# differences = []
# for i in range(len(model_Data)):
#     tempArray = []
#     for k in range(len(model_Data[0])):
#         tempArray.append(corrected_Data[i][k] - historical_Data[i][k]) #subtract model from historical , model - historical
#     differences.append(tempArray)
#
# print (len(differences[0]))
# Plot(differences, False)

differences = []

path_wbgt = '/Users/DavidKMYang/ClimateResearch/WBGT/gfdl_tasmax_nh/'
path_wbgt_corrected = '/Users/DavidKMYang/ClimateResearch/WBGT/BiasCorrected_v2/BiasCorrectedVals/'

tempData_wbgt = scipy.io.loadmat(path_wbgt + 'tasmax_2005_06_01.mat')
tempData_wbgt = tempData_wbgt['tasmax_2005_06_01'][0]

tempData_wbgt_corrected = scipy.io.loadmat(path_wbgt_corrected + 'tasmax_2005_06_01.mat_corrected.mat')
# tempData_wbgt_corrected = tempData_wbgt_corrected['tasmax_2005_06_01.mat_corrected'][0]
flatTLat = np.array(tempData_wbgt_corrected['tasmax_2005_06_01.mat_corrected_Lat'])
flatTLon = np.array(tempData_wbgt_corrected['tasmax_2005_06_01.mat_corrected_Long'])
flatTData = np.array(tempData_wbgt_corrected['tasmax_2005_06_01.mat_corrected_Val'])

for i in range(len(tempData_wbgt[0])):
    tempArray = []
    for k in range(len(tempData_wbgt[0][0])):
        tempArray.append(np.mean(tempData_wbgt[2][i][k]) - np.mean(flatTData[i][k]))
    differences.append(tempArray)

# print (tempData_wbgt[2][3][4])

# print (len(tempArray))

# print (len(tempData_wbgt))

# Plot(differences, False)


