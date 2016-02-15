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


def hist_plot(header, yearStart, yearEnd, monthBool, monthStart, monthEnd, customBool):

    if header == "tmax":
        path_wbgt = '/Users/DavidKMYang/ClimateResearch/WBGT/ncep_tasmax_nh/'
    if customBool == 0:
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

hist_plot("tasmax", 1995, 2004, 0, 0, 0, 0)
