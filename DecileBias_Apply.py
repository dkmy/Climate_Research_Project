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
from compiler.ast import flatten

def grid_apply():
    path_wbgt = '/Users/DavidKMYang/ClimateResearch/WBGT/ncep_tasmax_nh/'
    path_wbgt_model = '/Users/DavidKMYang/ClimateResearch/WBGT/gfdl_tasmax_nh/'
    os.chdir(path_wbgt)

    file_names_wbgt = glob.glob("*.mat")

    os.chdir(path_wbgt_model)

    file_names_wbgt_model = (glob.glob("*.mat"))[:len(file_names_wbgt)]

    totalDays = 0
    actualTotalDays = 0
    Val2D = []
    Val2D_model = []
    Val2D = np.array(Val2D)
    Val2D_model = np.array(Val2D_model)

    for n in range(len(file_names_wbgt_model)):

        tempData_wbgt = scipy.io.loadmat(path_wbgt + file_names_wbgt[n])
        tempData_wbgt = tempData_wbgt[file_names_wbgt[n][:-4]][0]
        tempData_Val_wbgt = tempData_wbgt[2]

        tempData_wbgt_model = scipy.io.loadmat(path_wbgt_model + file_names_wbgt_model[n])
        tempData_wbgt_model = tempData_wbgt_model[file_names_wbgt_model[n][:-4]][0]
        tempData_Val_wbgt_model = tempData_wbgt_model[2]

        for i in range(len(deciles)):




    np.savetxt("/Users/DavidKMYang/ClimateResearch/WBGT/BiasCorrectionsDeciles/Deciles_" + str(lat) + "_" + str(long) + ".csv", final, delimiter = ",", fmt="%s", comments='')


# print hist_plot(14, 14)

path_wbgt = '/Users/DavidKMYang/ClimateResearch/WBGT/ncep_tasmax_nh/'
os.chdir(path_wbgt)
file_names_wbgt = glob.glob("*.mat")
tempData_wbgt = scipy.io.loadmat(path_wbgt + file_names_wbgt[0])
tempData_wbgt = tempData_wbgt[file_names_wbgt[0][:-4]][0]
# tempData_Val_wbgt = tempData_wbgt[2]

# print len(tempData_wbgt[0][0])

for i in range(18):
    for j in range(180):
        print i
        decile_find(i, j)
