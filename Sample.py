__author__ = 'DavidKMYang'

#given a grid and time, average results of multiple models
import h5py
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

def BoundSearch(month):

    path_gfdl_cm3 = '/Users/DavidKMYang/ClimateResearch/WBGT/gfdl_tasmax_nh/'
    os.chdir(path_gfdl_cm3)

    if month < 10:
        file_names_gfdl_cm3 = glob.glob("*_0" + str(month) + "*.mat")
    else:
        file_names_gfdl_cm3 = glob.glob("*_" + str(month) + "*.mat")

    path_ncep = '/Users/DavidKMYang/ClimateResearch/WBGT/ncep_tasmax_usne/'
    os.chdir(path_ncep)

    if month < 10:
        file_names_ncep = glob.glob("*_0" + str(month) + "*.mat")
    else:
        file_names_ncep = glob.glob("*_" + str(month) + "*.mat")

    tot_Diff_List = [] #contains all the values in all the grid over the course of X years of a particular month

    for i in range(len(file_names_ncep)):
        print (i)

        tempData_gfdl = scipy.io.loadmat(path_gfdl_cm3 + file_names_gfdl_cm3[i])
        tempData_gfdl = tempData_gfdl[file_names_gfdl_cm3[i][:-4]][0]

        tempData_ncep = scipy.io.loadmat(path_ncep + file_names_ncep[i])
        tempData_ncep = tempData_ncep[file_names_ncep[i][:-4]][0]
        print (tempData_ncep[0][0])
        break




for i in range(12):
    BoundSearch(i+1)