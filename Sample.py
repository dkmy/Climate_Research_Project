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

    path_gfdl_cm3 = '/Users/DavidKMYang/ClimateResearch/WBGT/BiasCorrections/Temp/'
    os.chdir(path_gfdl_cm3)

    file_names_gfdl_cm3 = glob.glob("biasCorrection_" + str(month) + ".mat")



    tot_Diff_List = [] #contains all the values in all the grid over the course of X years of a particular month

    for i in range(len(file_names_gfdl_cm3)):
        print (i)

        tempData_gfdl = scipy.io.loadmat(path_gfdl_cm3 + file_names_gfdl_cm3[i])
        print (len(tempData_gfdl[file_names_gfdl_cm3[i][:-4]][0]))
        # tempData_gfdl = tempData_gfdl[file_names_gfdl_cm3[i][:-4]][0]
        # print (len(tempData_gfdl))



BoundSearch(1)