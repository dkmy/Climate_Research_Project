__author__ = 'DavidKMYang'
import os
import scipy.io
import csv
import numpy as np
import pandas as pd
from numpy import genfromtxt
import glob

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

path_ccsm4 = '/Users/DavidKMYang/ClimateResearch/WBGT/ccsm4_tasmax_nepal/'


os.chdir(path_ccsm4)
file_names_ccsm4 = glob.glob("tasmax_1980*.mat")

print (file_names_ccsm4)

for i in range(len(file_names_ccsm4)):
    print (file_names_ccsm4[i])
    tempData = scipy.io.loadmat(path_ccsm4 + file_names_ccsm4[i])
    tempData = tempData[file_names_ccsm4[i][:-4]][0]
    # print (tempData[2][0])
    # print (len(tempData[2][0][0]))
    print (len(tempData[0]), " ", len(tempData[0][0]))

    print (len(tempData[1]), " ", len(tempData[1][0]))


    print (len(tempData[2]), " ", len(tempData[2][4]), " ", len(tempData[2][4][0]))

    print (tempData[2][5][1])

    print (tempData[0][0])
    print (tempData[1][3])
    break

