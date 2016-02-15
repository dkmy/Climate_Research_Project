__author__ = 'DavidKMYang'

from scipy.io import netcdf
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


f = netcdf.netcdf_file('/Users/DavidKMYang/Downloads/air.2m.gauss.1979.nc', 'r')

print(f.history)

air_Data = f.variables['air']
long_Data = f.variables['lon']
lat_Data = f.variables['lat']
time_Data = f.variables['time']


# print (air_Data.shape)
print (lat_Data)
# print (air_Data[0][0][0])



del air_Data
del long_Data
del lat_Data
del time_Data

f.close()
# print (f.shape)