__author__ = 'DavidKMYang'

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
import math

def direct(lat, long, day):
    I0 = 1368
    x = 0.9856 * day - 2.72
    alpha = lat
    sinb = 0.3978+math.sin(x-77.51+1.92*math.sin(x))
    tlt = -7.66*math.sin(x)-9.87* math.sin(2*x+24.99+3.83*math.sin(x)) + 14
    gamma = (tlt-12*30) * 15/30

    sinh = math.sin(alpha) * sinb + math.cos(alpha) * (1-sinb^2)^.5 * math.cos(gamma)

    tlinke =math.log(G/(I0 * sinh * 0.84)) * sinh/(-0.027)

    I = I0 * math.exp((-tlinke * 1000/1023.25)/(.9+.94*sinh))




