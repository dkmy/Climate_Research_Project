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

import datetime
import dateutil.parser

def Julian(y, m, d):
    dt = datetime.datetime(y, m, d)
    tt = dt.timetuple()
    return tt.tm_yday

def direct(lat, long, Julday):
    I0 = 1368
    x = 0.9856 * Julday - 2.72
    alpha = lat
    sinb = 0.3978+math.sin(x-77.51+1.92*math.sin(x))
    tlt = -7.66*math.sin(x)-9.87* math.sin(2*x+24.99+3.83*math.sin(x)) + 14.0
    gamma = (tlt-12*30.0) * 15/30.0

    sinh = math.sin(alpha) * sinb + math.cos(alpha) * math.sqrt(1-math.pow(sinb, 2)) * math.cos(gamma)
    # tlinke =math.log(G/(I0 * sinh * 0.84)) * sinh/(-0.027)
    tlinke = 4.25
    I = I0 * math.exp((-tlinke * 1000/1023.25)/(.9+.94*sinh))
    I = I * sinh

    return [I, sinh]


def diffuse (J, B, sinh):

    C_j = 1+0.11 * math.cos((J - 15) * 2/365.0)
    D_r = 39.78* sinh^0.35
    D_b = 2.6* sinh^0.66 * (10^3 * B - 12)^0.81
    # KN_LM = 0.89 + 0.11 * 10^(0.17*NLM), (low cloud) NLM = cloud fraction = 0
    KN_LM = 0.89 + 0.11 * 10^(0.17*0)
    # KN_H = 1+0.035*NH, (high cloud) NH = cirrus fraction = 0
    KN_H = 1+0.035*0

    D=C_j*(D_r+D_b)(KN_LM+KN_H - 1)

    return D

print (direct(42, 0, Julian(2010, 12, 5)))
