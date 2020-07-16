# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 15:13:24 2018

@author: johndaniels
"""

# import modules
from __future__ import print_function
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from dateutil import parser
import csv
from bisect import bisect_left
import math
from scipy import interpolate
from scipy.signal import medfilt
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage.filters import gaussian_filter
from patient_data import load_from_xml
import scipy.io as sio

import time
import datetime


from time import mktime
from datetime import date


#################
out_dim = 1
n_prev  = 24


# ---------------------------------------------------------
# NORMALIZING DATA FROM ABC4D
# --

def normalize(data):
    temp = np.float32(data) - np.min(data)
    out = (temp / np.max(temp) - 0.5) * 2
    return out

def standardize(data, min, max):
    X_min = min
    X_max = max
    X_std = (data - X_min)/(X_max - X_min)
    return X_std

def transform(data, min, max):
    X_min = min
    X_max = max
    X_scaled = data * (X_max - X_min) + X_min
    return X_scaled



# ---------------------------------------------------------
# RETRIEVE PRE-PROCESSED DATA
# --

def parseData(data_src, file_num, xls_file_name, train):

    argFP       = []
    columnHeads = []


    if(train == 1):
        print("Ohio T1DM Training Dataset")
        ids = [xls_file_name]
        file_train_path = ['C:/Users/kaise/Projects/ARISES/BGLP2/OhioT1DM/Training/raw/' + id + '-ws-training.xml' for id in ids]
        dataRaw, categories, timeStamps, isRealBGL = load_from_xml(zip(ids, file_train_path), res=5, verbose=False, train = train)
        #print("")
    else:
        print("Ohio T1DM Testing Dataset")
        ids = [xls_file_name]
        file_test_path = ['C:/Users/kaise/Projects/ARISES/BGLP2/OhioT1DM/Testing/raw/' + id + '-ws-testing.xml' for id in ids]
        dataRaw, categories, timeStamps, isRealBGL = load_from_xml(zip(ids, file_test_path), res=5, verbose=False, train = train)

    # src_data, argFP = parseOhioT1DM(data_struct_p, train)
    src_data 	= np.array(dataRaw[xls_file_name])
    argFP = np.array(isRealBGL[xls_file_name])
    columnHeads = np.array(categories[xls_file_name])

    print('Ohio-T1DM - Clinical data')

    return src_data, argFP, columnHeads, timeStamps




def _load_data(dataFrame, n_prev = 24, pred_horizon = 6, out_dim = 1, train = 1, mode = 'basic'):
    """
    data should be pd.DataFrame()
    """
    dataFrame['G'] = dataFrame['G']/120
    dataFrame['I'] = dataFrame['I']/100
    dataFrame['M'] = dataFrame['M']/200
    m = np.array(dataFrame['exercise'].values.tolist())
    dataFrame['exercise'] = np.where(m > 0, 1, m).tolist()

    if(mode == 'basic'):
        data = dataFrame[['G','I','M','exercise']]
    else:
        data = dataFrame[['G','I','M','AT','ST','GSR','HR','STP','exercise']]#

    time_shift = pred_horizon #30-min window(6)/60-min window(12)

    docX, docY, docZ, docTrue, docAux = [], [], [], [], []
    for i in range(len(data) - n_prev - (time_shift) + 1):
        docX.append(data.iloc[i:(i + n_prev)].as_matrix())
        docY.append(data.iloc[i + n_prev + (pred_horizon) - 1]['G'] - data.iloc[i + n_prev - 1]['G']) # here 6 represents 30 minutes

        docZ.append(data.iloc[i + n_prev + (pred_horizon) - 1]['G'])
        docTrue.append(data.iloc[i + n_prev + (pred_horizon) - 1]['G'])

    alsX    = np.array(docX)
    alsY    = np.array(docY)
    alsZ    = np.array(docTrue)
    return alsX, alsY, alsZ



def data_prep_split(df, pred_window, mode):
    """
    This just splits data to training and testing parts
    """
    training = 1
    X_train, y_train, train_ref = _load_data(df, n_prev, pred_window, out_dim, training, mode )

    return (X_train, y_train, train_ref)
################
