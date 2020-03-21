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
    X_min = min #-85
    X_max = max #85
    X_std = (data - X_min)/(X_max - X_min)
    return X_std

def transform(data, min, max):
    X_min = min
    X_max = max
    X_scaled = data * (X_max - X_min) + X_min
    return X_scaled


# ---------------------------------------------------------
# TAKE THE CLOSEST NUMBER FOR INTERPOLATION
# --

def takeClosest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0], 0
    if pos == len(myList):
        return myList[-1], len(myList)
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
       return after, pos
    else:
       return before, pos



def to_adverse_event(norm_reference, num_classes):
    sHypo_x   = 54/120  # 0.039 # clinically relevant hypoglycaemia
    Hypo_x    = 70/120  # 0.083 # typically suggested level for hypoglycaemia
    Hyper_x   = 180/120 # 0.389 # typically suggested level for hyperglycaemia
    sHyper_x  = 240/120 # 0.556 # hyperglycaemia level that may result in ketones in blood

    lb = -3  #lower bound for time of event classification (~15 mins)
    ub = +4  #upper bound for time of event classification (~15 mins)

    reference = norm_reference
    event_ref = np.empty(len(reference))
    event_ref.fill(2)

    for index in range(len(reference)):
        if index < abs(lb):

            if(reference[1] <= sHypo_x ):
                event_ref[index] =  0
            if(sHypo_x < reference[1] <= Hypo_x):
                event_ref[index] =  1
            if(Hypo_x < reference[1]  < Hyper_x):
                event_ref[index] =  2
            if(Hyper_x <= reference[1] < sHyper_x):
                event_ref[index] =  3
            if(reference[1] >= sHyper_x):
                event_ref[index] =  4
        elif index > (len(reference) - abs(ub)):
            if(reference[1] <= sHypo_x ):
                event_ref[index] =  0
            if(sHypo_x < reference[1] <= Hypo_x ):
                event_ref[index] =  1
            if(Hypo_x < reference[1]  < Hyper_x):
                event_ref[index] =  2
            if(Hyper_x <= reference[1] < sHyper_x):
                event_ref[index] =  3
            if(reference[1] >= sHyper_x):
                event_ref[index] =  4
        else:
            sHypo_num  = 0
            Hypo_num   = 0
            Norm_num   = 0
            Hyper_num  = 0
            sHyper_num = 0

            for k in range(lb,ub):
                if(reference[index + k] <= sHypo_x ):
                    sHypo_num  = sHypo_num  + 1
                if(sHypo_x < reference[index + k] <= Hypo_x ):
                    Hypo_num   =  Hypo_num  + 1
                if(Hypo_x < reference[index + k]  < Hyper_x):
                    Norm_num   =  Norm_num  + 1
                if(Hyper_x <= reference[index + k] < sHyper_x):
                    Hyper_num  =  Hyper_num + 1
                if(reference[index + k] >= sHyper_x):
                    sHyper_num = sHyper_num + 1

            if(sHypo_num >= ub):
                event_ref[index] =  0
            elif(sHyper_num >= ub):
                event_ref[index] =  4
            elif(Hyper_num + sHyper_num >= ub):
                event_ref[index] =  3
            elif(Hypo_num + sHypo_num >= ub):
                event_ref[index] =  1
            elif(Norm_num >= ub):
                event_ref[index] =  2


        #print(event_ref[index])

    if(num_classes == 3):
        np.place(event_ref, event_ref<=1, [0])
        np.place(event_ref, event_ref==2, [1])
        np.place(event_ref, event_ref>=3, [2])
        # event_ref[event_ref <= 1] = 0
        # event_ref[event_ref == 2] = 1
        # event_ref[event_ref >= 3] = 2
        # plt.plot(event_ref)
        # plt.show()

    return event_ref

def to_adverse_hyper_event(reference, num_classes):

    adverse_events = to_adverse_event(reference, 5)

    if(num_classes == 2):
        np.place(adverse_events, adverse_events <= 2, [0])
        np.place(adverse_events, adverse_events >  2, [1])
        hyper_event = adverse_events
    else:
        np.place(adverse_events, adverse_events <= 2, [0])
        np.place(adverse_events, adverse_events == 3, [1])
        np.place(adverse_events, adverse_events == 4, [2])
        hyper_event = adverse_events

    return hyper_event

def to_adverse_hypo_event(reference, num_classes):

    adverse_events = to_adverse_event(reference, 5)

    if(num_classes == 2):
        np.place(adverse_events, adverse_events >= 2, [0])
        np.place(adverse_events, adverse_events <  2, [1])
        hypo_event = adverse_events
    else:
        hypo_event = adverse_events
        np.place(adverse_events, adverse_events >= 2, [2])
        np.place(adverse_events, adverse_events == 1, [1])
        np.place(adverse_events, adverse_events == 0, [0])
        hypo_event = adverse_events

    return hypo_event

# ---------------------------------------------------------
# DATA PREPARATION FOR THE CLINICAL DATA
# --

def parseABC4DClinicalData(root, xls_file_name):

    xls_no = 0
    csv_no = 0
    mol_thre = 3.9

    for path, subdirs, files in os.walk(root):
        for name in files:
            if name.startswith(xls_file_name) and name.endswith(".xls"):
                #print(os.path.join(path, name))
                xls_no = xls_no + 1

                '''
                read data from CGM and normalize time stamp
                '''
                # Import the excel file and call it xls_file
                xls_file = pd.ExcelFile(os.path.join(path, name))
                # Load the xls file's Sheet1 as a dataframe
                df = xls_file.parse('CGM')

                date_time = list(df['Date'][1:])
                mol = np.float64(df.iloc[1:,1]) # np.float64(df['26/11/2015 to 12/05/2016'][1:])
                # 17: 18/03/2016 to 02/09/2016;

                len_data = len(date_time)
                day_time = []
                begin_time = parser.parse(date_time[0], dayfirst=True)
                begin_tt = begin_time.timetuple()
                date_begin = date(begin_tt[0],begin_tt[1],begin_tt[2])
                for ii in range(len_data):
                    dt = parser.parse(date_time[ii], dayfirst=True)
                    tt = dt.timetuple()
                    decimal_frac = (tt.tm_hour*60+tt.tm_min)/(24*60)
                    date_tt = date(tt[0],tt[1],tt[2])
                    # calculate the relative date difference
                    integer_frac =  (date_tt - date_begin).days
                    day_time.append(integer_frac+decimal_frac)
                day_time = np.array(day_time)
                '''
                interpolate to fill the missing timestamp
                '''
                diff_day_time = day_time[1:]-day_time[:-1]
                day_time_step = np.median(diff_day_time)
                day_time_step15 = 1.5*np.median(diff_day_time)
                day_time_interpo = np.array([day_time[0]])
                mol_interpo = np.array([mol[0]])
                cur_ind = int(0)
                for ii in range(len(day_time)):
                    #print(ii)
                    while True:
                        if day_time[ii]-day_time_interpo[cur_ind]> day_time_step15:
                            day_time_interpo = np.append(day_time_interpo, [day_time_interpo[cur_ind]+day_time_step])
                            mol_interpo = np.append(mol_interpo, np.nan)
                            cur_ind += 1
                        else:
                            day_time_interpo = np.append(day_time_interpo, [day_time[ii]])
                            mol_interpo = np.append(mol_interpo, [mol[ii]])
                            cur_ind += 1
                            break
                day_time_interpo =  np.delete(day_time_interpo, 0)
                mol_interpol =  np.delete(mol_interpo, 0)

                mod_s = pd.Series(mol_interpol)
                mod_s_array = np.array(mod_s)
                mol_interpo = np.array(mod_s.interpolate(method = 'slinear'))

                day_time_decimal_interpo, day_time_int_interpo = np.modf(day_time_interpo)
                len_data_interpo = len(day_time_interpo)
                #data_2dim = np.vstack((day_time_decimal,mol)).T
                data_2dim = np.vstack((day_time_decimal_interpo,mol_interpo)).T
            if name.startswith(xls_file_name) and name.endswith(".csv"):
                #print(os.path.join(path, name))
                csv_no = csv_no + 1
                '''
                read log data and tidy them up
                '''
                csv_log_full = pd.read_csv(os.path.join(path, name))
                first_row_values = csv_log_full.loc[csv_log_full['Name/ID:'] == "Date"].index[0]
                csv_log = csv_log_full.iloc[first_row_values:,:]
                csv_log = csv_log.reset_index(drop=True)
                csv_log = csv_log.rename(columns=csv_log.iloc[0])
                csv_log = csv_log.iloc[1:,:]
                csv_log = csv_log.reset_index(drop=True)
                '''
                prepare the dataframe for learning
                '''

                datafm = pd.DataFrame(data=data_2dim, columns = ['day_time','mol'])

                datafm['diff_mol']   = pd.Series(np.zeros(len(datafm)), index=datafm.index)
                datafm['Carb(g)']    = pd.Series(np.zeros(len(datafm)), index=datafm.index)
                datafm['Insulin(U)'] = pd.Series(np.zeros(len(datafm)), index=datafm.index)
                datafm['Exercise']   = pd.Series(np.zeros(len(datafm)), index=datafm.index)
                datafm['Alcohol']    = pd.Series(np.zeros(len(datafm)), index=datafm.index)
                datafm['Stress']     = pd.Series(np.zeros(len(datafm)), index=datafm.index)
                datafm['Illness']    = pd.Series(np.zeros(len(datafm)), index=datafm.index)
                datafm['High_Fat']   = pd.Series(np.zeros(len(datafm)), index=datafm.index)

                datafm.ix['1':(len_data_interpo-1),'diff_mol'] = datafm['mol'][1:].values-datafm['mol'][:-1].values

                df_val_index_list = []
                len_csv_log = len(csv_log)
                for ii in range(len_csv_log):
                    #print(ii)
                    """should be around 16. <8 means it is not a proper time stamp"""
                    if len(csv_log['Timestamp'][ii]) < 8:
                        break
                    dt = parser.parse(csv_log['Timestamp'][ii], dayfirst=True)
                    tt = dt.timetuple()
                    decimal_frac = (tt.tm_hour*60+tt.tm_min)/(24*60)
                    date_tt = date(tt[0],tt[1],tt[2])
                    integer_frac =  (date_tt - date_begin).days
                    curr_time = integer_frac+decimal_frac
                    df_val, df_index = takeClosest(day_time_interpo.tolist(), curr_time)
                    df_val_index_list.append(np.array([df_val,df_index]))
                    # sometimes there are duplicated timestamp
                    if ii>0 and abs(df_val_index_list[-1][0]-df_val_index_list[-2][0])<1e-5:
                        datafm['Carb(g)'][df_index] = datafm['Carb(g)'][df_index]+float(csv_log[' Carbohydrates (g)'][ii])
                        datafm['Insulin(U)'][df_index] = datafm['Insulin(U)'][df_index]+float(csv_log['Inuslin (U)'][ii])
                        datafm['Exercise'][df_index] = datafm['Exercise'][df_index]+float(csv_log['Exercise'][ii])
                        datafm['Alcohol'][df_index] = datafm['Alcohol'][df_index]+float(csv_log['Alcohol'][ii])
                    else:
                        datafm['Carb(g)'][df_index] = float(csv_log[' Carbohydrates (g)'][ii])
                        datafm['Insulin(U)'][df_index] = float(csv_log['Inuslin (U)'][ii])
                        datafm['Exercise'][df_index] = float(csv_log['Exercise'][ii])
                        datafm['Alcohol'][df_index] = float(csv_log['Alcohol'][ii])
                    if np.isnan(float(csv_log[' Stress'][ii])):
                        continue
                    else:
                        datafm['Stress'][df_index] = float(csv_log[' Stress'][ii])
                        datafm['Illness'][df_index] = float(csv_log[' Illness'][ii])
                        datafm['High_Fat'][df_index] = float(csv_log[' High Fat'][ii])
                '''
                draw some time seires for show
                '''

                Carb_array     = datafm['Carb(g)'].as_matrix()
                Insulin_array  = datafm['Insulin(U)'].as_matrix()
                Exercise_array = datafm['Exercise'].as_matrix()
                Alcohol_array  = datafm['Alcohol'].as_matrix()
                Stress_array   = datafm['Stress'].as_matrix()
                Illness_array  = datafm['Illness'].as_matrix()
                high_fat_array = datafm['High_Fat'].as_matrix()

    return datafm, mol_interpo

def parseOhioT1DM(data_struct_ds, is_training):
    #read data to find inital time point
    CGM_timestamp   = np.squeeze(data_struct_ds['CGM_TimeStamp'][0,0])
    HR_timestamp    = np.squeeze(data_struct_ds['Basis_HR_Time'][0,0])
    GSR_timestamp   = np.squeeze(data_struct_ds['Basis_GSR_Time'][0,0])
    ST_timestamp    = np.squeeze(data_struct_ds['Basis_ST_Time'][0,0])
    AT_timestamp    = np.squeeze(data_struct_ds['Basis_AT_Time'][0,0])
    STP_timestamp   = np.squeeze(data_struct_ds['Basis_Steps_Time'][0,0])

    Meal_timestamp      =  np.squeeze(data_struct_ds['Meal_Time'][0,0])
    try:
        Exercise_timestamp  =  np.squeeze(data_struct_ds['Exercises_Time'][0,0])
    except:
        Exercise_timestamp = STP_timestamp
    Insulin_timestamp   =  np.squeeze(data_struct_ds['Insulin_Time'][0,0])

    timeInit_array = np.array([CGM_timestamp[0], HR_timestamp[0], GSR_timestamp[0], ST_timestamp[0], AT_timestamp[0], STP_timestamp[0]])
    valid_time     = np.amax(timeInit_array)

    array_len   = sum(index >= valid_time for index in CGM_timestamp)
    end_index   = len(CGM_timestamp)
    start_index = end_index - array_len

    start_time  = CGM_timestamp[start_index]

    reference_time  = []
    aligned_data    = []
    argFP           = []
    for index_0 in range(start_index, end_index):
        tt = CGM_timestamp[index_0] - start_time
        reference_time.append(tt)

    reference_time = np.array(reference_time)

    Basis_HR  = np.squeeze(data_struct_ds['Basis_HR'][0,0])
    Basis_GSR = np.squeeze(data_struct_ds['Basis_GSR'][0,0])
    Basis_ST  = np.squeeze(data_struct_ds['Basis_ST'][0,0])
    Basis_AT  = np.squeeze(data_struct_ds['Basis_AT'][0,0])
    Basis_STP = np.squeeze(data_struct_ds['Basis_Steps'][0,0])

    CGM       = np.squeeze(data_struct_ds['CGM_Measurement'][0,0])
    Meal      = np.squeeze(data_struct_ds['Meal'][0,0])
    try:
        Exercise  = np.squeeze(data_struct_ds['Exercises_Intensity'][0,0])
        Duration  = np.squeeze(data_struct_ds['Exercises_Duration'][0,0])
    except:
        Exercise = np.squeeze(np.zeros((len(Basis_STP),1)))
        Duration = np.squeeze(np.zeros((len(Basis_STP),1)))

    Insulin   = np.squeeze(data_struct_ds['Insulin'][0,0])

    #align data to CGM
    for index_1 in range(len(reference_time)):
        span     = reference_time[index_1]
        span_var = 150
        ref_time = start_time + span

        HR_val    = np.nan
        GSR_val   = np.nan
        ST_val    = np.nan
        AT_val    = np.nan
        Meal_val  = np.nan
        Steps_val = np.nan
        Intensity_val = np.nan
        Duration_val  = np.nan
        Insulin_val   = np.nan

        CGM_val = CGM[start_index + index_1]

        for index_2 in range(len(HR_timestamp)):
            if(ref_time - span_var < HR_timestamp[index_2] < ref_time + span_var):
                HR_val = Basis_HR[index_2]

        for index_2 in range(len(GSR_timestamp)):
            if(ref_time - span_var < GSR_timestamp[index_2] < ref_time + span_var):
                GSR_val = Basis_GSR[index_2]

        for index_2 in range(len(ST_timestamp)):
            if(ref_time - span_var < ST_timestamp[index_2] < ref_time + span_var):
                ST_val = Basis_ST[index_2]

        for index_2 in range(len(AT_timestamp)):
            if(ref_time - span_var < AT_timestamp[index_2] < ref_time + span_var):
                AT_val = Basis_AT[index_2]

        for index_2 in range(len(STP_timestamp)):
            if(ref_time - span_var < STP_timestamp[index_2] < ref_time + span_var):
                Steps_val = Basis_STP[index_2]

        for index_2 in range(len(Meal_timestamp)):
            if(ref_time - span_var < Meal_timestamp[index_2] < ref_time + span_var):
                Meal_val = Meal[index_2]

        for index_2 in range(len(Exercise_timestamp)):
            if(ref_time - span_var < Exercise_timestamp[index_2] < ref_time + span_var):
                Intensity_val = Exercise[index_2]
                Duration_val  = Duration[index_2]

        for index_2 in range(len(Insulin_timestamp)):
            if(ref_time - span_var < Insulin_timestamp[index_2] < ref_time + span_var):
                Insulin_val = Insulin[index_2]

        aligned_data.append([CGM_val, Insulin_val, Meal_val, HR_val, GSR_val, ST_val, AT_val, Steps_val, Intensity_val, Duration_val])

        if(index_1 != (len(reference_time) - 1)):
            MAR_datapoints = ((reference_time[index_1 + 1] - reference_time[index_1])/300) - 1
            MAR_num        = int(MAR_datapoints)

            for i in range(1, MAR_num + 1):
                aligned_data.append([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
                argFP.append(index_1 + i)

    aligned_data = np.array(aligned_data)
    argFP        = np.array(argFP)

    #interpolate to fill the missing timestamp
    mCGM = pd.Series(aligned_data[:,0])
    mHR  = pd.Series(aligned_data[:,3])
    mGSR = pd.Series(aligned_data[:,4])
    mST  = pd.Series(aligned_data[:,5])
    mAT  = pd.Series(aligned_data[:,6])


    if(is_training == 1):
      mCGM_interp  = np.array(mCGM.interpolate(method = 'slinear'))
      mCGM_interpo = medfilt(mCGM_interp, 5)
      mHR_interpo  = np.array(mHR.interpolate(method  = 'slinear'))
      mGSR_interpo = np.array(mGSR.interpolate(method = 'slinear'))
      mST_interpo  = np.array(mST.interpolate(method  = 'slinear'))
      mAT_interpo  = np.array(mAT.interpolate(method  = 'slinear'))
    else:
      mCGM_interpo = np.array(mCGM.fillna(method = 'ffill'))
      mHR_interpo  = np.array(mHR.fillna(method  = 'ffill'))
      mGSR_interpo = np.array(mGSR.fillna(method = 'ffill'))
      mST_interpo  = np.array(mST.fillna(method  = 'ffill'))
      mAT_interpo  = np.array(mAT.fillna(method  = 'ffill'))

    mCGM_interpo.shape = (len(mCGM_interpo),1)
    mHR_interpo.shape  = (len(mHR_interpo) ,1)
    mGSR_interpo.shape = (len(mGSR_interpo),1)
    mST_interpo.shape  = (len(mST_interpo) ,1)
    mAT_interpo.shape  = (len(mAT_interpo) ,1)

    mInsulin = pd.Series(aligned_data[:,1])
    mInsulin_interpo = np.array(mInsulin.fillna(0))
    mInsulin_interpo.shape = (len(mInsulin_interpo),1)
    mMeal = pd.Series(aligned_data[:,2])
    mMeal_interpo = np.array(mMeal.fillna(0))
    mMeal_interpo.shape = (len(mMeal_interpo),1)
    mSteps = pd.Series(aligned_data[:,7])
    mSteps_interpo = np.array(mInsulin.fillna(0))
    mSteps_interpo.shape = (len(mSteps_interpo),1)
    mIntensity = pd.Series(aligned_data[:,8])
    mIntensity_interpo = np.array(mIntensity.fillna(0))
    mDuration = pd.Series(aligned_data[:,9])
    mDuration_interpo = np.array(mDuration.fillna(0))


    mExercise_interpo = mIntensity_interpo

    for i in range(len(mIntensity_interpo)):
        if(mIntensity_interpo[i] != 0):

            duration = int(round((mDuration_interpo[i]/5)))
            for k in range(0, duration):
                if(mIntensity_interpo[i] == 1 or mIntensity_interpo[i] == 2 or mIntensity_interpo[i] == 3):
                    mExercise_interpo[i+k] = 1 # low intensity exercise
                elif(mIntensity_interpo[i] == 4 or mIntensity_interpo[i] == 5 or mIntensity_interpo[i] == 6 or mIntensity_interpo[i] == 7):
                    mExercise_interpo[i+k] = 1#2 # moderate intensity execise
                else:
                    mExercise_interpo[i+k] = 1#3 #high intensity exercise

    mExercise_interpo.shape = (len(mExercise_interpo),1)
    mCGM_diff = np.ediff1d(mCGM_interpo, to_begin= 0)
    mCGM_diff.shape = (len(mCGM_diff),1)

    # plt.figure(1)
    # plt.subplot(611)
    # plt.plot(to_adverse_event(mCGM_interpo/120, 3), 'g--')
    # plt.subplot(612)
    # plt.plot(mHR_interpo, 'b-o')
    # plt.subplot(613)
    # plt.plot(mGSR_interpo, 'r-o')
    # plt.subplot(614)
    # plt.plot(mST_interpo, 'g-o')
    # plt.subplot(615)
    # plt.plot(mAT_interpo, 'g-o')
    # plt.subplot(616)
    # plt.plot(mExercise_interpo, 'y-o')


    # plt.figure(2)
    # plt.plot(mCGM_diff, 'g-o')

    # plt.show()
    #  standardize(mCGM_diff, -105, 105)
    #  standardize(mCGM_interpo, 40, 400)
    pData = np.concatenate((mCGM_interpo/120, mInsulin_interpo/6000*70, mMeal_interpo/1000*5, mHR_interpo/200, mGSR_interpo/25, mST_interpo/122, mAT_interpo/122, mSteps_interpo/100, mExercise_interpo), axis=1)
    print(pData.shape)

    print("Preprocessing complete")

    return pData,  argFP

#data_full = (np.vstack((data_fm_data[:,1]*18, data_fm_data[:,4], data_fm_data[:,3], data_fm_data[:,2],  data_fm_data[:,0]))).T

# ---------------------------------------------------------
# RETRIEVE PRE-PROCESSED DATA
# --

def parseData(data_src, file_num, xls_file_name, train):
    fwin = 6
    BW = 70
    Ts = 5
    normalize_para = 120  # 120
    argFP = []
    columnHeads = []
    if data_src == 'sim_adult180':
        path = 'C:\\Users\\kaise\\Projects\\ARISES\\Datasets\\GlucosePrediction\\Simulator'
        data_struct = sio.loadmat(os.path.join(path, "data" + str(file_num) + ".mat"))

        G_data = data_struct['G']
        I_data = data_struct['I']
        M_data = data_struct['M']

        src_data = np.concatenate((G_data, I_data, M_data), axis=1)
        columnHeads = ['G', 'I', 'M']
        print('10 adults - 180 days')

    elif data_src == 'sim_adult360':
        path = 'C:\\Users\\kaise\\Projects\\ARISES\\Datasets\\GlucosePrediction\\Adult'
        data_struct_p = sio.loadmat(os.path.join(path, "adult360_" + str(file_num) + ".mat"))
        data_struct = data_struct_p['history']

        G_data = data_struct['CGM'][0,0]
        I_data = data_struct['u6'][0,0]
        M_data = data_struct['u1'][0,0]

        src_data    = np.concatenate((G_data, I_data, M_data), axis=1)
        columnHeads = ['G', 'I', 'M']
        print('10 adults - 360 days')

    elif data_src == 'clinical_abc4d':
        root = 'C:\\Users\\kaise\\Projects\\ARISES\\Datasets\\GlucosePrediction\\Clinical\\ABC4D'
        data_struct, argFP = parseABC4DClinicalData(root, xls_file_name)

        data_fm_data = data_struct.as_matrix()
        print(data_fm_data.shape)
        ''' let data be with length of even value  '''
        if data_fm_data.shape[0] % 2 == 1:
            data_fm_data = data_fm_data[:-1,]
        #src_data = (np.vstack((data_fm_data[:,1]*18, data_fm_data[:,4], data_fm_data[:,3], data_fm_data[:,2], data_fm_data[:,5], data_fm_data[:,6], data_fm_data[:,0]))).T
        data_1 = (data_fm_data[:,1]*18)
        data_2 = (data_fm_data[:,4])#/6000*BW
        data_3 = (data_fm_data[:,3])#/1000*Ts
        data_4 = (data_fm_data[:,5])
        #src_data = (np.vstack((data_fm_data[:,1]*18, data_fm_data[:,4], data_fm_data[:,3]))).T
        src_data = (np.vstack((data_1, data_2, data_3, data_4))).T

        print('ABC4D - Clinical data')

    elif data_src == 'OhioT1DM':
        if(train == 1):
            print("Ohio T1DM Training Dataset")
            ids = [xls_file_name]
            file_train_path = ['C:/Users/kaise/Projects/ARISES/BGLP2/OhioT1DM/Training/raw/' + id + '-ws-training.xml' for id in ids]
            dataRaw, categories, timeStamps, isRealBGL = load_from_xml(zip(ids, file_train_path), res=5, verbose=True)
            #print("")
        else:
            ids = [xls_file_name]
            file_test_path = ['C:/Users/kaise/Projects/ARISES/BGLP2/OhioT1DM/Testing/raw/' + id + '-ws-testing.xml' for id in ids]
            dataRaw, categories, timeStamps, isRealBGL = load_from_xml(zip(ids, file_test_path), res=5, verbose=True)

        # src_data, argFP = parseOhioT1DM(data_struct_p, train)
        src_data 	= np.array(dataRaw[xls_file_name])
        argFP = np.array(isRealBGL[xls_file_name])
        columnHeads = np.array(categories[xls_file_name])

        print('Ohio-T1DM - Clinical data')
    else:
        print('Error!')

    #return src_data, filterData
    return src_data, argFP, columnHeads




def _load_data(dataFrame, n_prev = 24, pred_horizon = 6, out_dim = 1, train = 1, mode = 'basic'):
    """
    data should be pd.DataFrame()
    """
    dataFrame['G'] = dataFrame['G']/120
    dataFrame['I'] = dataFrame['I']/6000*70
    dataFrame['M'] = dataFrame['M']/1000*5
    m = np.array(dataFrame['exercise'].values.tolist())
    dataFrame['exercise'] = np.where(m > 0, 1, m).tolist()
    if(mode == 'basic'):
        data = dataFrame[['G','I','M','exercise']]# 'AT','ST','GSR','HR','STP']] #'Exercise']]#
    elif(mode == 'aux_out'):
        data = dataFrame[['G','I','M','HR','GSR']]#,'STP','ST','AT']] #'Exercise']]#
    else:
        data = dataFrame[['G','I','M','AT','ST','GSR','HR','STP','exercise']]#

    window = pred_horizon #60-min window(12)
    if(window >= pred_horizon):
        time_shift = window
    else:
        time_shift = pred_horizon

    docX, docY, docZ, docTrue, docAux = [], [], [], [], []
    for i in range(len(data) - n_prev - (time_shift)):#+6
        docX.append(data.iloc[i:(i + n_prev)].as_matrix())
        docY.append(data.iloc[i + n_prev + (pred_horizon) - 1]['G'] - data.iloc[i + n_prev - 1]['G'])#/((pred_horizon)*5))#data.iloc[i + n_prev - 1]['G'])  # here 6 represents 30 minutes
        # docY.append((transform(data.iloc[i + n_prev + (pred_horizon) - 1]['G'], 40, 400) - transform(data.iloc[i + n_prev - 1]['G'], 40, 400)))
        # docY.append((data.iloc[i + n_prev + (pred_horizon) - 1]['D']))
        docZ.append(data.iloc[i + n_prev + (pred_horizon) - 1]['G'])
        docAux.append(dataFrame.iloc[i + n_prev - 1]['exercise'])
        docTrue.append(data.iloc[i + n_prev + (pred_horizon) - 1]['G'])
    alsX    = np.array(docX)
    # alsY    = standardize(np.array(docY), -105, 105) #-40 40
    alsY    = np.array(docY)
    alsZ    = np.array(docZ)
    alsEP   = to_adverse_event(alsZ, 3)
    alsAux  = np.array(docAux)
    # alsZ    = transform(np.array(docTrue), 40, 400)
    alsZ    = np.array(docTrue)
    return alsX, alsY, alsEP, alsAux, alsZ

def train_test_split(df, pred_window, test_size=0.5):
    """
    This just split data to training and testing parts
    """
    ntrn = np.int(round(len(df) * (1 - test_size)))

    training = 1
    X_train, y_train, y_train_case, train_ref = _load_data(df.iloc[0:ntrn], n_prev, pred_window, out_dim, training)
    X_test, y_test, y_test_case, test_ref      = [], [], [], []

    if(test_size != 0):
        training = 0
        X_test, y_test, y_test_case, test_ref = _load_data(df.iloc[ntrn:], n_prev, pred_window, out_dim, training)

    return (X_train, y_train, y_train_case), (X_test, y_test, y_test_case), (train_ref, test_ref)

def data_prep_split(df, pred_window, mode):
    """
    This just splits data to training and testing parts
    """
    training = 1
    X_train, y_train, y_train_case, aux_train, train_ref = _load_data(df, n_prev, pred_window, out_dim, training, mode )

    return (X_train, y_train, y_train_case), (aux_train, train_ref)
################
