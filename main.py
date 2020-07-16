# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 09:07:08 2018

@author: John Daniels
MTL-NN:  A multi-task neural network model that uses CGM data, insulin and meal data to predict the CGM value and adverse events at a given instant.
Dataset: uVA/Padova simulator dataset and ABC4D trial dataset
Result: 19.79 mg/dL (RMSE)
"""

#In[1]:

from comet_ml import Experiment
import pandas as pd


import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io as sio
import numpy as np
import keras
import os
import math
import csv
import time
import datetime
import hyperopt

from hyperopt import hp, STATUS_OK

from extended_keras import mard
from extended_keras import rmse_error
from extended_keras import LossMasking

from dateutil   import parser
from datetime   import datetime
from dataparser import parseData
from dataparser import transform
from dataparser import data_prep_split

from keras.models import Model
from keras.layers import Input, Convolution1D, MaxPooling1D, Dense, Flatten, Dropout, Bidirectional, BatchNormalization
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam, RMSprop, SGD
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras import backend as K

from models import getMTL_model, getSTL_model

import random as rn


#In[2]:
# Setting the API key (saved as environment variable)
experiment       = 'MTL-Personalisation'
experiment_model = 'CRNN'


# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

np.random.seed(0)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

rn.seed(0)#12345

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/set_random_seed

tf.set_random_seed(0)#1234

sess = tf.Session(graph=tf.get_default_graph(), config = session_conf)
K.set_session(sess)

#In[3]:
save_run = 0


file_path       = 'OhioT1DM'    # for clinical data = 'clinical_abc4d'/'Ohio_T1DM'; for simulated data 360d = 'sim_adult360'; for simulated data 180 = 'sim_adult180'
data_mode       = 'basic'#'basic'
pred_horizon    = [6, 12]


ID                = [["563","570","540","544","552","584","596"],["559","575","588","591","567"]]
userID            = ["563","570","540","544","552","584","596","559","575","588","591","567"]
real_fileID = "559"


seq_len   = 24 # previous 2 hrs
input_dim = 4
out_dim   = 1

# Hyper parameters
batch_size = 128
nb_epoch   = 1#200

# Parameters for LSTM network
nb_lstm_outputs  = 64        # number of hidden units

nb_clusters      = len(ID)
nb_users         = len(userID)
nb_classes       = 3         # number of classes of adverse glycaemic events
exercise_classes = 4         # number if classes of exercise intensity

dropout_conv     = 0.10       # drop probability for dropout @ conv layer 0.1 0.19
dropout_fc       = 0.50       # drop probability for dropout @ fc layer 0.5 0.01
dropout_lstm     = 0.20       # drop probability for dropout @ lstm layer 0.2 0.36

data_generator      = 'processed'
task_type           = 'regression'
run_from_file       = 0

#In[4}:
sample_size = 0
batch_num   = 0
sample_size_list = []

j = 0
prediction_horizon   = pred_horizon[j]

if(run_from_file ==0):
    if(file_path == 'OhioT1DM'):
        sim_fileID = 3
        for real_fileID in userID:
            is_training = 1
            if(data_generator == 'raw'):
                data_train, ip0 , train_labels, train_timestamps = parseData(file_path, sim_fileID, real_fileID, is_training)
                df_train = pd.DataFrame(data = data_train, columns=train_labels)
                # df_train.to_csv("C:/Users/kaise/Projects/ARISES/Datasets/OhioT1DM/Training/processed/" + real_fileID + ".csv")
                # np.savetxt("C:/Users/kaise/Projects/ARISES/Datasets/OhioT1DM/Training/processed/impute_" + real_fileID + ".csv", ip0, delimiter=",")
            else:
                df_train = pd.read_csv("../Datasets/OhioT1DM/Training/processed/" + real_fileID + ".csv",  sep=',')
                ip0 = pd.read_csv("../Datasets/OhioT1DM/Training/processed/impute_" + real_fileID + ".csv",  sep=',', header=None)


            (X_train_temp, y_train_temp, train_ref) = data_prep_split(df_train, prediction_horizon,data_mode)

            print(real_fileID)
            del_arrays = []
            num_index = 0
            for i in range(0, X_train_temp.shape[0]):
                a = ip0[i:i+seq_len]
                if(np.count_nonzero(a) < 12):
                    num_index = num_index + 1
                    del_arrays += [i]

            X_train_curr = np.delete(X_train_temp, del_arrays, axis=0)
            y_train_curr = np.delete(y_train_temp, del_arrays, axis=0)

            print('train samples sizes')
            print(num_index)
            print(X_train_temp.shape[0])
            print(y_train_curr.shape[0])
            print(X_train_curr.shape[0])

            if(real_fileID == userID[0]):
                X1_train   = X_train_curr

                y1_train = y_train_curr
                y1_train = y1_train.reshape(len(y1_train),1)


                sample_size_1 = len(y_train_curr)
                batch_num_1   = int(sample_size_1/batch_size)
                sample_size   = sample_size + sample_size_1
                batch_num     = batch_num + batch_num_1

                sample_size_list.append(sample_size_1)

            if(real_fileID == userID[1]):
                X2_train = X_train_curr

                y2_train = y_train_curr
                y2_train = y2_train.reshape(len(y2_train),1)

                sample_size_2 = len(y_train_curr)
                batch_num_2   = int(sample_size_2/batch_size)
                sample_size   = sample_size + sample_size_2
                batch_num     = batch_num + batch_num_2

                sample_size_list.append(sample_size_2)

            if(real_fileID == userID[2]):
                X3_train = X_train_curr

                y3_train = y_train_curr
                y3_train = y3_train.reshape(len(y3_train),1)


                sample_size_3   = len(y_train_curr)
                batch_num_3     = int(sample_size_3/batch_size)
                sample_size     = sample_size + sample_size_3
                batch_num       = batch_num + batch_num_3

                sample_size_list.append(sample_size_3)

            if(real_fileID == userID[3]):
                X4_train = X_train_curr

                y4_train = y_train_curr
                y4_train = y4_train.reshape(len(y4_train),1)

                sample_size_4   = len(y_train_curr)
                batch_num_4     = int(sample_size_4/batch_size)
                sample_size     = sample_size + sample_size_4
                batch_num       = batch_num + batch_num_4

                sample_size_list.append(sample_size_4)

            if(real_fileID == userID[4]):
                X5_train = X_train_curr

                y5_train = y_train_curr
                y5_train = y5_train.reshape(len(y5_train),1)


                sample_size_5   = len(y_train_curr)
                batch_num_5     = int(sample_size_5/batch_size)
                sample_size     = sample_size + sample_size_5
                batch_num       = batch_num + batch_num_5

                sample_size_list.append(sample_size_5)


            if(real_fileID == userID[5]):
                X6_train = X_train_curr

                y6_train = y_train_curr
                y6_train = y6_train.reshape(len(y6_train),1)


                sample_size_6   = len(y_train_curr)
                batch_num_6     = int(sample_size_6/batch_size)
                sample_size     = sample_size + sample_size_6
                batch_num       = batch_num + batch_num_6

                sample_size_list.append(sample_size_6)

            if(real_fileID == userID[6]):
                X7_train = X_train_curr

                y7_train = y_train_curr
                y7_train = y7_train.reshape(len(y7_train),1)


                sample_size_7   = len(y_train_curr)
                batch_num_7     = int(sample_size_7/batch_size)
                sample_size     = sample_size + sample_size_7
                batch_num       = batch_num + batch_num_7

                sample_size_list.append(sample_size_7)

            if(real_fileID == userID[7]):
                X8_train = X_train_curr

                y8_train = y_train_curr
                y8_train = y8_train.reshape(len(y8_train),1)


                sample_size_8   = len(y_train_curr)
                batch_num_8     = int(sample_size_8/batch_size)
                sample_size     = sample_size + sample_size_8
                batch_num       = batch_num + batch_num_8

                sample_size_list.append(sample_size_8)

            if(real_fileID == userID[8]):
                X9_train = X_train_curr

                y9_train = y_train_curr
                y9_train = y9_train.reshape(len(y9_train),1)

                sample_size_9   = len(y_train_curr)
                batch_num_9     = int(sample_size_9/batch_size)
                sample_size     = sample_size + sample_size_9
                batch_num       = batch_num + batch_num_9

                sample_size_list.append(sample_size_9)

            if(real_fileID == userID[9]):
                X10_train = X_train_curr

                y10_train = y_train_curr
                y10_train = y10_train.reshape(len(y10_train),1)


                sample_size_10   = len(y_train_curr)
                batch_num_10     = int(sample_size_10/batch_size)
                sample_size     = sample_size + sample_size_10
                batch_num       = batch_num + batch_num_10

                sample_size_list.append(sample_size_10)

            if(real_fileID == userID[10]):
                X11_train = X_train_curr

                y11_train = y_train_curr
                y11_train = y11_train.reshape(len(y11_train),1)


                sample_size_11   = len(y_train_curr)
                batch_num_11     = int(sample_size_11/batch_size)
                sample_size     = sample_size + sample_size_11
                batch_num       = batch_num + batch_num_11

                sample_size_list.append(sample_size_11)

            if(real_fileID == userID[11]):
                X12_train = X_train_curr

                y12_train = y_train_curr
                y12_train = y12_train.reshape(len(y12_train),1)

                sample_size_12   = len(y_train_curr)
                batch_num_12     = int(sample_size_12/batch_size)
                sample_size     = sample_size + sample_size_12
                batch_num       = batch_num + batch_num_12

                sample_size_list.append(sample_size_12)

        sample_size_min = min(sample_size_list)
        batch_num_min   = int(sample_size_min/batch_size)


        samples = nb_users*batch_size*(batch_num_min)
        y_train = np.zeros((samples, nb_users))

        for batch_set in range(0, batch_num_min):
            start = int(batch_set * batch_size)
            end   = int((batch_set + 1) * batch_size)

            for n in range(0, nb_users):
                if(n == 0):
                    temp_batch_x = X1_train[start:end]
                    temp_batch_y = y1_train[start:end]
                elif(n == 1):
                    temp_batch_x = X2_train[start:end]
                    temp_batch_y = y2_train[start:end]
                elif(n == 2):
                    temp_batch_x = X3_train[start:end]
                    temp_batch_y = y3_train[start:end]
                elif(n == 3):
                    temp_batch_x = X4_train[start:end]
                    temp_batch_y = y4_train[start:end]
                elif(n == 4):
                    temp_batch_x = X5_train[start:end]
                    temp_batch_y = y5_train[start:end]
                elif(n == 5):
                    temp_batch_x = X6_train[start:end]
                    temp_batch_y = y6_train[start:end]
                elif(n == 6):
                    temp_batch_x = X7_train[start:end]
                    temp_batch_y = y7_train[start:end]
                elif(n == 7):
                    temp_batch_x = X8_train[start:end]
                    temp_batch_y = y8_train[start:end]
                elif(n == 8):
                    temp_batch_x = X9_train[start:end]
                    temp_batch_y = y9_train[start:end]
                elif(n == 9):
                    temp_batch_x = X10_train[start:end]
                    temp_batch_y = y10_train[start:end]
                elif(n == 10):
                    temp_batch_x = X11_train[start:end]
                    temp_batch_y = y11_train[start:end]
                elif(n == 11):
                    temp_batch_x = X12_train[start:end]
                    temp_batch_y = y12_train[start:end]
                else:
                    print("No ID")

                if(batch_set == 0 and n ==0):
                    X_train = temp_batch_x
                else:
                    X_train = np.vstack((X_train, temp_batch_x))

                y_index = len(X_train)
                print(y_index)
                if(out_dim == 1):
                    y_train[(y_index - batch_size): y_index, n] = np.squeeze(temp_batch_y)
                else:
                    y_train[(y_index - batch_size): y_index, n] = np.array(temp_batch_y)


        #validation sets
        val_start = (batch_num_min - 1)*batch_size

        X1_val = X1_train[val_start:val_start+batch_size]
        y1_val = y1_train[val_start:val_start+batch_size]

        X2_val = X2_train[val_start:val_start+batch_size]
        y2_val = y2_train[val_start:val_start+batch_size]
        #
        X3_val = X3_train[val_start:val_start+batch_size]
        y3_val = y3_train[val_start:val_start+batch_size]

        X4_val = X4_train[val_start:val_start+batch_size]
        y4_val = y4_train[val_start:val_start+batch_size]
        #
        X5_val = X5_train[val_start:val_start+batch_size]
        y5_val = y5_train[val_start:val_start+batch_size]

        X6_val = X6_train[val_start:val_start+batch_size]
        y6_val = y6_train[val_start:val_start+batch_size]

        X7_val = X7_train[val_start:val_start+batch_size]
        y7_val = y7_train[val_start:val_start+batch_size]

        X8_val = X8_train[val_start:val_start+batch_size]
        y8_val = y8_train[val_start:val_start+batch_size]

        X9_val = X9_train[val_start:val_start+batch_size]
        y9_val = y9_train[val_start:val_start+batch_size]

        X10_val = X10_train[val_start:val_start+batch_size]
        y10_val = y10_train[val_start:val_start+batch_size]

        X11_val = X11_train[val_start:val_start+batch_size]
        y11_val = y11_train[val_start:val_start+batch_size]

        X12_val = X12_train[val_start:val_start+batch_size]
        y12_val = y12_train[val_start:val_start+batch_size]


#In[5]:

input_shape = (seq_len, input_dim)

opt = Adam(lr=0.00053)

model = getMTL_model(nb_clusters, ID)


loss_mask    = dict()
user_losses  = dict()
precision    = dict()
event_weight = dict()


loss_weight = []

for i in range(len(userID)):

    user_losses['user_'+ userID[i]] =  'mean_absolute_error'
    precision['user_'+ userID[i]]   =  rmse_error

    if(i == 0):
        loss_weight                      += [K.variable(1)]
        loss_mask['user_'+ userID[i]]   = loss_weight[i]
    else:
        loss_weight                      += [K.variable(0)]
        loss_mask['user_'+ userID[i]]   = loss_weight[i]



model.compile(optimizer = opt, loss = user_losses, loss_weights = loss_mask, metrics = precision)
model.summary()

iteration_limit = nb_epoch

LossMask = LossMasking(nb_users, iteration_limit, loss_weight)


y_train_hat = []

if(run_from_file == 0):
    for i in range(nb_users):
        y_train_hat += [y_train[:,i]]
    history = model.fit(X_train, y_train_hat, epochs = nb_epoch, batch_size = batch_size, shuffle = False, verbose = 0, validation_split = 0, callbacks=[LossMask])

    save_path = "INSERT_PATH_HERE"
    model.save_weights(save_path + "MTLmodel_" + str(run) + ".h5")
else:
    loadpath = save_path


#In[6]:

sample_size = 0
batch_num   = 0
sample_size_list = []

pred_horizon = [6]

file2 = open('../BGLP2/results/summaryMTL.txt','a')
file2.write("All figures in mg/dL \n\n")

for prediction_horizon in pred_horizon:
    resRMSE = []
    resMARD = []
    RMSE_1 = []
    MAE_1 = []
    RMSE_2 = []
    MAE_2 = []
    RMSE_3 = []
    MAE_3 = []
    RMSE_4 = []
    MAE_4 = []
    RMSE_5 = []
    MAE_5 = []
    RMSE_6 = []
    MAE_6 = []


    for run in range(1, 2):#11

        nMARD = []
        nRMSE = []
        test_ID = ["540","544","552","584","596","567"]

        for real_fileID in test_ID:

            params={'batch_size':batch_size,
                    'epochs':nb_epoch,
                    'ID': real_fileID,
                    'prediction_horizon':prediction_horizon,
                    'dropout_conv': dropout_conv,
                    'dropout_fc':dropout_fc,
                    'dropout_lstm':dropout_lstm,
                    'experiment_model': experiment_model
            }

            # experiment_log.log_parameters(params)
            data_generator = 'raw'
            sim_fileID = 3
            if(data_generator == 'raw'):

                is_training = 1
                data_train, ip0 , train_labels, train_timestamps= parseData(file_path, sim_fileID, real_fileID, is_training)
                append_train = data_train[-1*(prediction_horizon + 11):, :] #get data from
                df1   = pd.DataFrame(data = append_train , columns = train_labels)
                is_training = 0
                data_test, ip1 , test_labels, test_timestamps= parseData(file_path, sim_fileID, real_fileID, is_training)
                df2   = pd.DataFrame(data = data_test , columns = test_labels)
                frames = [df1, df2]
                df_test = pd.concat(frames)

                df_test.to_csv("../Datasets/OhioT1DM/Testing/processed/" + real_fileID + ".csv")
                np.savetxt("../Datasets/OhioT1DM/Testing/processed/impute_" + real_fileID + ".csv", ip1, delimiter=",")
            else:
                df_test = pd.read_csv("../Datasets/OhioT1DM/Testing/processed/" + real_fileID + ".csv",  sep=',')
                ip1 = np.squeeze(pd.read_csv("../Datasets/OhioT1DM/Testing/processed/impute_" + real_fileID + ".csv",  sep=','))

            (X_test, y_test, test_ref) = data_prep_split(df_test, prediction_horizon, data_mode)

            if(real_fileID == "559"):
                y_test   = y_test.reshape(len(y_test),1)

            if(real_fileID == "563"):
                y_test   = y_test.reshape(len(y_test),1)

            if(real_fileID == "570"):
                y_test   = y_test.reshape(len(y_test),1)

            if(real_fileID == "575"):
                y_test   = y_test.reshape(len(y_test),1)

            if(real_fileID == "588"):
                y_test   = y_test.reshape(len(y_test),1)

            if(real_fileID == "591"):
                y_test   = y_test.reshape(len(y_test),1)

            if(real_fileID == "540"):
                y_test   = y_test.reshape(len(y_test),1)


            if(real_fileID == "544"):
                y_test   = y_test.reshape(len(y_test),1)


            if(real_fileID == "552"):
                y_test   = y_test.reshape(len(y_test),1)


            if(real_fileID == "567"):
                y_test   = y_test.reshape(len(y_test),1)


            if(real_fileID == "584"):
                y_test   = y_test.reshape(len(y_test),1)


            if(real_fileID == "596"):
                y_test   = y_test.reshape(len(y_test),1)



            # load weights from MTL model into STL architecture
            loadpath = save_path
            modelSTL = getSTL_model(real_fileID, ID)

            modelSTL.load_weights(loadpath + "MTLmodel_"+ str(run) + ".h5", by_name=True)


            array_len = len(y_test)
            mreference = np.zeros((array_len,))
            mmol_curve = np.zeros((array_len,))
            imp_values = np.zeros((array_len,))

            k = prediction_horizon + seq_len - 1

            print(real_fileID)
            if(task_type == 'regression'):
                for index in range(array_len):

                    pred_ind = [modelSTL.predict(X_test[index:(index+1),:,:], batch_size = None)]

                    mmol_curve[index] = (X_test[index,-1,0] + pred_ind[0])

                    imp_values[index] = mmol_curve[index]
                    mreference[index] = (test_ref[index])

                    m = index + 1

                    if(m >= k):
                        X_test_ip  = ip1[(m - k) + 12 : (m - k) + 12 + seq_len]
                        X_test_hat = np.zeros((seq_len,))
                        mar   = seq_len - np.count_nonzero(X_test_ip)
                        s     = np.where(X_test_ip == False)
                        s_inv = np.where(X_test_ip == True)
                        if(mar != 0):
                            # model-based imputation
                            if(mar <= 6):
                                X_test_hat = imp_values[(m - k) : (m - k) + seq_len]
                            else:
                                # print(s)
                                s_max = np.max(s)
                                s_min = np.min(s)

                                if(s_max == seq_len - 1):
                                    # padding imputation
                                    X_test_hat[:-1]  = X_test[index:(index+1), 1:,0]
                                    X_test_hat[ -1]  = X_test[index:(index+1),-1, 0]
                                else:
                                    X_test_hat = X_test[m : m + 1, :, 0][0]

                            X_test[m : m + 1, s, 0] = X_test_hat[s]



                ip_index =  len(ip1) - len(y_test)

                a =  ip1[ip_index:]

                timestamp = test_timestamps[real_fileID]

                test_timestamp = np.delete(timestamp, np.where(ip1==False))
                test_timestamp_m = test_timestamp[12:]

                mol_curve = 120*np.delete(mmol_curve, np.where(a==False))
                reference = 120*np.delete(mreference, np.where(a==False))

                print(len(mol_curve))

                mol_curve_plot = 120*(np.where(a==False, np.nan, mmol_curve))
                reference_plot = 120*(np.where(a==False, np.nan, mreference))

                if(run == 2 and ((real_fileID == "596") or (real_fileID == "552"))):
                    plt.figure(1)
                    plt.plot(mol_curve_plot, 'r-', reference_plot, 'b--')

                    plt.ylabel('Glucose concentration level (mg/dL)')
                    plt.legend(('Prediction', 'Reference'))
                    plt.title('A comparison of prediction and reference timeseries')
                    plt.grid(True)
                    plt.figure(2)
                    plt.plot(test_timestamp_m, mol_curve, 'r-', test_timestamp_m, reference, 'b--')
                    plt.ylabel('Glucose concentration level (mg/dL)')
                    plt.xlabel('Datetime')
                    plt.legend(('Prediction', 'Reference'))
                    plt.grid(True)
                    plt.figure(3)
                    plt.plot(mol_curve, 'r-', reference, 'b--')
                    plt.ylabel('Glucose concentration level (mg/dL)')
                    plt.legend(('Prediction', 'Reference'))
                    plt.grid(True)
                    plt.show()

                save_path_2 = "INSERT_PATH_HERE"
                file1 = open(save_path_2 + "MTCRNN_" +  real_fileID + "_" + str(prediction_horizon*5) +".txt",'a')
                for u in range(0, len(mol_curve)):
                    g = test_timestamp_m[u]
                    t += g.strftime("%m/%d/%Y, %H:%M:%S")
                    file1.write(g.strftime("%m/%d/%Y, %H:%M:%S")+"\t"+str(mol_curve[u]) + "\n")

                ###
                RMSE = np.sqrt(np.mean((reference - mol_curve)**2))
                MARD = np.mean(np.abs((reference - mol_curve)))
                print("MODEL METRICS")
                print("Regression")
                print("RMSE: %.2f" % RMSE)
                print("MARD: %.2f" % MARD)

                metrics = {"RMSE":RMSE, "MARD":MARD}
                timestep   = np.array(range(len(mol_curve)))

                nRMSE +=[RMSE]
                nMARD +=[MARD]

                if(real_fileID == '540'):
                    RMSE_1 += [RMSE]
                    MAE_1 += [MARD]
                elif(real_fileID == '544'):
                    RMSE_2 += [RMSE]
                    MAE_2 +=[MARD]
                elif(real_fileID == '552'):
                    RMSE_3 += [RMSE]
                    MAE_3 += [MARD]
                elif(real_fileID == '567'):
                    RMSE_4 += [RMSE]
                    MAE_4 += [MARD]
                elif(real_fileID == '584'):
                    RMSE_5 += [RMSE]
                    MAE_5 += [MARD]
                elif(real_fileID == '596'):
                    RMSE_6 += [RMSE]
                    MAE_6 += [MARD]
                else:
                    print("file_ID error")



            if(save_run == 1):
                save_path_3 = "../BGLP2/results/" + str(run) + "/"
                if(experiment == 'MTL-Personalisation'):

                    if(run == 1):
                        np.savetxt(save_path_3 + str(real_fileID)  + "/MTCRNN_" +  real_fileID + "_" + str(prediction_horizon*5) +".csv", reference, delimiter=",")
                    np.savetxt(save_path_3 + str(real_fileID)  + "/MTCRNN_" +  real_fileID + "_" + str(prediction_horizon*5) +".csv", mol_curve, delimiter=",")
                    file1 = open(save_path_3 + "MTCRNN_" +  real_fileID + "_" + str(prediction_horizon*5) +".txt",'a')

                    for u in range(0, len(test_timestamp_m)):
                        g = test_timestamp_m[u]
                        print(g.strftime("%m/%d/%Y, %H:%M:%S"))
                        file1.write(g.strftime("%m/%d/%Y, %H:%M:%S")+"\t"+str(mol_curve[u]) + "\n")

        avRMSE = np.mean(nRMSE)
        print(avRMSE)
        avMARD = np.mean(nMARD)
        print(avMARD)

        resMARD += [avMARD]
        resRMSE += [avRMSE]

    print(resRMSE)
    print(np.mean(resRMSE))
    print(np.std(resRMSE))
    print(np.mean(resMARD))
    print(np.std(resMARD))

    file2.write("Individual - "+ str(prediction_horizon*5) +" minutes \n\n")
    file2.write("540\n")
    file2.write("RMSE:\t"+ "{:.2f}".format(np.mean(RMSE_1))+ "±" + "{:.2f}".format(np.std(RMSE_1))+"\n")
    file2.write("MAE:\t"+ "{:.2f}".format(np.mean(MAE_1))+ "±" + "{:.2f}".format(np.std(MAE_1))+"\n")
    file2.write("\n544\n")
    file2.write("544 RMSE:\t"+ "{:.2f}".format(np.mean(RMSE_2))+ "±" + "{:.2f}".format(np.std(RMSE_2))+"\n")
    file2.write("544 MAE:\t"+ "{:.2f}".format(np.mean(MAE_2))+ "±" + "{:.2f}".format(np.std(MAE_2))+"\n")
    file2.write("\n552\n")
    file2.write("RMSE:\t"+ "{:.2f}".format(np.mean(RMSE_3))+ "±" + "{:.2f}".format(np.std(RMSE_3))+"\n")
    file2.write("MAE:\t"+ "{:.2f}".format(np.mean(MAE_3))+ "±" + "{:.2f}".format(np.std(MAE_3))+"\n")
    file2.write("\n567\n")
    file2.write("RMSE:\t"+ "{:.2f}".format(np.mean(RMSE_4))+ "±" + "{:.2f}".format(np.std(RMSE_4))+"\n")
    file2.write("MAE:\t"+ "{:.2f}".format(np.mean(MAE_4))+ "±" + "{:.2f}".format(np.std(MAE_3))+"\n")
    file2.write("\n584\n")
    file2.write("RMSE:\t"+ "{:.2f}".format(np.mean(RMSE_5))+ "±" + "{:.2f}".format(np.std(RMSE_5))+"\n")
    file2.write("MAE:\t"+ "{:.2f}".format(np.mean(MAE_5))+ "±" + "{:.2f}".format(np.std(MAE_5))+"\n")
    file2.write("\n596\n")
    file2.write("RMSE:\t"+ "{:.2f}".format(np.mean(RMSE_6))+ "±" + "{:.2f}".format(np.std(RMSE_6))+"\n")
    file2.write("MAE:\t"+ "{:.2f}".format(np.mean(MAE_6))+ "±" + "{:.2f}".format(np.std(MAE_6))+"\n")

    file2.write("\nOverall - "+ str(prediction_horizon*5) +" minutes \n")
    file2.write("RMSE:\t"+ "{:.2f}".format(np.mean(resRMSE))+ "±" + "{:.2f}".format(np.std(resRMSE))+"\n")
    file2.write("MAE:\t" +"{:.2f}".format(np.mean(resMARD))+ "±" + "{:.2f}".format(np.std(resMARD))+"\n")
