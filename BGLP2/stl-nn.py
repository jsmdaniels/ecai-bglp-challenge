# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 13:07:08 2017

@author: Kezhi Li and John Daniels
simulator_2.0: predict the change of the glucose level, test in simulator data
training: data1.mat, testing: data2.mat
best achieve: rmse_result=10.01 under 3(24)-128-1-80epoch,0.4drop,
"""

from comet_ml import Experiment
import pandas as pd
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import keras
import os
import math
import csv
import time
import datetime

from dateutil   import parser
from dataparser import parseData
from dataparser import data_prep_split
from dataparser import train_test_split

#from extended_keras import LossHistory
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import class_weight
#create an experiment with your api key
#experiment = Experiment(api_key="i2qwxnnc4e1prPNpOieJvJOCe",
#                        project_name='BGLP',
#                        auto_param_logging=False)

from keras.utils import np_utils
from models import CRNN

import random as rn

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.


# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

np.random.seed(7)

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

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

save_run = 0


model_run     = 'regression'
data_mode     = 'basic'
file_path     = 'OhioT1DM'    # for clinical data = 'clinical_abc4d'/'Ohio_T1DM'; for simulated data 360d = 'sim_adult360'; for simulated data 180 = 'sim_adult180'
real_fileID   = "570"         # for Ohio_T1DM # ID_array = ["559", "563", "570", "575", "588", "591","540","544","552","567","584","596"]
sim_fileID    =  3            # for sim_adult360 (1-10); for sim_adult180 (1-2) NOT USEFUL

pred_window     = 6       #

experiment       = 'BGLP'
experiment_model = 'CRNN'
experiment_task  = str(pred_window*5)

# ID_array = ["559"]# "563", "570", "575", "588", "591","540","544","552","567","584","596"]

# for real_fileID in ID_array:
seq_len   = 24 # previous 2 hrs
input_dim = 4
out_dim   = 1

# Hyper parameters
batch_size = 128
nb_epoch   = 20
data_generator = 'processed'

dropout_conv     = 0.19       # drop probability for dropout @ conv layer 0.1
dropout_fc       = 0.01       # drop probability for dropout @ fc layer 0.5
dropout_lstm     = 0.36       # drop probability for dropout @ lstm layer 0.2

if(file_path == 'OhioT1DM'):
    #training of testing data
    is_training = 1
    if(data_generator == 'raw'):
        data_train, ip0, columnHeads = parseData(file_path, sim_fileID, real_fileID, is_training)
        df_train = pd.DataFrame(data = data_train, columns = columnHeads)
        df_train.to_csv("C:/Users/kaise/Projects/ARISES/BGLP2/OhioT1DM/Training/processed/" + real_fileID + ".csv")
        np.savetxt("C:/Users/kaise/Projects/ARISES/BGLP2/OhioT1DM/Training/processed/impute_" + real_fileID + ".csv", ip0, delimiter=",")
    else:
        df_train = pd.read_csv("C:/Users/kaise/Projects/ARISES/BGLP2/OhioT1DM/Training/processed/" + real_fileID + ".csv",  sep=',')

    (X_train_curr, y_train_curr, y_train_case), (aux_train, train_ref) = data_prep_split(df_train, pred_window,data_mode)

    is_training = 0
    if(data_generator == 'raw'):
        data_test, ip1, columnHeads = parseData(file_path, sim_fileID, real_fileID, is_training)
        df_test   = pd.DataFrame(data = data_test , columns=['G', 'I', 'M', 'HR', 'GSR', 'ST', 'AT', 'STP', 'Exercise'])#, 'A', 'B', 'C', 'D'])
        df_test.to_csv("C:/Users/kaise/Projects/ARISES/BGLP2/OhioT1DM/Testing/processed/" + real_fileID + ".csv")
        np.savetxt("C:/Users/kaise/Projects/ARISES/BGLP2/OhioT1DM/Testing/processed/impute_" + real_fileID + ".csv", ip1, delimiter=",")
    else:
        df_test = pd.read_csv("C:/Users/kaise/Projects/ARISES/BGLP2/OhioT1DM/Testing/processed/" + real_fileID + ".csv",  sep=',')
        ip1 = np.squeeze(pd.read_csv("C:/Users/kaise/Projects/ARISES/BGLP2/OhioT1DM/Testing/processed/impute_" + real_fileID + ".csv",  sep=','))
    (X_test, y_test, y_test_case), (aux_test, test_ref) = data_prep_split(df_test, pred_window, data_mode)

    X_train = X_train_curr

    y_train = np.squeeze(y_train_curr)
    y_test  = np.squeeze(y_test)



# create an experiment with your api key
experiment_log = Experiment(api_key="i2qwxnnc4e1prPNpOieJvJOCe",
                       project_name = experiment,# experiment_type + str(train_mode),
                       auto_param_logging=True)


model = CRNN(dropout_conv, dropout_lstm, dropout_fc)


# and now train the model
# batch_size should be appropriate to your memory size
# number of epochs should be higher for real world problems
#with experiment.train():


params={'batch_size': batch_size,
        'epochs': nb_epoch,
        'ID': real_fileID,
        'dropout_conv': dropout_conv,
        'dropout_fc': dropout_fc,
        'dropout_lstm': dropout_lstm,
        'experiment_task': experiment_task,
        'experiment_model': experiment_model
}

with experiment_log.train():
    history = model.fit(X_train, y_train, batch_size = batch_size, epochs = nb_epoch, validation_split = 0.1)  #   validation_split=0.05 128


with experiment_log.test():
    predicted = model.predict(X_test, batch_size = None)


array_len = len(predicted)

mreference = np.zeros((array_len,1))
mmol_curve = np.zeros((array_len,1))

# experiment_log.log_parameters(params)
for index in range(array_len):
    mmol_curve[index] = (X_test[index,-1,0] + (predicted[index]))*120
    mreference[index] = (test_ref[index])*120

# a =  ip1 - seq_len
a =  ip1[(seq_len + pred_window):]
mol_curve = np.delete(mmol_curve, a[a== 0])
reference = np.delete(mreference, a[a== 0])

###
RMSE = np.sqrt(np.mean((reference - mol_curve)**2))
MARD = np.mean(np.abs(100*(reference - mol_curve)/reference))
print("MODEL METRICS")
print("Regression")
print("RMSE: %.2f" % RMSE)
print("MARD: %.2f" % MARD)

metrics = {"RMSE":RMSE, "MARD":MARD}
# experiment_log.log_metrics(metrics)
timestep   = np.array(range(len(mol_curve)))

plt.figure(1)
plt.plot(timestep, mol_curve, 'r-*', timestep, reference, 'b--')
plt.show()


#keras.models.save_model(model, "C:\\Users\\kaise\\Projects\\ARISES\\MTL-Prediction\\Results\\MTL-NN\\safeModel_"+str(sim_fileID))
# red dashes, blue squares and green triangles

### save to csv
if(save_run == 1):
    save_path = "../BGLP2/results/" + experiment_task + "/"+ experiment_run + "/"
    np.savetxt(save_path + "/referenceSTL_" + real_fileID + ".csv", reference, delimiter=",")
    np.savetxt(save_path + "/predictionSTL_" + real_fileID + ".csv", mol_curve, delimiter=",")
    model.save_weights(save_path + "STLmodel_" + real_fileID + ".h5")
