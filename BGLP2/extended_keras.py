import sys
import numpy as np
from functools import partial
import math
from itertools import product
from scipy.stats.mstats import gmean
import tensorflow as tf
import keras
from keras import backend as K
from keras.layers import Input, Dense, Lambda, Layer
from keras.initializers import Constant
from keras.models import Model
from sklearn.metrics import f1_score
'''
 ' Huber loss.
 ' https://en.wikipedia.org/wiki/Huber_loss
'''

clip_delta = 5./120#0.063 #0.17 #4 #0.07 #0.43 #1.71

def huber_loss(y_true, y_pred, clip_delta=1.0):
  error = y_true - y_pred
  cond  = K.abs(error) < clip_delta
  squared_loss = 0.5 * K.square(error)
  linear_loss  = clip_delta * (K.abs(error) - 0.5 * clip_delta)

  return K.switch(cond, squared_loss, linear_loss)
  #return  K.switch(cond, linear_loss, squared_loss)

'''
 ' Same as above but returns the mean loss.
'''
def huber_loss_mean(y_true, y_pred, clip_delta=1.0):
  return K.mean(huber_loss(y_true, y_pred, clip_delta))


def rmse_error(y_true, y_pred):
    error =  K.abs(y_true - y_pred) * (120)
    return K.mean(error)

def mard(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                            K.epsilon(),
                                            None))
    #1 - K.mean(diff, axis=-1)
    #error = (K.abs(y_pred - y_true))
    #reference = K.abs(y_true)
    return K.clip(K.mean(diff, axis=-1),0.01, 1) #K.clip(K.mean(error/reference),0.01, 1)
