import sys
import keras
import math

import numpy as np
import tensorflow as tf

from keras import backend as K


def rmse_error(y_true, y_pred):
    error =  K.abs(y_true - y_pred) * (120)
    return K.mean(error)

def mard(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                            K.epsilon(),
                                            None))

    return K.clip(K.mean(diff, axis=-1),0.01, 1)

class LossMasking(keras.callbacks.Callback):
    def __init__(self, nb_users, iteration_limit, task_weights):
        self.batch_count = 1
        self.task_weights   = task_weights
        self.iter           = iteration_limit
        self.outputs        = nb_users


    def on_train_begin(self, logs={}):
        self.w_mu = []

    def on_batch_end(self, batch, logs={}):

        if(self.batch_count == self.outputs):
            self.batch_count                   = 1

            K.set_value(self.task_weights[self.outputs - 1], 0)
            K.set_value(self.task_weights[self.batch_count - 1], 1)

        else:
            K.set_value(self.task_weights[self.batch_count - 1], 0)
            self.batch_count                  += 1

            K.set_value(self.task_weights[self.batch_count - 1], 1)


        # self.w_mu.append(K.get_value(self.task_weights[0]))
        # for i in range(len(task_weights)):
        #     K.set_value(self.loss_mask[i], task_weights[i])
