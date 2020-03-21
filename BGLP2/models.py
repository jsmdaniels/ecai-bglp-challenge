from keras.models import Model, Sequential
from keras.layers import Input, Convolution1D, MaxPooling1D, Dense, Flatten, Dropout, Bidirectional, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam, RMSprop, SGD
from keras.utils import np_utils

from extended_keras import huber_loss_mean
from extended_keras import rmse_error


def CRNN(dropout_conv = 0.1, dropout_lstm = 0.2, dropout_fc = 0.5, model_run = 'regression', seq_len = 24, input_dim = 4, out_dim = 1):
    # Build DeepCGM network
    model = Sequential()

    # conv1 layer
    model.add(Convolution1D(8, 4, activation = 'linear', padding = 'causal', input_shape = (seq_len, input_dim)))#4
    model.add(MaxPooling1D(pool_size= 2))

    #model.add(Dropout(prob_drop_conv))

    # conv2 layer
    model.add(Convolution1D(16, 4, activation = 'linear', padding = 'causal'))
    model.add(MaxPooling1D(pool_size = 2))
    model.add(Dropout(dropout_conv))

    # conv3 layer
    model.add(Convolution1D(32, 4, activation='linear', padding = 'causal'))
    model.add(MaxPooling1D(pool_size = 2))
    model.add(Dropout(dropout_conv))

    #lstm layer
    model.add(LSTM(64, return_sequences=True))#, input_shape = (seq_len, input_dim)))
    model.add(Dropout(dropout_lstm))
    model.add(Flatten())

    if(model_run == 'regression'):
        # fc3 layer
        # fc1 layer
        model.add(Dense(256, activation = 'linear')) #256
        model.add(Dropout(dropout_fc))

        # fc2 layer
        model.add(Dense(32, activation = 'linear'))#32
        model.add(Dropout(dropout_fc))

        model.add(Dense(out_dim, activation = 'linear'))

        # opt = RMSprop(lr=0.002, rho=0.9) #0.002
        opt = Adam(lr=0.00053) #0.002
        # opt = Adam(lr=0.002783338849404002) #0.002 #0.002783338849404002
        model.compile(optimizer = opt, loss = huber_loss_mean, metrics = [rmse_error])
    else:
        # fc1 layer
        model.add(Dense(64, activation = 'relu')) #256
        model.add(Dropout(dropout_fc))

        # fc2 layer
        model.add(Dense(10, activation = 'relu'))#32
        model.add(Dropout(dropout_fc))

        # fc3 layer
        model.add(Dense(nb_classes, activation = 'softmax'))

        #opt = RMSprop(lr=0.002, rho=0.9) #0.002 #0.002783338849404002
        opt = Adam(lr=0.0005) #0.002
        model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

    return model
