from keras.models import Model, Sequential
from keras.layers import Input, Convolution1D, MaxPooling1D, Dense, Flatten, Dropout, Bidirectional, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam, RMSprop, SGD
from keras.utils import np_utils

from extended_keras import rmse_error


def getSTL_model(real_fileID, ID, dropout_conv = 0.1, dropout_lstm = 0.2, dropout_fc = 0.5, seq_len = 24, input_dim = 4, out_dim = 1):
    # Build DeepCGM network
    model = Sequential()

    # conv1 layer
    model.add(Convolution1D(8, 4, padding = 'causal', input_shape = (seq_len, input_dim), name='Shared_Conv_1'))#4
    model.add(MaxPooling1D(pool_size= 2, name='Shared_MP_1'))

    # conv2 layer
    model.add(Convolution1D(16, 4, padding = 'causal', name='Shared_Conv_2'))
    model.add(MaxPooling1D(pool_size = 2, name='Shared_MP_2'))
    model.add(Dropout(dropout_conv))

    # conv3 layer
    model.add(Convolution1D(32, 4, padding = 'causal', name='Shared_Conv_3'))
    model.add(MaxPooling1D(pool_size = 2, name='Shared_MP_3'))
    model.add(Dropout(dropout_conv))

    #lstm layer
    model.add(LSTM(64, return_sequences=False, name='Shared_Layer_9'))#, input_shape = (seq_len, input_dim)))
    model.add(Dropout(dropout_lstm))

    for i in range(0,len(ID)):
        if (real_fileID in ID[i]):
            j = i

    # fc1 layer
    model.add(Dense(256, name = 'dense_1'+str(j))) #256
    model.add(Dropout(dropout_fc))

    # fc2 layer
    model.add(Dense(32, name = 'dense_2'+str(j)))#32
    model.add(Dropout(dropout_fc))

    # fc3 layer
    model.add(Dense(out_dim, name = "user_" + real_fileID))

    opt = Adam(lr=0.00053) #0.002
    model.compile(optimizer = opt, loss = 'mean_absolute_error', metrics = [rmse_error])


    return model

def getMTL_model(nb_tasks, ID, dropout_conv = 0.1, dropout_lstm = 0.2, dropout_fc = 0.5, seq_len = 24, input_dim = 4, out_dim = 1):

    input_shape = (seq_len, input_dim)
    main_input = Input(shape = input_shape, dtype = 'float32', name = 'Input')

    # conv1 layer
    x = Convolution1D(8, 4, padding = 'causal', name='Shared_Conv_1')(main_input)# 8 5
    x = MaxPooling1D(pool_size = 2, name='Shared_MP_1')(x)

    # conv2 layer padding = 'causal'
    x = Convolution1D(16, 4, padding = 'causal', name='Shared_Conv_2')(x)# 16 5
    x = MaxPooling1D(pool_size = 2, name='Shared_MP_2')(x)
    x = Dropout(dropout_conv, name='Dropout_1')(x)

    # conv3 layer
    x = Convolution1D(32, 4, padding = 'causal', name='Shared_Conv_3')(x)#32 5
    x = MaxPooling1D(pool_size = 2, name='Shared_MP_3')(x)
    x = Dropout(dropout_conv, name='Dropout_2')(x)

    #lstm layer
    x = LSTM(64, return_sequences = False, name='Shared_Layer_9')(x)#(main_input)#(x)
    x = Dropout(dropout_lstm, name='Dropout_3')(x)

    x_cluster = []
    output    = []

    for i in range(nb_tasks):
        x_cluster += [Dense(256, name = 'dense_1'+str(i))(x)]#256
        x_cluster[i] = Dropout(dropout_fc, name = 'dropout_4'+str(i))(x_cluster[i])

        # fc2 layer
        x_cluster[i] = Dense(32, name = 'dense_2'+str(i))(x_cluster[i])#32
        x_cluster[i] = Dropout(dropout_fc, name = 'dropout_5'+str(i))(x_cluster[i])

        cluster_num = len(ID[i])
        for w in range(cluster_num):
            # fc3 layer
            output += [Dense(out_dim, name = "user_" + ID[i][w])(x_cluster[i])]

    return Model([main_input], output)
