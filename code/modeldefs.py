from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import GlobalAveragePooling1D


def model_conv_30_100(model, dims):
    '''
    One Conv1D layer with 30 kernels of length 100
    '''
    model.add(Conv1D(30, kernel_size=(100), 
            activation = 'relu', input_shape=dims))
    model.add(GlobalAveragePooling1D())

    return model

def model_conv_30_50(model, dims):
    '''
    One Conv1D layer with 30 kernels of length 50
    '''
    model.add(Conv1D(30, kernel_size=(50), 
            activation = 'relu', input_shape=dims))
    model.add(GlobalAveragePooling1D())

    return model

def model_conv_30_200(model, dims):
    '''
    One Conv1D layer with 30 kernels of length 200
    '''
    model.add(Conv1D(30, kernel_size=(200), 
            activation = 'relu', input_shape=dims))
    model.add(GlobalAveragePooling1D())

    return model

def model_lstm_16(model, dims):
    '''
    One LSTM layer with 16 units
    '''
    model.add(LSTM(16, input_shape=dims))

    return model


modeldefs = { 'conv_30_100': model_conv_30_100,
                'conv_30_200': model_conv_30_200,
                'conv_30_50': model_conv_30_50,
                'lstm_16': model_lstm_16,
 }
