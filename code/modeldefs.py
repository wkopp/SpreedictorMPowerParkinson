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

def model_conv_30_300(model, dims):
    '''
    One Conv1D layer with 30 kernels of length 300
    '''
    model.add(Conv1D(30, kernel_size=(300), 
            activation = 'relu', input_shape=dims))
    model.add(GlobalAveragePooling1D())

    return model

def model_conv_30_400(model, dims):
    '''
    One Conv1D layer with 30 kernels of length 400
    '''
    model.add(Conv1D(30, kernel_size=(400), 
            activation = 'relu', input_shape=dims))
    model.add(GlobalAveragePooling1D())

    return model

def model_conv_30_500(model, dims):
    '''
    One Conv1D layer with 30 kernels of length 500
    '''
    model.add(Conv1D(30, kernel_size=(500), 
            activation = 'relu', input_shape=dims))
    model.add(GlobalAveragePooling1D())

    return model

def model_lstm_16(model, dims):
    '''
    One LSTM layer with 16 units
    '''
    model.add(LSTM(16, input_shape=dims))

    return model

def model_lstm_32(model, dims):
    '''
    One LSTM layer with 32 units
    '''
    model.add(LSTM(32, input_shape=dims))

    return model


modeldefs = { 'conv_30_100': model_conv_30_100,
                'conv_30_200': model_conv_30_200,
                'conv_30_300': model_conv_30_300,
                'conv_30_400': model_conv_30_400,
                'conv_30_500': model_conv_30_500,
                'conv_30_50': model_conv_30_50,
        #        'lstm_16': model_lstm_16,
        #        'lstm_32': model_lstm_32,
 }
