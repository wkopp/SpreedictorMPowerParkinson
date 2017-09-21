from keras.layers import Input, Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.pooling import AveragePooling1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise
from keras.layers.core import Dense, Dropout, Flatten, Activation
from functools import wraps


def compatibilityCheck(expected_args):
    def decorator(func):
        @wraps(func)
        def wrapper(*args):
            print(args)
            print(expected_args)
            if not type(args[0]) == dict:
                raise Exception("argument must be a dictionary")
            if not set(args[0].keys()) == set(expected_args):
                raise Exception("data and model not compatible")
            return func(*args)
        return wrapper
    return decorator

@compatibilityCheck(['input_1'])
def model_conv_glob(data, paramdims):
    '''
    Conv1D:
        {} x {}, relu
        GlobPool
    '''
    input = Input(shape=data['input_1'].shape, name='input_1')
    layer = Conv1D(paramdims[0], kernel_size=(paramdims[1]), activation = 'relu')(input)
    output = GlobalAveragePooling1D()(layer)
    return input, output

@compatibilityCheck(['input_1'])
def model_pool_conv_glob(data, paramdims):
    '''
    Conv1D:
        {} pool
        {} x {}, relu
        GlobPool
    '''
    input = Input(shape=data['input_1'].shape, name='input_1')
    layer = AveragePooling1D(paramdims[0])(input)
    layer = Conv1D(paramdims[1], kernel_size=(paramdims[2]), activation = 'relu')(layer)
    output = GlobalAveragePooling1D()(layer)
    return input, output

@compatibilityCheck(['input_1'])
def model_conv_2l_glob(data, paramdims):
    '''
    Conv1D:
        {} x {}, relu
        {} pooling
        {} x
        GlobPool
    '''
    input = Input(shape=data['input_1'].shape, name='input_1')
    layer = Conv1D(paramdims[0], kernel_size=(paramdims[1]),
            activation = 'relu')(input)

    layer = BatchNormalization()(layer)

    layer = MaxPooling1D(pool_size=paramdims[2])(layer)

    layer = Conv1D(paramdims[3], kernel_size=(paramdims[4]),
            activation = 'relu')(layer)
    layer = BatchNormalization()(layer)
    output = GlobalAveragePooling1D()(layer)
    return input, output

@compatibilityCheck(['input_1'])
def model_conv_3l_glob(data, paramdims):
    '''
    Conv1D:
        {} x {}, relu
        {} pooling
        {} x {}
        {} pooling
        {} Dense
        GlobPool
    '''
    input = Input(shape=data['input_1'].shape, name='input_1')
    layer = Conv1D(paramdims[0], kernel_size=(paramdims[1]),
            activation = 'relu')(input)
    layer = BatchNormalization()(layer)

    layer = MaxPooling1D(pool_size=paramdims[2])(layer)

    layer = Conv1D(paramdims[3], kernel_size=(paramdims[4]),
            activation = 'relu')(layer)
    layer = BatchNormalization()(layer)

    layer = MaxPooling1D(pool_size=paramdims[5])(layer)
    layer = Dense(paramdims[6])(layer)
    layer = Dropout(0.5)(layer)

    output = GlobalAveragePooling1D()(layer)
    return input, output

@compatibilityCheck(['input_1'])
def model_gauss_conv_2l_glob(data, paramdims):
    '''
    Conv1D:
        {} x {}, relu
        {} pooling
        {} x {}
        GlobPool
    '''
    input = Input(shape=data['input_1'].shape, name='input_1')
    layer = Conv1D(paramdims[0], kernel_size=(paramdims[1]),
            activation = 'relu')(input)

    layer = GaussianNoise(0.1)(layer)

    layer = MaxPooling1D(pool_size=paramdims[2])(layer)
    layer = Conv1D(paramdims[3], kernel_size=(paramdims[4]),
            activation = 'relu')(layer)
    output = GlobalAveragePooling1D()(layer)
    return input, output

@compatibilityCheck(['input_1'])
def model_gauss_conv_lstm(data, paramdims):
    '''
    Conv1D:
        {} x {}, relu
        {} pooling
        {} x
        GlobPool
    '''
    input = Input(shape=data['input_1'].shape, name='input_1')
    layer = Conv1D(paramdims[0], kernel_size=(paramdims[1]),
            activation = 'relu')(input)

    layer = GaussianNoise(0.1)(layer)

    layer = MaxPooling1D(pool_size=paramdims[2])(layer)
    layer = LSTM(paramdims[3], return_sequences=True)(layer)
    output = GlobalAveragePooling1D()(layer)
    return input, output

@compatibilityCheck(['input_1'])
def model_lstm(data, paramdims):
    '''
    LSTM:
        {}, relu
        GlobPool
    '''
    input = Input(shape=data['input_1'].shape, name='input_1')
    layer = LSTM(paramdims[0], return_sequences = True)(input)
    #layer = Conv1D(paramdims[0], kernel_size=(paramdims[1]), activation = 'relu')(input)
    output = GlobalAveragePooling1D()(layer)

    return input, output

@compatibilityCheck(['input_1'])
def model_ff(data, paramdims):
    '''
    FFNN:
        {} x {}, relu
        GlobPool
    '''
    input = Input(shape=data['input_1'].shape, name='input_1')
    #layer = Dense(data['input_1'].shape[0]*data['input_1'].shape[1], activation='relu')(input)
    #layer = Dropout(0.4)(layer)
    layer = Dense(paramdims[0])(input)
    #layer = Dropout(0.4)(layer)
    layer = Dense(paramdims[1])(layer)
    layer = Dense(2)(layer)

    #output = Activation('sigmoid')
    #layer = Dropout(0.4)(layer)
    #output = GlobalAveragePooling1D()(layer)
    output = Flatten()(layer)

    return input, output

@compatibilityCheck(['input_1'])
def model_logreg(data, paramdims):
    '''
    logistig regresscion:
        {}
        Flatten
    '''
    input = Input(shape=data['input_1'].shape, name='input_1')
    output = Flatten()(input)

    return input, output

modeldefs = { 'conv_30_100': (model_conv_glob, (30,100)),
                'conv_30_200': (model_conv_glob, (30,200)),
                'conv_30_300': (model_conv_glob, (30,300)),
                'conv_10_200': (model_conv_glob, (10,200)),
                'conv_50_200': (model_conv_glob, (50,200)),
                'poolconv_10_50_20': (model_pool_conv_glob, (10,50,20)),
                'poolconv_10_30_20': (model_pool_conv_glob, (10,30,20)),
                'poolconv_10_30_30': (model_pool_conv_glob, (10,30,30)),
                'conv2l_30_300_10_20_30': (model_conv_2l_glob, (30,300,10,20,30)),
                'conv2l_50_300_10_20_30': (model_conv_2l_glob, (50,300,10,20,30)),
                'conv2l_50_300_10_40_30': (model_conv_2l_glob, (50,300,10,40,30)),
                'conv2l_30_300_10_40_30': (model_conv_2l_glob, (30,300,10,40,30)),
                'conv3l_50_300_10_20_30_10_10': (model_conv_3l_glob, (50,300,10,20,30, 10,10)),
                'conv3l_30_300_10_20_30_10_10': (model_conv_3l_glob, (30,300,10,20,30, 10,10)),
                'conv3l_30_300_10_40_30_10_10': (model_conv_3l_glob, (30,300,10,40,30, 10,10)),
                'conv3l_50_300_10_40_30_10_10': (model_conv_3l_glob, (50,300,10,40,30, 10,10)),
                'conv3l_70_300_10_20_30_10_10': (model_conv_3l_glob, (70,300,10,20,30, 10,10)),
                'conv3l_50_300_10_20_30_10_10': (model_conv_3l_glob, (50,300,10,20,30, 10,10)),
                'cgc_30_300_10_20_30': (model_gauss_conv_2l_glob, (30,300,10,20,30)),
                #'convlstm_30_300_10_20': (model_gauss_conv_lstm, (30,300,10,20)),
                #'ffnn_64' : (model_ff, (64,32)),
}
