from keras.layers import Input
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import GlobalAveragePooling1D
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
    input = Input(shape=data['input_1'].getShape(), name='input_1')
    layer = Conv1D(paramdims[0], kernel_size=(paramdims[1]), activation = 'relu')(input)
    output = GlobalAveragePooling1D()(layer)
    return input, output

@compatibilityCheck(['input_1'])
def model_lstm(data, paramdims):
    '''
    LSTM: 
        {}, relu
        GlobPool
    '''
    input = Input(shape=data['input_1'].getShape(), name='input_1')
    layer = LSTM(paramdims[0], return_sequences = True)(input)
    #layer = Conv1D(paramdims[0], kernel_size=(paramdims[1]), activation = 'relu')(input)
    output = GlobalAveragePooling1D()(layer)

    return input, output

modeldefs = { 'conv_30_100': (model_conv_glob, (30,100)),
                'conv_30_200': (model_conv_glob, (30,200)),
                'conv_30_300': (model_conv_glob, (30,300)),
                'conv_30_400': (model_conv_glob, (30,400)),
                'conv_30_500': (model_conv_glob, (30,500)),
                'conv_30_50': (model_conv_glob, (30,50)),
                #'lstm_16': (model_lstm, (16,)),
                #'lstm_32': (model_lstm, (32,)),
}
