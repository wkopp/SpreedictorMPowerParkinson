import os
import joblib
import itertools
import synapseclient
import pandas as pd
import numpy as np
from .numpydataset import NumpyDataset

datadir = os.getenv('PARKINSON_DREAM_DATA')

class MtSpectrum_SvdUserAccel(NumpyDataset):
    def __init__(self, variant, reload_ = False, training = True):
        self.npcachefile = os.path.join(datadir,
                "mtspectrasvduseraccel_{}.pkl".format(variant))

        self.columns = list(itertools.product(["spec_ua"], \
                    ["x","y","z"]))
        NumpyDataset.__init__(self, "deviceMotion", variant, reload_, training)
        #self.data = self.data[:, 0:1, :]

    def getValues(self, df):
        print('error: getValue() not implemented for mtspectrum*')
        return np.nan

#    def transformData(self, data):
#        return batchRandomRotation(data)

class MtSpectrum_SvdUserAccelOutbound(MtSpectrum_SvdUserAccel):
    '''
    SVD userAcceleration data for outbound walk
    '''
    def __init__(self, reload_ = False, training = True):
        MtSpectrum_SvdUserAccel.__init__(self, "outbound", reload_, training)

class MtSpectrum_SvdUserAccelRest(MtSpectrum_SvdUserAccel):
    '''
    SVD userAcceleration data for rest phase
    '''
    def __init__(self, reload_ = False, training = True):
        MtSpectrum_SvdUserAccel.__init__(self, "rest", reload_, training)

class MtSpectrum_SvdUserAccelReturn(MtSpectrum_SvdUserAccel):
    '''
    SVD userAcceleration data for return walk
    '''
    def __init__(self, reload_ = False, training = True):
        MtSpectrum_SvdUserAccel.__init__(self, "return", reload_, training)
