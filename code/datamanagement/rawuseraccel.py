import os
import joblib
import itertools
import synapseclient
import pandas as pd
import numpy as np
from .numpydataset import NumpyDataset


datadir = os.getenv('PARKINSON_DREAM_DATA')

class RawUserAccel(NumpyDataset):

    def __init__(self, variant, reload_ = False, training = True, rmnan = True):
        self.npcachefile = os.path.join(datadir,
                "rawuseraccel_{}.pkl".format(variant))

        self.columns = list(itertools.product(["userAcceleration"], \
                    ["x","y","z"]))
        NumpyDataset.__init__(self, "deviceMotion", variant, reload_, training, rmnan)

    def getValues(self, df):
        M = df[[ "_".join(el) for \
            el in self.columns]].values.astype("float32")
        shift = M.mean(axis=0)
        M -= shift
        return M

#    def transformData(self, data):
#        return batchRandomRotation(data)

class RawUserAccelOutbound(RawUserAccel):
    '''
    Raw userAcceleration data for outbound walk
    '''
    def __init__(self, reload_ = False, training = True, rmnan = True):
        RawUserAccel.__init__(self, "outbound", reload_, training, rmnan)

class RawUserAccelRest(RawUserAccel):
    '''
    Raw userAcceleration data for rest phase
    '''
    def __init__(self, reload_ = False, training = True, rmnan = True):
        RawUserAccel.__init__(self, "rest", reload_, training, rmnan)

class RawUserAccelReturn(RawUserAccel):
    '''
    Raw userAcceleration data for return walk
    '''
    def __init__(self, reload_ = False, training = True, rmnan = True):
        RawUserAccel.__init__(self, "return", reload_, training, rmnan)
