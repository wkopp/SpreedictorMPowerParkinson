import os
import joblib
import itertools
import synapseclient
import pandas as pd
import numpy as np
from numpydataset import NumpyDataset

from utils import batchRandomRotation

datadir = os.getenv('PARKINSON_DREAM_DATA')

class RawUserAccel(NumpyDataset):
    
    def __init__(self, variant, reload_ = False):
        self.npcachefile = os.path.join(datadir, 
                "rawuseraccel_{}.pkl".format(variant))

        self.columns = list(itertools.product(["userAcceleration"], \
                    ["x","y","z"]))
        NumpyDataset.__init__(self, "deviceMotion", variant, reload_)

    def getValues(self, df):
        M = df[[ "_".join(el) for \
            el in self.columns]].values
        shift = M.mean(axis=0)
        M -= shift
        return M
    
    def transformData(self, data):
        return batchRandomRotation(data)

class RawUserAccelOutbound(RawUserAccel):
    '''
    Raw userAcceleration data for outbound walk
    '''
    def __init__(self, reload_ = False):
        RawUserAccel.__init__(self, "outbound", reload_)

class RawUserAccelRest(RawUserAccel):
    '''
    Raw userAcceleration data for rest phase
    '''
    def __init__(self, reload_ = False):
        RawUserAccel.__init__(self, "rest", reload_)

class RawUserAccelReturn(RawUserAccel):
    '''
    Raw userAcceleration data for return walk
    '''
    def __init__(self, reload_ = False):
        RawUserAccel.__init__(self, "return", reload_)
