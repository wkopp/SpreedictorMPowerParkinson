import os
import joblib
import itertools
import synapseclient
import pandas as pd
import numpy as np
from numpydataset import NumpyDataset

datadir = os.getenv('PARKINSON_DREAM_DATA')

class RawUserAccel(NumpyDataset):
    def __init__(self, variant, limit = None, reload_ = False):
        self.npcachefile = os.path.join(datadir, 
                "rawuseraccel_{}.pkl".format(variant))

        self.columns = list(itertools.product(["userAcceleration"], \
                    ["x","y","z"]))
        NumpyDataset.__init__(self, variant, limit, reload_)

    def getValues(self, df):
        return df[[ "_".join(el) for \
            el in self.columns]].values

class RawUserAccelOutbound(RawUserAccel):
    '''
    Raw userAcceleration data for outbound walk
    '''
    def __init__(self, limit = None):
        RawUserAccel.__init__(self, "outbound", limit)

class RawUserAccelRest(RawUserAccel):
    '''
    Raw userAcceleration data for rest phase
    '''
    def __init__(self, limit = None):
        RawUserAccel.__init__(self, "rest", limit)

class RawUserAccelReturn(RawUserAccel):
    '''
    Raw userAcceleration data for return walk
    '''
    def __init__(self, limit = None):
        RawUserAccel.__init__(self, "return", limit)
