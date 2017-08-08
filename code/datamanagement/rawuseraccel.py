import os
import joblib
import itertools
import synapseclient
import pandas as pd
import numpy as np
from numpydataset import NumpyDataset

datadir = os.getenv('PARKINSON_DREAM_DATA')
def rotX(angle):
    return np.array(
            [[1,        0,                  0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]])
def rotY(angle):
    return np.array(
            [[np.cos(angle), 0, np.sin(angle)],
            [0,              1,         0],
            [-np.sin(angle), 0, np.cos(alpha)]])
def rotZ(angle):
    return np.array(
            [[np.cos(angle), -np.sin(angle),    0],
            [np.sin(angle), np.cos(angle),      0],
            [0,                 0,              1]])

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
