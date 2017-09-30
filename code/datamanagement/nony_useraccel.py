import os
import joblib
import itertools
import synapseclient
import pandas as pd
import numpy as np
from .numpydataset import NumpyDataset

from .utils import batchRandomRotation

datadir = os.getenv('PARKINSON_DREAM_DATA')

class NonYUserAccel(NumpyDataset):

    def __init__(self, variant, reload_ = False, training = True):
        self.npcachefile = os.path.join(datadir,
                "nonyuseraccel_{}.pkl".format(variant))

        self.columns = list(itertools.product(["userAcceleration"],
            ["x","y","z"]))
        NumpyDataset.__init__(self, "deviceMotion", variant, reload_, training)

    def getValues(self, df):
        df = df[(df.gravity_y>0.6) | (df.gravity_y<-0.6)]
        M = df[[ "userAcceleration_x",
            "userAcceleration_y", "userAcceleration_z"]].values.astype("float32")
        shift = M.mean(axis=0)
        M -= shift
        return M


class NonYUserAccelOutbound(NonYUserAccel):
    '''
    NonY userAcceleration data for outbound walk
    '''
    def __init__(self, reload_ = False, training = True):
        NonYUserAccel.__init__(self, "outbound", reload_, training)

class NonYUserAccelRest(NonYUserAccel):
    '''
    NonY userAcceleration data for rest phase
    '''
    def __init__(self, reload_ = False, training = True):
        NonYUserAccel.__init__(self, "rest", reload_, training)

class NonYUserAccelReturn(NonYUserAccel):
    '''
    NonY userAcceleration data for return walk
    '''
    def __init__(self, reload_ = False, training = True):
        NonYUserAccel.__init__(self, "return", reload_, training)
