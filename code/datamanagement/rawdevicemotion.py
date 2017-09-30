import os
import joblib
import itertools
import synapseclient
import pandas as pd
import numpy as np
from .numpydataset import NumpyDataset

datadir = os.getenv('PARKINSON_DREAM_DATA')

class RawDeviceMotion(NumpyDataset):
    def __init__(self, variant, reload_ = False, training = True):
        self.npcachefile = os.path.join(datadir,
                "rawdevicemotion_{}.pkl".format(variant))

        self.columns = list(itertools.product(["userAcceleration",
            "gravity", "rotationRate"], ["x","y","z"]))

        NumpyDataset.__init__(self, "deviceMotion", variant, reload_, training)

    def getValues(self, df):
        M = df[[ "_".join(el) for \
            el in self.columns]].values.astype("float32")
        shift = M.mean(axis=0)
        M -= shift
        return M
        #return df[[ "_".join(el) for \
        #    el in  self.columns ]].values

class RawDeviceMotionOutbound(RawDeviceMotion):
    '''
    Raw device motion data for outbound walk
    '''
    def __init__(self, reload_ = False, training = True):
        RawDeviceMotion.__init__(self, "outbound", reload_, training)

class RawDeviceMotionRest(RawDeviceMotion):
    '''
    Raw device motion data for rest phase
    '''
    def __init__(self, reload_ = False, training = True):
        RawDeviceMotion.__init__(self, "rest", reload_, training)

class RawDeviceMotionReturn(RawDeviceMotion):
    '''
    Raw device motion data for return walk
    '''
    def __init__(self, reload_ = False, training = True):
        RawDeviceMotion.__init__(self, "return", reload_, training)
