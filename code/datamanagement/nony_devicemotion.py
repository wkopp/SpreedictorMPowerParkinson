import os
import joblib
import itertools
import synapseclient
import pandas as pd
import numpy as np
from numpydataset import NumpyDataset

datadir = os.getenv('PARKINSON_DREAM_DATA')

class NonYDeviceMotion(NumpyDataset):
    def __init__(self, variant, reload_ = False):
        self.npcachefile = os.path.join(datadir, 
                "nonydevicemotion_{}.pkl".format(variant))

        self.columns = list(itertools.product(["userAcceleration",
            "gravity", "rotationRate"], ["x","y","z"]))
        NumpyDataset.__init__(self, "deviceMotion", variant, reload_)

    def getValues(self, df):
        df = df[(df.gravity_y>0.8) | (df.gravity_y<-0.8)]
        M = df[[ "_".join(el) for \
            el in self.columns]].values
        shift = M.mean(axis=0)
        M -= shift
        return M

class NonYDeviceMotionOutbound(NonYDeviceMotion):
    '''
    Filtered Non-Y up or down device motion data for outbound walk
    '''
    def __init__(self, reload_ = False):
        NonYDeviceMotion.__init__(self, "outbound", reload_)

class NonYDeviceMotionRest(NonYDeviceMotion):
    '''
    Filtered Non-Y up or down device motion data for rest phase
    '''
    def __init__(self, reload_ = False):
        NonYDeviceMotion.__init__(self, "rest", reload_)

class NonYDeviceMotionReturn(NonYDeviceMotion):
    '''
    Filtered Non-Y up or down device motion data for return walk
    '''
    def __init__(self, reload_ = False):
        NonYDeviceMotion.__init__(self, "return", reload_)
