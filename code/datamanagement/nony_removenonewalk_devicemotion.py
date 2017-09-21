import os
import joblib
import itertools
import synapseclient
import pandas as pd
import numpy as np
from .numpydataset import NumpyDataset

datadir = os.getenv('PARKINSON_DREAM_DATA')

class NonYRNWDeviceMotion(NumpyDataset):
    def __init__(self, variant, reload_ = False):
        self.npcachefile = os.path.join(datadir,
                "nyrnwdevicemotion_{}.pkl".format(variant))

        self.columns = list(itertools.product(["userAcceleration",
            "gravity", "rotationRate"], ["x","y","z"]))
        NumpyDataset.__init__(self, "deviceMotion", variant, reload_)

    def getValues(self, df):

        # remove non-walking data
        df["score"] = df["userAcceleration_x"]**2 + \
            df["userAcceleration_y"]**2 + df["userAcceleration_z"]**2

        # take the threshold to be the extreme outliers
        threshold = (df.score.quantile(.75)- \
                df.score.quantile(.25))*2 + df.score.quantile(.75)

        idx = np.where(df.score >= threshold)[0]

        if len(idx)>0:
            df = df.iloc[idx[0]:idx[-1]]

        # remove non-y gravity 1/-1
        df = df[(df.gravity_y>0.6) | (df.gravity_y<-0.6)]
        M = df[[ "_".join(el) for \
            el in self.columns]].values
        shift = M.mean(axis=0)
        M -= shift
        return M

class NonYRNWDeviceMotionOutbound(NonYRNWDeviceMotion):
    '''
    Filtered Non-Y up or down device motion data for outbound walk
    '''
    def __init__(self, reload_ = False):
        NonYRNWDeviceMotion.__init__(self, "outbound", reload_)

class NonYRNWDeviceMotionRest(NonYRNWDeviceMotion):
    '''
    Filtered Non-Y up or down device motion data for rest phase
    '''
    def __init__(self, reload_ = False):
        NonYRNWDeviceMotion.__init__(self, "rest", reload_)

class NonYRNWDeviceMotionReturn(NonYRNWDeviceMotion):
    '''
    Filtered Non-Y up or down device motion data for return walk
    '''
    def __init__(self, reload_ = False):
        NonYRNWDeviceMotion.__init__(self, "return", reload_)
