import os
import joblib
import itertools
import synapseclient
import pandas as pd
import numpy as np
from numpydataset import NumpyDataset

datadir = os.getenv('PARKINSON_DREAM_DATA')

class RemoveNoneWalkDeviceMotion(NumpyDataset):
    def __init__(self, variant, reload_ = False):
        self.npcachefile = os.path.join(datadir, 
                "removednonwalk_devicemotion_{}.pkl".format(variant))

        self.columns = list(itertools.product(["userAcceleration",
            "gravity", "rotationRate"], ["x","y","z"]))
        NumpyDataset.__init__(self, "deviceMotion", variant, reload_)

    def getValues(self, df):
        # only retain timepoints with y>0.8 or y< -0.8
        df["score"] = df["userAcceleration_x"]**2 + \
            df["userAcceleration_y"]**2 + df["userAcceleration_z"]**2

        # take the threshold to be the extreme outliers
        threshold = (df.score.quantile(.75)- \
                df.score.quantile(.25))*2 + df.score.quantile(.75)

        idx = np.where(df.score >= threshold)[0]

        if len(idx)>0:
            df = df.iloc[idx[0]:idx[-1]]

        return df[[ "_".join(el) for \
            el in self.columns ]].values
        
class RemoveNoneWalkDeviceMotionOutbound(RemoveNoneWalkDeviceMotion):
    '''
    Filtered Non-Y up or down device motion data for outbound walk
    '''
    def __init__(self):
        RemoveNoneWalkDeviceMotion.__init__(self, "outbound")

class RemoveNoneWalkDeviceMotionRest(RemoveNoneWalkDeviceMotion):
    '''
    Filtered Non-Y up or down device motion data for rest phase
    '''
    def __init__(self):
        RemoveNoneWalkDeviceMotion.__init__(self, "rest")

class RemoveNoneWalkDeviceMotionReturn(RemoveNoneWalkDeviceMotion):
    '''
    Filtered Non-Y up or down device motion data for return walk
    '''
    def __init__(self):
        RemoveNoneWalkDeviceMotion.__init__(self, "return")
