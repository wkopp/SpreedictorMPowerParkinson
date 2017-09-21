import os
import joblib
import itertools
import synapseclient
import pandas as pd
import numpy as np
from .numpydataset import NumpyDataset
import datamanagement.quaternion as quaternion

from .utils import batchRandomRotation

datadir = os.getenv('PARKINSON_DREAM_DATA')
class WorldCoordNYRNWUserAccel(NumpyDataset):

    def __init__(self, variant, reload_ = False):
        self.npcachefile = os.path.join(datadir,
                "worldconyrnw_useraccel_{}.pkl".format(variant))

        self.columns = list(itertools.product(["userAcceleration"], \
                    ["x","y","z"]))
        NumpyDataset.__init__(self, "deviceMotion", variant, reload_)

    def getValues(self, df):
        df["score"] = df["userAcceleration_x"]**2 + \
            df["userAcceleration_y"]**2 + df["userAcceleration_z"]**2

        # take the threshold to be the extreme outliers
        threshold = (df.score.quantile(.75)- \
                df.score.quantile(.25))*2 + df.score.quantile(.75)

        idx = np.where(df.score >= threshold)[0]

        if len(idx)>0:
            df = df.iloc[idx[0]:idx[-1]]

        df = df[(df.gravity_y>0.6) | (df.gravity_y<-0.6)]

        M = df[[ "_".join(el) for \
            el in self.columns]].values
        col = list(itertools.product(['attitude'], ["x","y","z", 'w']))
        Q = df[[ "_".join(el) for el in col]].values

        #shift = M.mean(axis=0)
        #M -= shift

        return np.asarray([ quaternion.quat_mult(tuple(Q[x,:]), M[x,:]) \
                for x in range(M.shape[0])])

#    def transformData(self, data):
#        return batchRandomRotation(data)

class WorldCoordNYRNWUserAccelOutbound(WorldCoordNYRNWUserAccel):
    '''
    WorldCoordNYRNW userAcceleration data for outbound walk
    '''
    def __init__(self, reload_ = False):
        WorldCoordNYRNWUserAccel.__init__(self, "outbound", reload_)

class WorldCoordNYRNWUserAccelRest(WorldCoordNYRNWUserAccel):
    '''
    WorldCoordNYRNW userAcceleration data for rest phase
    '''
    def __init__(self, reload_ = False):
        WorldCoordNYRNWUserAccel.__init__(self, "rest", reload_)

class WorldCoordNYRNWUserAccelReturn(WorldCoordNYRNWUserAccel):
    '''
    WorldCoordNYRNW userAcceleration data for return walk
    '''
    def __init__(self, reload_ = False):
        WorldCoordNYRNWUserAccel.__init__(self, "return", reload_)
