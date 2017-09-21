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
class WorldCoordUserAccel(NumpyDataset):

    def __init__(self, variant, reload_ = False):
        self.npcachefile = os.path.join(datadir,
                "worldcouseraccel_{}.pkl".format(variant))

        self.columns = list(itertools.product(["userAcceleration"], \
                    ["x","y","z"]))
        NumpyDataset.__init__(self, "deviceMotion", variant, reload_)

    def getValues(self, df):
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

class WorldCoordUserAccelOutbound(WorldCoordUserAccel):
    '''
    WorldCoord userAcceleration data for outbound walk
    '''
    def __init__(self, reload_ = False):
        WorldCoordUserAccel.__init__(self, "outbound", reload_)

class WorldCoordUserAccelRest(WorldCoordUserAccel):
    '''
    WorldCoord userAcceleration data for rest phase
    '''
    def __init__(self, reload_ = False):
        WorldCoordUserAccel.__init__(self, "rest", reload_)

class WorldCoordUserAccelReturn(WorldCoordUserAccel):
    '''
    WorldCoord userAcceleration data for return walk
    '''
    def __init__(self, reload_ = False):
        WorldCoordUserAccel.__init__(self, "return", reload_)
