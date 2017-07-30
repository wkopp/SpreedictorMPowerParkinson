import os
import joblib
import itertools
import synapseclient
import pandas as pd
import numpy as np
from devicemotion import DeviceMotion

datadir = os.getenv('PARKINSON_DREAM_DATA')

class RemoveNoneWalkDeviceMotion(object):
    def __init__(self, variant, limit = None):
        self.dm = DeviceMotion()
        self.npcachefile = os.path.join(datadir, 
                "removednonwalk_devicemotion_{}.pkl".format(variant))
        self.load(variant)

    def load(self, variant):
        if not os.path.exists(self.npcachefile):
            nrows = self.dm.commondescr.shape[0]
            data = np.zeros((nrows, 2000, 9), dtype="float32")
            for idx in range(nrows):
                df = self.dm.getEntries([idx], variant)
                
                # only retain timepoints with y>0.8 or y< -0.8
                df["score"] = df["userAcceleration_x"]**2 + \
                    df["userAcceleration_y"]**2 + df["userAcceleration_z"]**2

            # take the threshold to be the extreme outliers
                threshold = (df.score.quantile(.75)- \
                        df.score.quantile(.25))*2 + df.score.quantile(.75)

                idx = np.where(df.score >= threshold)[0]

                if len(idx)>0:
                    df = df.iloc[idx[0]:idx[-1]]
                
                #df = df[(df.gravity_y>0.8) | (df.gravity_y<-0.8)]
                
                if df.shape[0]>2000:
                    df = df.iloc[:2000]
                
                #elif dftrimmed.shape[0]<500:
                    #usedf = df
                data[idx, :df.shape[0], :] = df[[ "_".join(el) for \
                    el in itertools.product(["userAcceleration", 
                        "gravity", "rotationRate"], ["x","y","z"]) ]].values

            labels = self.dm.commondescr["professional-diagnosis"].apply(
                lambda x: 1 if x==True else 0)
            joblib.dump((data, labels), self.npcachefile)

        self.data, self.labels = joblib.load(self.npcachefile)
    def getData(self):
        return self.data

    def getLabels(self):
        return self.labels.values

    def getShape(self):
        return self.data.shape[1:]

class RemoveNoneWalkDeviceMotionOutbound(RemoveNoneWalkDeviceMotion):
    '''
    Filtered Non-Y up or down device motion data for outbound walk
    '''
    def __init__(self, limit = None):
        RemoveNoneWalkDeviceMotion.__init__(self, "outbound", limit)

class RemoveNoneWalkDeviceMotionRest(RemoveNoneWalkDeviceMotion):
    '''
    Filtered Non-Y up or down device motion data for rest phase
    '''
    def __init__(self, limit = None):
        RemoveNoneWalkDeviceMotion.__init__(self, "rest", limit)

class RemoveNoneWalkDeviceMotionReturn(RemoveNoneWalkDeviceMotion):
    '''
    Filtered Non-Y up or down device motion data for return walk
    '''
    def __init__(self, limit = None):
        RemoveNoneWalkDeviceMotion.__init__(self, "return", limit)
