import os
import joblib
import itertools
import synapseclient
import pandas as pd
import numpy as np
from .cachedwalkingactivity import CachedWalkingActivity as WalkingActivity


class NumpyDataset(object):
    def __init__(self, modality, variant, reload_ = False):
        self.reload = reload_
        self.load(modality, variant)

    def load(self, modality, variant):
        if not os.path.exists(self.npcachefile) or self.reload:
            activity = WalkingActivity()
            nrows = activity.getCommonDescriptor().shape[0]
            data = np.zeros((nrows, 2000, len(self.columns)), dtype="float32")
            keepind = np.ones((nrows), dtype=bool)
            
            for idx in range(nrows):
                df = activity.getEntryByIndex(idx, modality, variant)
                
                if df.empty:
                    keepind[idx] = False
                    continue

                if df.shape[0]>2000:
                    df = df.iloc[:2000]

                df =  self.getValues(df)
                data[idx, :df.shape[0], :] = df

            data = data[keepind]
            
            
            labels = activity.getCommonDescriptor()["professional-diagnosis"].apply(
                lambda x: 1 if x==True else 0)
            labels = labels[keepind]
            
            joblib.dump((data, labels, keepind), self.npcachefile)

        self.data, self.labels, self.keepind = joblib.load(self.npcachefile)

    def getData(self, transform = False):
        if transform:
            return self.transformData(self.data)
        else:
            return self.data

    def getHealthCode(self):
        activity = WalkingActivity()
        annotation = activity.getCommonDescriptor().iloc[self.keepind]
        return annotation["healthCode"].values
        
    def transformData(self, data):
        return data

    def getLabels(self):
        return self.labels.values

    def getShape(self):
        return self.data.shape[1:]

    def getNdatapoints(self):
        return self.data.shape[0]
