from __future__ import print_function, division
import os
import joblib
import itertools
import synapseclient
import pandas as pd
import numpy as np
from .cachedwalkingactivity_training import CachedWalkingActivityTraining as WalkingActivityTraining
from .cachedwalkingactivity_test import CachedWalkingActivityTest as WalkingActivityTest
from .cachedwalkingactivity_suppl import CachedWalkingActivitySuppl as WalkingActivitySuppl
import progressbar


from .utils import batchRandomRotation

class NumpyDataset(object):
    def __init__(self, modality, variant, reload_ = False, training = True, rmnan=True):
        self.reload = reload_
        self.training = training
        self.rmnan = rmnan
        self.load(modality, variant, training)

    def loadTraining(self, modality, variant):
        if not os.path.exists(self.npcachefile) or self.reload:
            activity = WalkingActivityTraining()
            nrows = activity.getCommonDescriptor().shape[0]
            data = np.zeros((nrows, 2000, len(self.columns)), dtype="float32")
            keepind = np.ones((nrows), dtype=bool)

            bar = progressbar.ProgressBar()

            for idx in bar(range(nrows)):
                #self.printStatusUpdate(idx, nrows)
                df = activity.getEntryByIndex(idx, modality, variant)

                if df.empty:
                    keepind[idx] = False
                    continue

                #if df.shape[0]>2000:
                #    df = df.iloc[:2000]

                df =  self.getValues(df)
                data[idx, :min(df.shape[0], 2000), :] = df[:min(df.shape[0], 2000), :]

            data = data[keepind]

            labels = activity.getCommonDescriptor()["professional-diagnosis"].apply(
                lambda x: 1 if x==True else 0)

            labels = labels[keepind]

            joblib.dump((data, labels, keepind), self.npcachefile)

        self.data, self.labels, self.keepind = joblib.load(self.npcachefile)

        # this is kind of a hack, because most cache files have
        # the missing values already removed.
        # so here we re-instantiate the full dataset that contains the zeros
        if self.rmnan == False:
            activity = WalkingActivityTraining()
            nrows = activity.getCommonDescriptor().shape[0]
            data = np.zeros((nrows, 2000, len(self.columns)), dtype="float32")
            data[self.keepind] = self.data
            self.data = data

            # also reload the labels
            labels = activity.getCommonDescriptor()["professional-diagnosis"].apply(
                lambda x: 1 if x==True else 0)
            self.labels = labels


    def loadTest(self, modality, variant):
        # prepend "test"
        self.npcachefile = self.npcachefile.split(".")[0]
        self.npcachefile = self.npcachefile + "_all.pkl"
        print("process {}".format(self.npcachefile))
        activities = [WalkingActivityTraining(),
                        WalkingActivityTest(), WalkingActivitySuppl()]

        recordIds = np.concatenate([a.commondescr.recordId.values for a in activities])
        self.recordIds = recordIds
        #    recordIds.append(activities[1].commondescr.recordId, ignore_index=True)
        #    recordIds.append(activities[2].commondescr.recordId, ignore_index=True)

        if not os.path.exists(self.npcachefile) or self.reload:

            nrows = [activity.getCommonDescriptor().shape[0] for activity in
                activities]


            data = np.zeros((np.sum(nrows), 2000, len(self.columns)),
                dtype="float32")

            offset = 0
            for act in activities:
                print("process activity")
                bar = progressbar.ProgressBar()
                for idx in bar(range(act.getCommonDescriptor().shape[0])):
                    df = act.getEntryByIndex(idx, modality, variant)

                    if df.empty:
                        continue

                    df =  self.getValues(df)

                    data[idx + offset, :min(df.shape[0], 2000), :] = df[:min(df.shape[0], 2000), :]
                offset += act.getCommonDescriptor().shape[0]

            labels = np.zeros((np.sum(nrows)))
            keepind = np.ones((np.sum(nrows)), dtype=bool)

            joblib.dump((data, labels, keepind, self.recordIds), self.npcachefile)

        self.data, labels, self.keepind, _ = joblib.load(self.npcachefile)
        self.labels = pd.Series(labels)

    def load(self, modality, variant, training):
        if training:
            self.loadTraining(modality, variant)
        else:
            self.loadTest(modality, variant)

    def getData(self, idx = None, transform = False):
        if type(idx) == type(None):
            data = self.data
        else:
            data = self.data[idx]

        if transform:
            return self.transformData(data)
        else:
            return data

    def transformDataNoise(self, data):
        return data + np.random.normal(scale=np.sqrt(0.1), size=data.shape)

    def transformDataRotate(self, data):
        return batchRandomRotation(data)

    def transformDataFlipSign(self, data):
        for t in range(data.shape[0]):
            data[t] = np.matmul(data[t], np.diag(np.random.choice([1,-1], 3)))

        return data

    def transformDataFlipRotate(self, data):
        data = self.transformDataRotate(data)
        data = self.transformDataFlipSign(data)

        return data

    @property
    def healthCode(self):
        if self.training:
            activity = WalkingActivityTraining()
            if self.rmnan:
                annotation = activity.getCommonDescriptor().iloc[self.keepind]
            else:
                annotation = activity.getCommonDescriptor()
            return annotation["healthCode"].values
        else:
            activity = [WalkingActivityTraining(), WalkingActivityTest(), WalkingActivitySuppl()]
            annotation = np.concatenate([a.getCommonDescriptor()['healthCode'].values for a in activity])
            return annotation



    @property
    def labels(self):
        return self.lab.values

    @labels.setter
    def labels(self, lab):
        self.lab = lab

    @property
    def shape(self):
        return self.data.shape[1:]

    def __len__(self):
        return self.data.shape[0]
