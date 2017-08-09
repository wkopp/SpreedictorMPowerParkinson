import synapseclient
import pandas as pd
import numpy as np
import os
import joblib
from demographics import Demographics
from walkingactivity import WalkingActivity
import itertools
import quaternion

datadir = os.getenv('PARKINSON_DREAM_DATA')

class CachedWalkingActivity(WalkingActivity):

    def __init__(self, limit = None, download_jsons = True, reload_ = False):

        self.downloadpath = datadir + "download/"

        if limit:
            self.cachepath = self.downloadpath + \
                             "json_file_map_{:d}.pkl".format(limit)
        else:
            self.cachepath = self.downloadpath + "json_file_map.pkl"

        if not os.path.exists(self.downloadpath) or reload_ or \
                not os.path.exists(self.cachepath):

            if not os.path.exists(self.downloadpath):
                os.mkdir(self.downloadpath)
            WalkingActivity.__init__(self, limit, download_jsons)
            joblib.dump((self.commondescr, self.file_map), self.cachepath)

        else:
            self.load()

    def load(self):
        self.commondescr, self.file_map = joblib.load(self.cachepath)

if __name__ == '__main__':
    wa = CachedWalkingActivity(limit = None, download_jsons = True)
    #ts = wa.getEntryByIndex(0, modality='pedometer', variant='outbound')
    #wa.convertUserAccelerationToWorldFrame(ts)
    #print wa.modality_variants
    print "extracting TS"
    wa.extractTimeseriesLengths(limit=None, reload_=True)

