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

            os.mkdir(self.downloadpath, ignore_errors = True)
            WalkingActivtiy.__init__(self, limit, download_jsons)
            joblib.dump((self.commondescr, self.file_map), self.cachepath)

        else:
            self.load()

if __name__ == '__main__':
    wa = CachedWalkingActivity(limit = 3000, download_jsons = True)
    #ts = wa.getEntryByIndex(0, modality='pedometer', variant='outbound')
    #wa.convertUserAccelerationToWorldFrame(ts)
    #print wa.modality_variants
    wa.extractTimeseriesLengths(limit=1000, reload_=True)

