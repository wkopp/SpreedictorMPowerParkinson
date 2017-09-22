import synapseclient
import pandas as pd
import numpy as np
import os
import joblib
from .demographics import Demographics
import itertools
import datamanagement.quaternion as quaternion
from .walkingactivity import WalkingActivity

datadir = os.getenv('PARKINSON_DREAM_DATA')

class WalkingActivityTest(WalkingActivity):

    def __init__(self, limit = None, download_jsons = True):

        self.synapselocation = 'syn10733842'
        self.keepcolumns = [
            'deviceMotion_walking_outbound.json.items',
            'deviceMotion_walking_return.json.items',
            'deviceMotion_walking_rest.json.items']

        WalkingActivity.__init__(self, limit, download_jsons)

if __name__ == '__main__':
    wa = WalkingActivityTest(limit = None, download_jsons = True)
    #ts = wa.getEntryByIndex(0, modality='pedometer', variant='outbound')
    #wa.convertUserAccelerationToWorldFrame(ts)
    #print wa.modality_variants
    #wa.extractTimeseriesLengths(limit=1000)
