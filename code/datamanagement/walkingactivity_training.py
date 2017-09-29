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

class WalkingActivityTraining(WalkingActivity):

    def __init__(self, limit = None, download_jsons = True):

        self.synapselocation = 'syn10146553'
        self.keepcolumns = [
            'deviceMotion_walking_outbound.json.items',
            'deviceMotion_walking_return.json.items',
            'deviceMotion_walking_rest.json.items']

        WalkingActivity.__init__(self, limit, download_jsons)

    def download(self, limit, download_jsons = True):

        WalkingActivity.download(self, limit, download_jsons)

        dem = Demographics().getData()[["healthCode", "professional-diagnosis"]]

        self.commondescr = pd.merge(self.commondescr, dem, on = "healthCode")

