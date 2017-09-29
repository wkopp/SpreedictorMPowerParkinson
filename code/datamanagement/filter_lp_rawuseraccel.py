import os
import joblib
import itertools
import synapseclient
import pandas as pd
import numpy as np
from .numpydataset import NumpyDataset
import datamanagement.quaternion as quaternion
import scipy.signal as signal

from .utils import batchRandomRotation

datadir = os.getenv('PARKINSON_DREAM_DATA')
class FilterLowPassRawUserAccel(NumpyDataset):

    def __init__(self, variant, reload_ = False, training = True):
        self.npcachefile = os.path.join(datadir,
                "filter_lp_rawuseraccel_{}.pkl".format(variant))

        self.columns = list(itertools.product(["userAcceleration"], \
                    ["x","y","z"]))
        NumpyDataset.__init__(self, "deviceMotion", variant, reload_, training)

    def createFilter(self, sample_rate = 100.0):
        # The Nyquist rate of the signal.
        nyq_rate = sample_rate / 2.0

        # The desired width of the transition from pass to stop,
        # relative to the Nyquist rate.
        width = 2.0 / nyq_rate

        # The desired attenuation in the stop band, in dB.
        ripple_db = 60.0

        # Compute the order and Kaiser parameter for the FIR filter.
        N, beta = signal.kaiserord(ripple_db, width)

        # The cutoff frequency of the filter.
        cutoff_hz = 10.0

        # Use firwin with a Kaiser window to create a lowpass FIR filter.
        taps = signal.firwin(N, cutoff_hz / nyq_rate, window=('kaiser', beta))

        ## The phase delay of the filtered signal.
        #delay = 0.5 * (N - 1) / sample_rate

        return (taps, N)



    def getValues(self, df):
        M = df[[ "_".join(el) for \
            el in self.columns]].values.astype("float32")

        (taps, N) = self.createFilter(sample_rate = 100.0)

        # Use lfilter to filter x with the FIR filter.
        filtered_x = signal.lfilter(taps, 1.0, M, axis=0)

        # Plot just the "good" part of the filtered signal.  The first N-1
        # samples are "corrupted" by the initial conditions.
        return filtered_x[(N - 1):,:]


class FilterLowPassRawUserAccelOutbound(FilterLowPassRawUserAccel):
    '''
    WorldCoord userAcceleration data for outbound walk
    '''
    def __init__(self, reload_ = False, training = True):
        FilterLowPassRawUserAccel.__init__(self, "outbound", reload_, training)

class FilterLowPassRawUserAccelRest(FilterLowPassRawUserAccel):
    '''
    WorldCoord userAcceleration data for rest phase
    '''
    def __init__(self, reload_ = False, training = True):
        FilterLowPassRawUserAccel.__init__(self, "rest", reload_, training)

class FilterLowPassRawUserAccelReturn(FilterLowPassRawUserAccel):
    '''
    WorldCoord userAcceleration data for return walk
    '''
    def __init__(self, reload_ = False, training = True):
        FilterLowPassRawUserAccel.__init__(self, "return", reload_, training)
