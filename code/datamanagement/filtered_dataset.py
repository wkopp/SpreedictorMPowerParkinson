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
import re

datadir = os.getenv('PARKINSON_DREAM_DATA')

def convert(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

class FilteredData(NumpyDataset):

    def __init__(self, reload_ = False):

        res = re.match('Filter(Band|High|Low)Pass(.*?)(UserAccel|RotationRate)(Outbound|Rest|Return)$', self.__class__.__name__)

        if not res:
            print('invalid class name', self.__class__.__name__)
            return None

        self.filter_type = res.group(1).lower()
        self.data_transform = res.group(2).lower()
        data_type = res.group(3).lower()
        variant = res.group(4).lower()

        var_name = {'useraccel': 'userAcceleration', 'rotationrate': 'rotationRate'}[data_type]

        self.npcachefile = os.path.join(datadir,
            "filter_{}p_{}_{}_{}.pkl".format(self.filter_type[0], self.data_transform, data_type, variant))

        print(self.npcachefile)

        self.columns = list(itertools.product([var_name], \
                    ["x","y","z"]))

        NumpyDataset.__init__(self, "deviceMotion", variant, reload_)

    def createFilterBandPass(self, sample_rate = 100.0):
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
        cutoff_low_hz = 0.5
        cutoff_high_hz = 10.0

        # Use firwin with a Kaiser window to create a lowpass FIR filter.
        taps = signal.firwin(N, [cutoff_low_hz / nyq_rate, cutoff_high_hz / nyq_rate], window=('kaiser', beta),
                      pass_zero=False)

        ## The phase delay of the filtered signal.
        #delay = 0.5 * (N - 1) / sample_rate

        return (taps, N)

    def createFilterHighPass(self, sample_rate = 100.0):
        nyq_rate = sample_rate / 2.0
        width = 2.0 / nyq_rate
        ripple_db = 60.0
        N, beta = signal.kaiserord(ripple_db, width)
        #N = N + 1
        cutoff_hz = 0.5
        taps = signal.firwin(N, cutoff_hz / nyq_rate, window=('kaiser', beta), pass_zero=False)

        return (taps, N)

    def createFilterLowPass(self, sample_rate = 100.0):
        nyq_rate = sample_rate / 2.0
        width = 2.0 / nyq_rate
        ripple_db = 60.0

        N, beta = signal.kaiserord(ripple_db, width)

        cutoff_hz = 10.0
        taps = signal.firwin(N, cutoff_hz / nyq_rate, window=('kaiser', beta))

        return (taps, N)

    def createFilter(self, filter_type, sample_rate):
        if filter_type == 'band':
            return self.createFilterBandPass()
        elif filter_type == 'high':
            return self.createFilterHighPass()
        elif filter_type == 'low':
            return self.createFilterLowPass()
        else:
            print('invalid filter type')
            return None

    def getValues(self, df):
        M = df[[ "_".join(el) for \
            el in self.columns]].values

        if self.data_transform == 'worldcoord':
            #print('worldcoord')
            col = list(itertools.product(['attitude'], ["x", "y", "z", 'w']))
            Q = df[["_".join(el) for el in col]].values

            M = np.asarray([quaternion.quat_mult(tuple(Q[x, :]), M[x, :]) \
                               for x in range(M.shape[0])])
        elif self.data_transform == 'raw':
            #print('raw')
            pass
        else:
            raise Exception('unknown data transform:', self.data_transform)

        (taps, N) = self.createFilter(self.filter_type, sample_rate = 100.0)

        # Use lfilter to filter x with the FIR filter.
        filtered_x = signal.lfilter(taps, 1.0, M, axis=0)

        # Return just "good" part of the filtered signal.  The first N-1
        # samples are "corrupted" by the initial conditions.
        return filtered_x[(N - 1):,:]


class FilterBandPassWorldCoordUserAccelOutbound(FilteredData):
    pass

class FilterBandPassWorldCoordUserAccelReturn(FilteredData):
    pass

class FilterBandPassWorldCoordUserAccelRest(FilteredData):
    pass

class FilterHighPassWorldCoordUserAccelOutbound(FilteredData):
    pass

class FilterHighPassWorldCoordUserAccelReturn(FilteredData):
    pass

class FilterHighPassWorldCoordUserAccelRest(FilteredData):
    pass

class FilterLowPassWorldCoordUserAccelOutbound(FilteredData):
    pass

class FilterLowPassWorldCoordUserAccelReturn(FilteredData):
    pass

class FilterLowPassWorldCoordUserAccelRest(FilteredData):
    pass


class FilterBandPassRawUserAccelOutbound(FilteredData):
    pass

class FilterBandPassRawUserAccelReturn(FilteredData):
    pass

class FilterBandPassRawUserAccelRest(FilteredData):
    pass

class FilterHighPassRawUserAccelOutbound(FilteredData):
    pass

class FilterHighPassRawUserAccelReturn(FilteredData):
    pass

class FilterHighPassRawUserAccelRest(FilteredData):
    pass

class FilterLowPassRawUserAccelOutbound(FilteredData):
    pass

class FilterLowPassRawUserAccelReturn(FilteredData):
    pass

class FilterLowPassRawUserAccelRest(FilteredData):
    pass


class FilterBandPassRawRotationRateOutbound(FilteredData):
    pass

class FilterBandPassRawRotationRateReturn(FilteredData):
    pass

class FilterBandPassRawRotationRateRest(FilteredData):
    pass

class FilterHighPassRawRotationRateOutbound(FilteredData):
    pass

class FilterHighPassRawRotationRateReturn(FilteredData):
    pass

class FilterHighPassRawRotationRateRest(FilteredData):
    pass

class FilterLowPassRawRotationRateOutbound(FilteredData):
    pass

class FilterLowPassRawRotationRateReturn(FilteredData):
    pass

class FilterLowPassRawRotationRateRest(FilteredData):
    pass
