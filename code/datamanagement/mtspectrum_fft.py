import os
import re
import itertools
import numpy as np
from .numpydataset import NumpyDataset

datadir = os.getenv('PARKINSON_DREAM_DATA')

class MtSpectrum_FFT(NumpyDataset):

    def __init__(self, reload_ = False):

        res = re.match('MtSpectrum_FFT_(Svd|Raw)(UserAccel|RotationRate)(Outbound|Rest|Return)$', self.__class__.__name__)

        if not res:
            print('invalid class name', self.__class__.__name__)
            return None

        data_transform = res.group(1)
        data_type = res.group(2).lower()
        variant = res.group(3).lower()

        var_name = {'useraccel': 'userAcceleration', 'rotationrate': 'rotationRate'}[data_type]

        self.npcachefile = os.path.join(datadir,
            "mtspectra_fft_{}{}_{}.pkl".format(data_transform.lower(), data_type, variant))

        print(self.npcachefile)

        self.columns = list(itertools.product([var_name], \
                    ["x","y","z"]))

        NumpyDataset.__init__(self, "deviceMotion", variant, reload_)


    def getValues(self, df):
        print('error: getValue() not implemented for mtspectrum*')
        return np.nan

#    def transformData(self, data):
#        return batchRandomRotation(data)

class MtSpectrum_FFT_SvdUserAccelOutbound(MtSpectrum_FFT):
    pass

class MtSpectrum_FFT_SvdUserAccelRest(MtSpectrum_FFT):
    pass

class MtSpectrum_FFT_SvdUserAccelReturn(MtSpectrum_FFT):
    pass



class MtSpectrum_FFT_SvdRotationRateOutbound(MtSpectrum_FFT):
    pass

class MtSpectrum_FFT_SvdRotationRateRest(MtSpectrum_FFT):
    pass

class MtSpectrum_FFT_SvdRotationRateReturn(MtSpectrum_FFT):
    pass



class MtSpectrum_FFT_RawRotationRateOutbound(MtSpectrum_FFT):
    pass

class MtSpectrum_FFT_RawRotationRateRest(MtSpectrum_FFT):
    pass

class MtSpectrum_FFT_RawRotationRateReturn(MtSpectrum_FFT):
    pass



class MtSpectrum_FFT_RawUserAccelOutbound(MtSpectrum_FFT):
    pass

class MtSpectrum_FFT_RawUserAccelRest(MtSpectrum_FFT):
    pass

class MtSpectrum_FFT_RawUserAccelReturn(MtSpectrum_FFT):
    pass
