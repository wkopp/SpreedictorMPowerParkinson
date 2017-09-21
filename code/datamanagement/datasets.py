from .rawuseraccel import *
#from rawrotationrate import  RawRotationRateRest, RawRotationRateReturn, RawRotationRateOutbound
from .rawrotationrate import *
from .rawgravity import *
from .rawdevicemotion import *
from .nony_devicemotion import *
from .nony_useraccel import *
from .nony_removenonewalk_devicemotion import *
from .ny_rnw_wc_useraccel import *
from .removenonewalk_devicemotion import *
from .removenonewalk_useraccel import *
from .svduseraccel import *
from .svdrotationrate import *
from .worldcoorduseraccel import *
from .mtspectrum_svduseraccel import *
from .filter_lp_rawuseraccel import *
from .filter_hp_rawuseraccel import *
from .filter_bp_rawuseraccel import *
from .filter_lp_rawrotationrate import *
from .filter_hp_rawrotationrate import *
from .filter_bp_rawrotationrate import *

dataset = {
        #Raw userAccel
        'ruaout':{'input_1':RawUserAccelOutbound},
        'ruaret':{'input_1':RawUserAccelReturn},
        'ruares':{'input_1':RawUserAccelRest},


        #Raw rotationRate
        'rrotout':{'input_1':RawRotationRateOutbound},
        'rrotret':{'input_1':RawRotationRateReturn},
        'rrotres':{'input_1':RawRotationRateRest},

        #Raw Gravity
        'rgrout':{'input_1':RawGravityOutbound},
        'rgrret':{'input_1':RawGravityReturn},
        'rgrres':{'input_1':RawGravityRest},

        #Raw DeviceMotion (all measurements)
        'rdmout':{'input_1':RawDeviceMotionOutbound},
        'rdmret':{'input_1':RawDeviceMotionReturn},
        'rdmres':{'input_1':RawDeviceMotionRest},

        #Remove None-Y
        'nydmout':{'input_1':NonYDeviceMotionOutbound},
        'nydmres':{'input_1':NonYDeviceMotionRest},
        'nydmret':{'input_1':NonYDeviceMotionReturn},

        #Remove None-Y
        'nyrnwdmout':{'input_1':NonYRNWDeviceMotionOutbound},
        'wcnyrnwdmout':{'input_1':WorldCoordNYRNWUserAccelOutbound},

        #Remove None-Y
        'nyuaout':{'input_1':NonYUserAccelOutbound},
        'nyuares':{'input_1':NonYUserAccelRest},
        'nyuaret':{'input_1':NonYUserAccelReturn},

        #RemoveNoneWalk
        'rnwdmout': {'input_1':RemoveNoneWalkDeviceMotionOutbound},
        'rnwdmres': {'input_1':RemoveNoneWalkDeviceMotionRest},
        'rnwdmret': {'input_1':RemoveNoneWalkDeviceMotionReturn},

        #RemoveNoneWalk
        'rnwuaout': {'input_1':RemoveNoneWalkUserAccelOutbound},
        'rnwuares': {'input_1':RemoveNoneWalkUserAccelRest},
        'rnwuaret': {'input_1':RemoveNoneWalkUserAccelReturn},

        #SVD userAccel
        'svduaout':{'input_1':SvdUserAccelOutbound},
        'svduaret':{'input_1':SvdUserAccelReturn},
        'svduares':{'input_1':SvdUserAccelRest},

        #SVD rotationRate
        'svdrotout':{'input_1':SvdRotationRateOutbound},
        'svdrotret':{'input_1':SvdRotationRateReturn},
        'svdrotres':{'input_1':SvdRotationRateRest},

        #World userAccel
        'wcuaout':{'input_1':WorldCoordUserAccelOutbound},
        'wcuaret':{'input_1':WorldCoordUserAccelReturn},
        'wcuares':{'input_1':WorldCoordUserAccelRest},

        #MTSpectrum SVD userAccel
        'mtssvduao' : {'input_1':MtSpectrum_SvdUserAccelOutbound},

        # low pass filtered raw userAccel
        'flpruaout':{'input_1':FilterLowPassRawUserAccelOutbound},
        'flpruaret':{'input_1':FilterLowPassRawUserAccelReturn},
        'flpruares':{'input_1':FilterLowPassRawUserAccelRest},

        # high pass filtered raw userAccel
        'fhpruaout': {'input_1': FilterHighPassRawUserAccelOutbound},
        'fhpruaret': {'input_1': FilterHighPassRawUserAccelReturn},
        'fhpruares': {'input_1': FilterHighPassRawUserAccelRest},

        # band pass filtered raw userAccel
        'fbpruaout': {'input_1': FilterBandPassRawUserAccelOutbound},
        'fbpruaret': {'input_1': FilterBandPassRawUserAccelReturn},
        'fbpruares': {'input_1': FilterBandPassRawUserAccelRest},

        # low pass filtered raw rotationRate
        'flprotout': {'input_1': FilterLowPassRawRotationRateOutbound},
        'flprotret': {'input_1': FilterLowPassRawRotationRateReturn},
        'flprotres': {'input_1': FilterLowPassRawRotationRateRest},

        # high pass filtered raw rotationRate
        'fhprotout': {'input_1': FilterHighPassRawRotationRateOutbound},
        'fhprotret': {'input_1': FilterHighPassRawRotationRateReturn},
        'fhprotres': {'input_1': FilterHighPassRawRotationRateRest},

        # band pass filtered raw rotationRate
        'fbprotout': {'input_1': FilterBandPassRawRotationRateOutbound},
        'fbprotret': {'input_1': FilterBandPassRawRotationRateReturn},
        'fbprotres': {'input_1': FilterBandPassRawRotationRateRest},
}
