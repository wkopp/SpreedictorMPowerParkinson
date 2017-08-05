from rawuseraccel import *
#from rawrotationrate import  RawRotationRateRest, RawRotationRateReturn, RawRotationRateOutbound
from rawrotationrate import *
from rawgravity import *
from rawdevicemotion import *
from nony_devicemotion import *
from removenonewalk_devicemotion import *
from svduseraccel import *

dataset = {
        #Raw userAccel
        'ruao':{'input_1':RawUserAccelOutbound},
        'ruaret':{'input_1':RawUserAccelReturn},
        'ruar':{'input_1':RawUserAccelRest},


        #Raw rotationRate
        'ruao':{'input_1':RawRotationRateOutbound},
        'ruaret':{'input_1':RawRotationRateReturn},
        'ruar':{'input_1':RawRotationRateRest},

        #Raw rotationRate
        'ruao':{'input_1':RawGravityOutbound},
        'ruaret':{'input_1':RawGravityReturn},
        'ruar':{'input_1':RawGravityRest},

        #Raw DeviceMotion (all measurements)
        'rdmo':{'input_1':RawDeviceMotionOutbound},
        'rdmret':{'input_1':RawDeviceMotionReturn}, 
        'rdmr':{'input_1':RawDeviceMotionRest}, 

        #Remove None-Y
        'nydmo':{'input_1':NonYDeviceMotionOutbound},

        #RemoveNoneWalk
        'rnwdmo': {'input_1':RemoveNoneWalkDeviceMotionOutbound},
        'rnwdmr': {'input_1':RemoveNoneWalkDeviceMotionRest},
        'rnwdmret': {'input_1':RemoveNoneWalkDeviceMotionReturn},

        #SVD userAccel
        'svduao':{'input_1':SvdUserAccelOutbound},
        'svduaret':{'input_1':SvdUserAccelReturn},
        'svduar':{'input_1':SvdUserAccelRest},
}
