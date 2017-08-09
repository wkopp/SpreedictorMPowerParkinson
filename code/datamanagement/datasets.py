from rawuseraccel import *
#from rawrotationrate import  RawRotationRateRest, RawRotationRateReturn, RawRotationRateOutbound
from rawrotationrate import *
from rawgravity import *
from rawdevicemotion import *
from nony_devicemotion import *
from removenonewalk_devicemotion import *
from svduseraccel import *
from svdrotationrate import *
from worldcoorduseraccel import *

dataset = {
        #Raw userAccel
        'ruao':{'input_1':RawUserAccelOutbound},
        'ruaret':{'input_1':RawUserAccelReturn},
        'ruar':{'input_1':RawUserAccelRest},


        #Raw rotationRate
        'rroto':{'input_1':RawRotationRateOutbound},
        'rrotret':{'input_1':RawRotationRateReturn},
        'rrotr':{'input_1':RawRotationRateRest},

        #Raw Gravity
        'rgro':{'input_1':RawGravityOutbound},
        'rgrret':{'input_1':RawGravityReturn},
        'rgrr':{'input_1':RawGravityRest},

        #Raw DeviceMotion (all measurements)
        'rdmo':{'input_1':RawDeviceMotionOutbound},
        'rdmret':{'input_1':RawDeviceMotionReturn}, 
        'rdmr':{'input_1':RawDeviceMotionRest}, 

        #Remove None-Y
        'nydmo':{'input_1':NonYDeviceMotionOutbound},
        'nydmr':{'input_1':NonYDeviceMotionRest},
        'nydmret':{'input_1':NonYDeviceMotionReturn},

        #RemoveNoneWalk
        'rnwdmo': {'input_1':RemoveNoneWalkDeviceMotionOutbound},
        'rnwdmr': {'input_1':RemoveNoneWalkDeviceMotionRest},
        'rnwdmret': {'input_1':RemoveNoneWalkDeviceMotionReturn},

        #SVD userAccel
        'svduao':{'input_1':SvdUserAccelOutbound},
        'svduaret':{'input_1':SvdUserAccelReturn},
        'svduar':{'input_1':SvdUserAccelRest},

        #SVD rotationRate
        'svdroto':{'input_1':SvdRotationRateOutbound},
        'svdrotret':{'input_1':SvdRotationRateReturn},
        'svdrotr':{'input_1':SvdRotationRateRest},

        #World userAccel
        'wcuao':{'input_1':WorldCoordUserAccelOutbound},
        'wcuaret':{'input_1':WorldCoordUserAccelReturn},
        'wcuar':{'input_1':WorldCoordUserAccelRest},
}
