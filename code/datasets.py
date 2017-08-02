from rawuseraccel import RawUserAccelOutbound
from rawuseraccel import RawUserAccelRest
from rawuseraccel import RawUserAccelReturn
from rawdevicemotion import RawDeviceMotionOutbound, RawDeviceMotionRest, RawDeviceMotionReturn
from nony_devicemotion import NonYDeviceMotionOutbound
from removenonewalk_devicemotion import RemoveNoneWalkDeviceMotionOutbound
from removenonewalk_devicemotion import RemoveNoneWalkDeviceMotionRest
from removenonewalk_devicemotion import RemoveNoneWalkDeviceMotionReturn

dataset = {'ruao':RawUserAccelOutbound,
#            'ruaret':RawUserAccelReturn,
            'ruar':RawUserAccelRest,
        'rdmo':RawDeviceMotionOutbound,
#        'rdmret':RawDeviceMotionReturn, 
        'rdmr':RawDeviceMotionRest, 
        'nydmo':NonYDeviceMotionOutbound,
        'rnwdmo': RemoveNoneWalkDeviceMotionOutbound,
        'rnwdmr': RemoveNoneWalkDeviceMotionRest,
#        'rnwdmret': RemoveNoneWalkDeviceMotionReturn,

}
