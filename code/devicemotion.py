import itertools
import synapseclient
import pandas as pd
import numpy as np
from cachedwalkingactivity import CachedWalkingActivity as WalkingActivity

class DeviceMotion(WalkingActivity):
    def __init__(self, limit = None):
        WalkingActivity.__init__(self, limit)

    def getEntries(self, idxs, variant):
        '''
        This method retrieves the DeviceMotion measurements
        for a given list of indices idxs and a variant (e.g. "outbound").
        '''

        dflist = []
        for idx in idxs:
            fid = str(self.commondescr["deviceMotion_walking_"+ \
                    variant+".json.items"].iloc[idx])
            if fid in self.file_map:
                content = pd.read_json(self.file_map[fid])
                content = self.process_deviceMotion(content)
            else:
                # what to return for missing values?
                # Maybe there are better alternatives
                names =[ '_'.join(el) for el in itertools.product(\
                        ["magneticField", "gravity", "userAcceleration", \
                            "rotationRate", "attitude"], ["x","y","z"])]
                content = pd.DataFrame(columns = names + \
                        ["timestamp", "attitude_w", "magneticField_accuracy"])
            content['healthCode'] = self.commondescr.iloc[idx].healthCode
            content['recordId'] = self.commondescr.iloc[idx].recordId
            dflist.append(content)

        return pd.concat(dflist)
