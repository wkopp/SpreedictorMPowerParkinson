import itertools
import synapseclient
import pandas as pd
import numpy as np
from walkingactivity import WalkingActivity

class AccelWalk(WalkingActivity):
    def __init__(self, limit = None):
        WalkingActivity.__init__(self, limit)

    def getEntries(self, idxs, variant):
        '''
        This method retrieves the accelerated walking measurements
        for a given list of indices idxs and a variant (e.g. "outbound").
        '''

        dflist = []
        for idx in idxs:

            fid = str(self.commondescr["accel_walking_"+variant+\
                  ".json.items"].iloc[idx])
            if fid in self.file_map:
                content = pd.read_json(self.file_map[fid])
            else:
                content = pd.DataFrame(columns = ["x","y","z","timestamp"])

            content['healthCode'] = self.commondescr.iloc[idx].healthCode
            content['recordId'] = self.commondescr.iloc[idx].recordId
            dflist.append(content)


        return pd.concat(dflist)
