import itertools
import synapseclient
import pandas as pd
import numpy as np
from walkingactivity import WalkingActivity

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
                for col in content.columns:
                    if col == "timestamp":
                        continue
                    for key in content[col][0]:
                        content[col + "_" + key] = [ d[key] \
                            for d in content[col].values]
                content.drop(['attitude', 'magneticField', 'gravity', \
                        'rotationRate', 'userAcceleration'], \
                        inplace =True, axis=1)
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
