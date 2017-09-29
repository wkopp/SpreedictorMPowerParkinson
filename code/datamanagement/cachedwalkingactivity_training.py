import os
import joblib
from .walkingactivity_training import WalkingActivityTraining as WalkingActivity

datadir = os.getenv('PARKINSON_DREAM_DATA')

class CachedWalkingActivityTraining(WalkingActivity):

    def __init__(self, limit = None, download_jsons = True, reload_ = False):

        self.downloadpath = os.path.join(datadir,"download")

        if limit:
            self.cachepath = os.path.join(self.downloadpath,
                             "json_file_map_{:d}.pkl".format(limit))
        else:
            self.cachepath = os.path.join(self.downloadpath,
                            "json_file_map.pkl")

        if not os.path.exists(self.downloadpath):
            os.mkdir(self.downloadpath)

        if  reload_ or not os.path.exists(self.cachepath):
            WalkingActivity.__init__(self, limit, download_jsons)
            joblib.dump((self.commondescr, self.file_map), self.cachepath)

        else:
            self.load()

    def load(self):
        self.commondescr, self.file_map = joblib.load(self.cachepath)

if __name__ == '__main__':
    wa = CachedWalkingActivityTraining(limit = 100, download_jsons = True)
    #ts = wa.getEntryByIndex(0, modality='pedometer', variant='outbound')
    #wa.convertUserAccelerationToWorldFrame(ts)
    #print wa.modality_variants
    wa.extractTimeseriesLengths(limit=None, reload_=True)

