import synapseclient
import pandas as pd
import numpy as np
import os
import joblib
import itertools
import quaternion

datadir = os.getenv('PARKINSON_DREAM_DATA')

class WalkingActivity(object):
    modalities = ['deviceMotion', 'accel', 'pedometer']
    variants = ['outbound', 'rest', 'return']
    modality_variants = sorted(list(set([x for x in itertools.product(modalities, variants)]) - set([('pedometer', 'rest')])))
    modality_variants_timeseries = sorted([x for x in itertools.product(['deviceMotion', 'accel'], ['outbound', 'rest', 'return'])])

    def __init__(self, limit = None, download_jsons = True, reload_ = False):

        self.synapselocation = 'syn10146553'

        self.downloadpath = datadir + "download/"

        if limit:
            self.cachepath = self.downloadpath + \
                             "json_file_map_{:d}.pkl".format(limit)
        else:
            self.cachepath = self.downloadpath + "json_file_map.pkl"

        if not os.path.exists(self.downloadpath) or reload_ or \
                not os.path.exists(self.cachepath):
            self.download(limit, download_jsons)
        else:
            self.load()

    def load(self):
        self.commondescr, self.file_map = joblib.load(self.cachepath)

    def getCommonDescriptor(self):
        '''
        Method returns the pandas dataframe containing 'recordId' and 'healthCode'
        '''
        return self.commondescr

    def getFileMap(self):
        '''
        Method returns a dictionary containing the filehandles as keys and
        the respective paths
        '''
        return self.file_map

    def download(self, limit, download_jsons = True):

        if not os.path.exists(self.downloadpath):
            os.mkdir(self.downloadpath)

        syn = synapseclient.Synapse()

        syn.login()

        selectstr = 'select * from {}'.format(self.synapselocation)
        if limit:
            selectstr += " limit {:d}".format(limit)
        results = syn.tableQuery(selectstr)

        df = results.asDataFrame()

        df[["createdOn"]] = df[["createdOn"]].apply(pd.to_datetime)
        #df.fillna(value=-1, inplace=True)
        #df[df.columns[5:-1]] = df[df.columns[5:-1]].astype("int64")

        filemap = {}

        if download_jsons:
            for col in df.columns[5:-1]:
                print("Downloading {}".format(col))
                json_files = syn.downloadTableColumns(results, col)
                filemap.update(json_files)

        self.commondescr = df
        self.file_map = filemap

        joblib.dump((df, filemap), self.cachepath)

        syn.logout()

    def getColNames(self, colname):
        suffixes = ['x', 'y', 'z']

        if colname == 'attitude':
            suffixes.append('w')

        return ['_'.join(x) for x in itertools.product([colname], suffixes)]


    def process_deviceMotion(self, content):
        for col in content.columns:
            if col == "timestamp":
                continue
            for key in content[col][0]:
                content[col + "_" + key] = [d[key] \
                                            for d in content[col].values]
        content.drop(['attitude', 'magneticField', 'gravity', \
                      'rotationRate', 'userAcceleration'], \
                     inplace=True, axis=1)
        return content

    def getEntryByRecordId(self, recordId, modality, variant):

        dataEntry = self.commondescr[self.commondescr["recordId"] == recordId]

        if dataEntry.size == 0:
            raise Exception("Invalid recordId: {}".format(recordId))

        colname = modality + "_walking_" + variant + ".json.items"

        if not colname in self.commondescr.columns:
            raise Exception("Invalid modality/variant combination: {} / {}".format(modality, variant))

        fid = dataEntry[colname]

        if pd.isnull(fid[0]):
            return pd.DataFrame()

        fid = str(int(fid))
        if fid not in self.file_map:
            raise Exception('fileid "{}" not found'.format(fid))

        content = pd.read_json(self.file_map[fid])

        if modality == 'deviceMotion':
            self.process_deviceMotion(content)

        content['healthCode'] = dataEntry["healthCode"][0]
        content['recordId'] = recordId
        if not modality == 'pedometer':
            content['time_in_task'] = content['timestamp'] - content['timestamp'][0]

        return content


    def getEntryByIndex(self, idx, modality, variant):
        return self.getEntryByRecordId(self.commondescr.iloc[idx]['recordId'], modality, variant)


    def convertUserAccelerationToWorldFrame(self, ts):
        colNames = wa.getColNames('userAcceleration_world')

        if ts.columns.intersection(colNames).size == len(colNames):
            return ts

        rot = np.empty((ts.shape[0], 3))
        rot[:] = np.NAN
        for i in range(ts.shape[0]):
                rot[i, :] = quaternion.quat_mult(ts[['attitude_x', 'attitude_y', 'attitude_z', 'attitude_w']].iloc[i],
                                     ts[['userAcceleration_x', 'userAcceleration_y', 'userAcceleration_z']].iloc[i])

        ts[colNames] = pd.DataFrame(rot, index=ts.index)

        return ts

    def extractTimeseriesLengths(self, limit = None, reload_ = False):
        cache_filename_ts_lengths = 'ts_lengths.pkl' if not limit else 'ts_lengths_{:d}.pkl'.format(
            limit)
        cachepath_ts_lengths = os.path.join(self.downloadpath, cache_filename_ts_lengths)

        if os.path.exists(cachepath_ts_lengths) and not reload_:
            df_ts_lengths = joblib.load(cachepath_ts_lengths)
        else:
            n_modality_variants = len(self.modality_variants_timeseries)
            n_timeseries = self.commondescr.shape[0]
            ts_lengths = np.empty((n_timeseries, n_modality_variants * 2))
            ts_lengths[:] = np.nan

            for i_col, (modality, variant) in enumerate(self.modality_variants_timeseries):
                print modality, variant

                for i_row in range(n_timeseries):
                    ts = self.getEntryByIndex(i_row, modality=modality, variant=variant)
                    if ts.size > 0:
                        ts_lengths[i_row, i_col] = ts.shape[0]
                        ts_lengths[i_row, i_col + n_modality_variants] = ts['timestamp'][1] - ts['timestamp'][0]

            df_ts_lengths = pd.DataFrame(
                data=ts_lengths,
                columns=['_'.join(x) for x in self.modality_variants_timeseries] +
                        ['_'.join(x + ('timestep',)) for x in self.modality_variants_timeseries]
            )
            df_ts_lengths[['recordId', 'healthCode']] = self.getCommonDescriptor()[
                ['recordId', 'healthCode']].reset_index(drop=True)

            joblib.dump(df_ts_lengths, cachepath_ts_lengths)

        return df_ts_lengths

if __name__ == '__main__':
    wa = WalkingActivity(limit = 3000, download_jsons = True)
    #ts = wa.getEntryByIndex(0, modality='pedometer', variant='outbound')
    #wa.convertUserAccelerationToWorldFrame(ts)
    #print wa.modality_variants
    wa.extractTimeseriesLengths(limit=1000, reload_=True)
