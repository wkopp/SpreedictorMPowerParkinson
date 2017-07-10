import synapseclient
import itertools
import pandas as pd
import shutil
import numpy as np
import joblib
import os

datadir = os.getenv('PARKINSON_DREAM_DATA')

class WalkingActivity(object):
    def __init__(self, limit = None):

        self.synapselocation = 'syn10146553'

        self.download(limit)


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

    def download(self, limit):

        syn = synapseclient.Synapse()

        syn.login()

        selectstr = 'select * from {}'.format(self.synapselocation)
        if limit:
            selectstr += " limit {:d}".format(limit)
        results = syn.tableQuery(selectstr)

        df = results.asDataFrame()

        df[["createdOn"]] = df[["createdOn"]].apply(pd.to_datetime)
        df.fillna(value=-1, inplace=True)
        df[df.columns[5:-1]] = df[df.columns[5:-1]].astype("int64")

        filemap = {}

        for col in df.columns[5:-1]:
            print("Downloading {}".format(col))
            json_files = syn.downloadTableColumns(results, col)
            filemap.update(json_files)

        self.commondescr = df
        self.file_map = filemap
        syn.logout()
