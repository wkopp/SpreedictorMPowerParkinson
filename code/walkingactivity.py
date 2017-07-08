import itertools
import pandas as pd
import shutil
import numpy as np
import joblib
import os
from downloadable import Downloadable

datadir = os.getenv('PARKINSON_DREAM_DATA')

class WalkingActivity(object):
    def __init__(self, reload_ = False):
        
        self.downloadpath = datadir + "download/"
        self.cachepath =  self.downloadpath + "json_file_map.pkl"
        
        self.synapselocation = 'syn10146553'

        if not os.path.exists(self.downloadpath) or reload_ or \
            not os.path.exists(self.cachepath):
            shutil.rmtree(self.downloadpath)
            self.download()
        
        self.load()
        
    
    def load(self):
        self.commondescr, self.file_map =  joblib.load(self.cachepath)
    
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

    def download(self):

        os.mkdir(self.downloadpath)
        syn = synapseclient.Synapse()

        syn.login()

        results = syn.tableQuery('select * from {}'.format(self.synapselocation))
    

        df = results.asDataFrame()

        df[["createdOn"]] = df[["createdOn"]].apply(pd.to_datetime)
        df.fillna(value=-1, inplace=True)
        df[df.columns[5:-1]] = df[df.columns[5:-1]].astype("int64")
    
        json_files = syn.downloadTableColumns(results, df.columns[5:-1])

        for fileid in file_map:
            shutil.move(file_map[fileid], self.downloadpath + \
                    file_map[fileid].split("/")[-1])
            json_files[fileid] = self.downloadpath + \
                    file_map[fileid].split("/")[-1]

        joblib.dump((df, file_map), self.cachepath)
    
