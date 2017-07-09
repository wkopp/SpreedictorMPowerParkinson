import synapseclient
import itertools
import pandas as pd
import shutil
import numpy as np
import joblib
import os

datadir = os.getenv('PARKINSON_DREAM_DATA')

class WalkingActivity(object):
    def __init__(self, limit = None, reload_ = False):
        
        self.downloadpath = datadir + "download/"
        if limit:
            self.cachepath =  self.downloadpath + \
                    "json_file_map_{:d}.pkl".format(limit)
        else:
            self.cachepath =  self.downloadpath + "json_file_map.pkl"
        
        self.synapselocation = 'syn10146553'

        if not os.path.exists(self.downloadpath) or reload_ or \
            not os.path.exists(self.cachepath):
            #shutil.rmtree(self.downloadpath, ignore_errors = True)
            self.download(limit)
        
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

    def download(self, limit):

        os.mkdir(self.downloadpath)
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

            for fileid in json_files:
                shutil.move(json_files[fileid], self.downloadpath + \
                        json_files[fileid].split("/")[-1])
                json_files[fileid] = self.downloadpath + \
                        json_files[fileid].split("/")[-1]
            filemap.update(json_files)

        joblib.dump((df, filemap), self.cachepath)
    
