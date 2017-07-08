import synapseclient
import itertools
import pandas as pd
import shutil
import joblib
import os

datadir = os.getenv('PARKINSON_DREAM_DATA')

class Demographics():

    def __init__(self, reload_ = False):
        self.downloadpath = datadir + "download/"
        self.cachepath =  self.downloadpath + "demographics.pkl"
        self.synapselocation = "syn10146552"
        
        if not os.path.exists(self.downloadpath) or reload_ or \
            not os.path.exists(self.cachepath):
            shutil.rmtree(self.downloadpath)
            self.download()
        
        self.load()
        
    
    def load(self):
        self.dataframe =  joblib.load(self.cachepath)

    def download(self):
        '''
        Download dataset as pandas dataframe
        '''

        os.mkdir(self.downloadpath)
        syn = synapseclient.Synapse()

        syn.login()

        results = syn.tableQuery('select * from {}'.format(
            self.synapselocation))
    

        df = results.asDataFrame()
        df["idx"]=df.index

        joblib.dump(df, self.cachepath)

    def getData(self):
        return self.dataframe
