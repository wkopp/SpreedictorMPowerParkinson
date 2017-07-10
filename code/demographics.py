import synapseclient
import itertools
import pandas as pd
import shutil
import joblib
import os

datadir = os.getenv('PARKINSON_DREAM_DATA')

class Demographics():

    def __init__(self, reload_ = False):
        self.synapselocation = "syn10146552"
        
        self.download()
        
    
    def download(self):
        '''
        Download dataset as pandas dataframe
        '''

        syn = synapseclient.Synapse()

        syn.login()

        results = syn.tableQuery('select * from {}'.format(
            self.synapselocation))
    

        df = results.asDataFrame()
        df["idx"]=df.index

        syn.logout()
        self.dataframe = df

    def getData(self):
        return self.dataframe
