import synapseclient
import itertools
import pandas as pd
import shutil
import numpy as np
import joblib
import os

datadir = os.getenv('PARKINSON_DREAM_DATA')

class DataHandler(object):
    def __init__(self, reload_ = False):
        
        self.downloadpath = datadir + "download/"
        self.cachepath =  self.downloadpath + "cache.pkl"
        
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

    def awByIndexFromJson(self, idx, variant):
        fid = str(self.commondescr["accel_walking_"+variant+".json.items"].iloc[idx])
        if fid in self.file_map:
            content = pd.read_json(self.file_map[fid])
        else:
            # what to return for missing values?
            # Maybe there are better alternatives
            content = pd.DataFrame(columns = ["x","y","z","timestamp"])

        return content
    
    def dmByIndexFromJson(self, idx, variant):
        fid = str(self.commondescr["deviceMotion_walking_"+variant+".json.items"].iloc[idx])
        if fid in self.file_map:
            content = pd.read_json(self.file_map[fid])
            for col in content.columns:
                if col == "timestamp":
                    continue
                for key in content[col][0]:
                    content[col + "_" + key] = [ d[key] for d in content[col].values]
            content.drop(['attitude', 'magneticField', 'gravity', \
                    'rotationRate', 'userAcceleration'], inplace =True, axis=1)
        else:
            # what to return for missing values?
            # Maybe there are better alternatives
            names =[ '_'.join(el) for el in itertools.product(\
                    ["magneticField", "gravity", "userAcceleration", \
                        "rotationRate", "attitude"], ["x","y","z"])]
            content = pd.DataFrame(columns = names + \
                    ["timestamp", "attitude_w", "magneticField_accuracy"])

        return content
    
    def peByIndexFromJson(self, idx, variant):
        fid = str(self.commondescr["pedometer_walking_"+variant+".json.items"].iloc[idx])
        if fid in self.file_map:
            content = pd.read_json(self.file_map[fid])
        else:
            # what to return for missing values?
            # Maybe there are better alternatives
            content = pd.DataFrame(columns = ['distance', \
                'startDate', 'endDate', 'floorsAscended', 'floorsDescended',\
                'numberOfSteps'])
        return content

    def download(self):

        os.mkdir(self.downloadpath)
        syn = synapseclient.Synapse()

        syn.login()

        results = syn.tableQuery('select * from syn5713119')
    

    
        df = results.asDataFrame()

        df[["createdOn"]] = df[["createdOn"]].apply(pd.to_datetime)
        df.fillna(value=-1, inplace=True)
        df[df.columns[7:-1]] = df[df.columns[7:-1]].astype("int64")
    
        file_map = syn.downloadTableColumns(results, df.columns[5:-1])
        for fileid in file_map:
            shutil.move(file_map[fileid],datadir + "download/" + file_map[fileid].split("/")[-1])
            file_map[fileid] = datadir + "download/" + file_map[fileid].split("/")[-1]

        joblib.dump((df, file_map), self.cachepath)
    
