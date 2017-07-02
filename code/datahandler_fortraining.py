
class ModelDataHandler(FileHandler):
        self.awo = self.loadAW("outbound")
        self.awr = self.loadAW("return")
        self.awrest = self.loadAW("rest")
        
        self.dmwo = self.loadDM("outbound")
        self.dmwr = self.loadDM("return")
        self.dmwrest = self.loadDM("rest")
    def getAWO(self, indices):
        return self.awo[indices]
                                             
    def getAWR(self, indices):
        return self.awr[indices]
                                             
    def getAWREST(self, indices):
        return self.awrest[indices]
                                             
    def loadJson(self, func, variant):
        '''
        This method reads accel. walking outbound (AWO)
        for all datapoints from the json files
        and returns a numpy array
        of dimensions (Ntasks, Ntimepoints, Ncoordinates).
        '''
        #get maximum Ntimepoints
        x = np.empty((self.data.shape[0], 2000, 3), dtype="float32")
        ntimepoints = []
        for idx in range(self.data.shape[0]):
            ntimepoints.append(func(idx, variant).shape[0])

        x = np.empty((self.data.shape[0], max(ntimepoints), 3), dtype="float32")
        for idx in range(self.data.shape[0]):
            x[idx,:ntimepoints[idx],:] = func(idx, variant)

        return x
                                             
