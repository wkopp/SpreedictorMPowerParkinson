import os
import synapseclient
from modeldefs import modeldefs
from datamanagement.datasets import dataset
import itertools
from classifier import Classifier
import numpy as np
import pandas as pd


outputdir = os.getenv('PARKINSON_DREAM_DATA')


class Featurizer(object):
    '''
    This class reuses the pretrained Classifiers
    and generates feature predictions on the given
    dataset
    '''

    def __init__(self):
        '''
        Init Featurizer
        '''
        summary_path = os.path.join(outputdir, "submission")
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)
        self.summary_path = summary_path



    def generateSubmissionFileV1(self):
        # determine

        all_combinations = [('svdrotout','conv3l_30_300_10_20_30_10_10'),
        ('flprotres', 'conv2l_30_300_10_20_30'),
        ('rrotret', 'conv2l_30_300_10_20_30'),
        ('fbpwcuaout', 'conv3l_30_300_10_40_30_10_10'),
        ('fbpwcuares', 'conv2l_30_300_10_20_30'),
        ('svduaret', 'conv2l_30_300_10_20_30')]

        da = dataset[all_combinations[0][0]]['input_1'](training=False)
        #features = np.empty((len(da), 10+20+20+10+20+20), dtype="float32")
        recordids = da.recordIds
        del da

        features = []
        for comb in all_combinations:

            name = '.'.join(comb)
            name = '_'.join([name, "rofl"])

            da = {}
            for k in dataset[comb[0]].keys():
                da[k] = dataset[comb[0]][k](training=False)
                da[k].transformData = da[k].transformDataFlipRotate

            print("Featurize {}".format(comb))
            model = Classifier(da, modeldefs[comb[1]], name=name, epochs = 1)
            model.loadModel()
            features.append(model.featurize())

        features = np.concatenate(features, axis=1)
        print("Feature Dimension: {}".format(features.shape))
        pdfeatures = pd.DataFrame(data = features,
            index = pd.Index(data=recordids, name="recordId"),
            columns= list(['{}{}'.format(x,y) for x,y in
                    itertools.product(['Feature'], range(features.shape[1]))]))

        pdfeatures.to_csv(os.path.join(self.summary_path, "submission_v1.csv"),
                sep = ",")
        print("Submission file written to {}".format(self.name + "_v1.csv"))

    def submitV1(self):
        folderid = 'syn10932057'
        import synapseclient
        from synapseclient import File, Evaluation
        syn = synapseclient.login()

        # upload the file to the synapse project folder
        submissionfile = File(os.path.join(self.summary_path, "submission_v1.csv"),
            parent = folderid)
        submissionfile = syn.store(submissionfile)

        team_entity = syn.getTeam("Spreedictors")
        submission = syn.submit(evaluation = 9606375,
            entity = submissionfile, name = "Spreedictor_submission_v1",
            team = team_entity)

        syn.logout()

if __name__ == "__main__":
    fe = Featurizer()
    #fe.generateSubmissionFileV1()
    fe.submitV1()
