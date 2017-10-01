import os
import synapseclient
from modeldefs import modeldefs
from datamanagement.datasets import dataset
import itertools
from classifier import Classifier
from merge_classifier import MergeClassifier
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
        print("Submission file written to {}".format("submission_v1.csv"))

    def generateSubmissionFileV2(self):
            # determine
            print("Generate submission_v2")


            all_combinations = {'alldata.integration1':[('svdrotout', 'conv2l_30_300_10_40_30_rofl'),
                    ('flprotres','conv2l_30_300_10_40_30_rofl'),
                    ('rrotret','conv2l_30_300_10_20_30_rofl'),
                    ('fhpwcuaout','conv3l_50_300_10_20_30_10_10_rofl'),
                    ('fbpwcuares','conv2l_50_300_10_40_30_rofl'),
                    ('fhpwcuaret','conv2l_30_300_10_40_30_rofl')] }

            # extract recordids
            da = dataset[all_combinations['alldata.integration1'][0][0]]['input_1'](training=False)
            recordids = da.recordIds
            del da


            model = MergeClassifier(all_combinations["alldata.integration1"],
                name="alldata.integration1", epochs = 1, training = False)

            model.loadModel()
            features = model.featurize()

            print("Feature Dimension: {}".format(features.shape))
            pdfeatures = pd.DataFrame(data = features,
                index = pd.Index(data=recordids, name="recordId"),
                columns= list(['{}{}'.format(x,y) for x,y in
                        itertools.product(['Feature'], range(features.shape[1]))]))

            pdfeatures.to_csv(os.path.join(self.summary_path, "submission_v2.csv"),
                    sep = ",")
            print("Submission file written to {}".format("submission_v2.csv"))

    def submit(self, name):
        folderid = 'syn10932057'
        import synapseclient
        from synapseclient import File, Evaluation
        syn = synapseclient.login()

        # upload the file to the synapse project folder
        submissionfile = File(os.path.join(self.summary_path, name + ".csv"),
            parent = folderid)
        submissionfile = syn.store(submissionfile)

        team_entity = syn.getTeam("Spreedictors")
        submission = syn.submit(evaluation = 9606375,
            entity = submissionfile, name = "Spreedictor_{}".format(name),
            team = team_entity)

        syn.logout()

if __name__ == "__main__":
    fe = Featurizer()
    fe.generateSubmissionFileV1()
    #fe.submit("submission_v1")
    fe.generateSubmissionFileV2()
    #fe.submit("submission_v2")
