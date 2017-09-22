from .classifier import Classifier


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




    def generateSubmissionFile(self):
        # determine

        results = list(self.name.split('.'))

        idxs = range(self.data.shape[0])

        rest = 1 if len(idxs)%self.batchsize > 0 else 0

        features = self.dnn.predict_generator(generate_data(self.data,
            idxs, self.batchsize, False),
            steps = len(idxs)//self.batchsize + rest)


        perf = pd.DataFrame([results],
            columns=["recordId"] +
            list(['{}{}'.format(x,y) for x,y in
                itertools.product(['Feature'], range(10))]))

        summary_path = os.path.join(outputdir, "submission")
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)

        perf.to_csv(os.path.join(summary_path, self.name + ".csv"),
            header = True, index = False, sep = ",")
        self.logger.info("Submission file written to {}".format(self.name + ".csv"))

