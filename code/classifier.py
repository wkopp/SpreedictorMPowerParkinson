from datasets import dataset
from modeldefs import modeldefs
import logging
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from sklearn import metrics
import sys

outputdir = os.getenv('PARKINSON_DREAM_DATA')
#logdir = os.getenv('PARKINSON_LOG_DIR')

class Classifier(object):
    def __init__(self, dataset, model_definition, name, epochs,
        logs = "model.log"):
        '''
        :input: is a class that contains the input for the prediction
        :model: is a function that defines a keras model for predicting the PD
        :name: used to store the params and logging

        '''
        logdir = os.path.join(outputdir, "logs")
        print("logdir {}".format(logdir))
        if not os.path.exists(logdir):
            os.makedirs(logdir)

        logging.basicConfig(filename = "/".join([logdir, logs]),
            level = logging.DEBUG,
            format = '%(asctime)s:%(name)s:%(message)s',
            datefmt = '%m/%d/%Y %I:%M:%S')
        self.logger = logging.getLogger(name)

        self.name = name
        self.data = dataset
        self.Xtrain, self.Xtest, self.ytrain, self.ytest = \
             train_test_split(dataset.getData(), dataset.getLabels(),
                test_size = 0.3, random_state=1234)
        self.modelfct = model_definition
        self.epochs = epochs
        self.dnn = self.defineModel()

    def defineModel(self):
        model = Sequential()

        model = self.modelfct(model, self.data.getShape())
        
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(loss='binary_crossentropy',
                    optimizer='adadelta',
                    metrics=['accuracy'])
        self.logger.info(str(model.summary()))
        self.logger.info(model.get_config())
        return model

    def fit(self):
        self.logger.info("Start training ...")
        X = self.Xtrain
        y = self.ytrain
        self.dnn.fit(X, y, batch_size = 100, epochs = self.epochs, 
                validation_split = 0.1)
        self.logger.info("Finished training ...")

    def saveModel(self):
        if not os.path.exists(outputdir + "/models/"):
            os.mkdir(outputdir + "/models/")
        filename = outputdir + "/models/" + self.name + ".h5"
        self.logger.info("Save model {}".format(filename))
        self.dnn.save(filename)

    def loadModel(self, name):
        filename = outputdir + "/models/" + self.name + ".h5"
        self.logger.info("Load model {}".format(filename))
        self.dnn = load_model(filename)


    def evaluateModel(self):
        # determine

        X = self.Xtest
        y = self.ytest
        
        scores = self.dnn.predict(X, batch_size = 100)

        auc = metrics.roc_auc_score(y, scores)
        prc = metrics.average_precision_score(y, scores)
        f1score = metrics.f1_score(y, scores.round())
        acc = metrics.accuracy_score(y, scores.round())
        dname, mname = self.name.split('.')
        perf = pd.DataFrame([[dname, mname,auc,prc,f1score, acc]], 
            columns=["dataset", "model", "auROC", "auPRC", "F1", "Accuracy"])

        summary_path = os.path.join(outputdir, "perf_summary")
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)
        perf.to_csv(os.path.join(summary_path, self.name + ".csv"),
            header = False, index = False, sep = "\t")
        self.logger.info("Results written to {}".format(self.name + ".csv"))
        
if __name__ == "__main__":

    import argparse
    from argparse import RawTextHelpFormatter

    parser = argparse.ArgumentParser(description = \
            'Collection of models for parkinson prediction', formatter_class = \
            RawTextHelpFormatter)
    helpstr = "Models:\n"
    for mkey in modeldefs:
            helpstr += mkey +"\t"+modeldefs[mkey].__doc__+"\n"
    parser.add_argument('model', choices = [ k for k in modeldefs],
            help = "Selection of Models:" + helpstr)
    helpstr = "Datasets:\n"
    for dkey in dataset:
            helpstr += dkey +"\t"+dataset[dkey].__doc__+"\n"
    parser.add_argument('data', choices = [ k for k in dataset],
            help = "Selection of Datasets:" + helpstr)
    parser.add_argument('--name', dest="name", default="", help = "Name-tag")
    parser.add_argument('--epochs', dest="epochs", type=int, 
            default=30, help = "Number of epochs")

    args = parser.parse_args()
    print(args.name)
    name = '.'.join([args.data, args.model])

    model = Classifier(dataset[args.data](), 
            modeldefs[args.model], name=name,
                        epochs = args.epochs)
    model.fit()
    model.saveModel()
    model.evaluateModel()
