from datamanagement.datasets import dataset
from modeldefs import modeldefs
import logging
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))

from keras.models import Model, load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
from sklearn import metrics
import sys

outputdir = os.getenv('PARKINSON_DREAM_DATA')

def generate_data(dataset, indices, batchsize, augment = True):
    while 1:
        #create a dict
        ib = 0
        while ib <= len(indices)//batchsize:
            Xinput = {}
            for ipname in dataset.keys():
                Xinput[ipname] = dataset[ipname].getData(augment)[
                    indices[ib*batchsize:(ib+1)*batchsize]]
                
            yinput = dataset['input_1'].getLabels()[
                    indices[ib*batchsize:(ib+1)*batchsize]]
            ib += 1

            #print(yinput.shape)
            yield Xinput, yinput
        
#logdir = os.getenv('PARKINSON_LOG_DIR')

class Classifier(object):
    def __init__(self, datadict, model_definition, name, epochs,
        logs = "model.log"):
        '''
        :input: is a class that contains the input for the prediction
        :model: is a function that defines a keras model for predicting the PD
        :name: used to store the params and logging
        '''
        logdir = os.path.join(outputdir, "logs")
        if not os.path.exists(logdir):
            os.makedirs(logdir)

        logging.basicConfig(filename = "/".join([logdir, logs]),
            level = logging.DEBUG,
            format = '%(asctime)s:%(name)s:%(message)s',
            datefmt = '%m/%d/%Y %I:%M:%S')
        self.logger = logging.getLogger(name)

        self.name = name
        self.data = datadict
        self.Ndatapoints = datadict['input_1'].getNdatapoints()
        #try to use a block instead of random datapoints
        
        hcode = datadict['input_1'].getHealthCode()

        test_fraction = 0.3

        # split the dataset by participants
        # first select training and test participants
        individuals = np.asarray(list(set(hcode)))
        test_individuals = np.random.choice(individuals, 
                size=int(test_fraction*len(individuals)), 
                replace=False)
        train_individuals = set(individuals) - set(test_individuals)

        # then retrieve the associated samples for the participants
        #permidxs = np.arange(int(0.3*self.Ndatapoints), self.Ndatapoints)
        #np.random.shuffle(permidxs)
        #idxs = np.zeros((self.Ndatapoints), dtype=bool)
        self.test_idxs = []
        for indiv in test_individuals:
            self.test_idxs += list(np.where(hcode == indiv)[0])
        
        self.train_idxs = []
        for indiv in train_individuals:
            self.train_idxs += list(np.where(hcode == indiv)[0])

#        np.random.shuffle(self.train_idxs)
#        permidxs =  np.arange(self.Ndatapoints)
#        np.random.shuffle(permidxs)
#        self.test_idxs = permidxs[:int(0.3*self.Ndatapoints)]
#        self.train_idxs = permidxs[int(0.3*self.Ndatapoints):]

        self.logger.info("Number of training participants: {}".format(len(train_individuals)))
        self.logger.info("Number of test participants: {}".format(len(test_individuals)))
        self.logger.info("Total number of examples: {}".format(self.Ndatapoints))
        self.logger.info("Number of training examples: {}".format(len(self.train_idxs)))
        self.logger.info("Number of test examples: {}".format(len(self.test_idxs)))

        self.logger.info("Input dimensions:")
        for k in datadict:
            self.logger.info("\t{}: {} x {}".format(k, datadict[k].getNdatapoints(),
                                    datadict[k].getShape()))

        #self.logger.info("Output dimension:")
            #self.logger.info("\t{}: {}".format(k, 
        #self.train_idxs = permidxs#[int(0.3*self.Ndatapoints):]
        #self.test_idxs = permidxs[:int(0.3*self.Ndatapoints)]
        #self.test_idxs = range(int(0.3*self.Ndatapoints))

        self.modelfct = model_definition[0]
        self.modelparams = model_definition[1]
        self.epochs = epochs
        self.dnn = self.defineModel()

    def defineModel(self):

        inputs, outputs = self.modelfct(self.data, self.modelparams)
        
        outputs = Dense(1, activation='sigmoid', name="main_output")(outputs)
        model = Model(inputs = inputs, outputs = outputs)
        
        model.compile(loss='binary_crossentropy',
                    optimizer='adadelta',
                    metrics=['accuracy'])

        model.summary()
        model.summary(print_fn = self.logger.info)
        return model

    def fit(self):
        self.logger.info("Start training ...")

        train_idx = self.train_idxs[:int(len(self.train_idxs)*.9)]
        val_idx = self.train_idxs[int(len(self.train_idxs)*.9):]

        history = self.dnn.fit_generator(generate_data(self.data, train_idx, 100),
            steps_per_epoch = len(train_idx)//100, epochs = self.epochs, 
            validation_data = generate_data(self.data, val_idx, 100),
            validation_steps = len(val_idx)//100, use_multiprocessing = False)

        self.logger.info("Performance after {} epochs: loss {:1.3f}, val-loss {:1.3f}, acc {:1.3f}, val-acc {:1.3f}".format(self.epochs,
                history.history["loss"][-1],
                history.history["val_loss"][-1],
                history.history["acc"][-1],
                history.history["val_acc"][-1]))
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


    def evaluate(self):
        # determine

        yinput = self.data['input_1'].getLabels()
        y = yinput[self.test_idxs]
        
        rest = 1 if len(self.test_idxs)%100 > 0 else 0
        scores = self.dnn.predict_generator(generate_data(self.data, 
            self.test_idxs, 100, False), 
            steps = len(self.test_idxs)//100 + rest)

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
            helpstr += mkey +"\t"+modeldefs[mkey][0].__doc__.format(*modeldefs[mkey][1])+"\n"
    parser.add_argument('model', choices = [ k for k in modeldefs],
            help = "Selection of Models:" + helpstr)
    helpstr = "Datasets:\n"
    for dkey in dataset:
            # for now lets stick with input_1 only
            # TODO: later make this more general
            helpstr += dkey +"\t"+dataset[dkey]['input_1'].__doc__+"\n"
    parser.add_argument('data', choices = [ k for k in dataset],
            help = "Selection of Datasets:" + helpstr)
    parser.add_argument('--name', dest="name", default="", help = "Name-tag")
    parser.add_argument('--epochs', dest="epochs", type=int, 
            default=30, help = "Number of epochs")

    args = parser.parse_args()
    name = '.'.join([args.data, args.model])

    da = {}
    for k in dataset[args.data].keys():
        da[k] = dataset[args.data][k]()
    model = Classifier(da, 
            modeldefs[args.model], name=name,
                        epochs = args.epochs)
    model.fit()
    model.saveModel()
    model.evaluate()
