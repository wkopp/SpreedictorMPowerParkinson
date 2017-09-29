from datamanagement.datasets import dataset
from modeldefs import modeldefs
import logging
import itertools
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ReduceLROnPlateau

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

def generate_fit_data(dataset, indices, sample_weights, batchsize, augment = True):
    while 1:
        ib = 0
        if len(indices) == 0:
            raise Exception("index list is empty")
        while ib < (len(indices)//batchsize + (1 if len(indices)%batchsize > 0 else 0)):
            Xinput = {}
            for ipname in dataset.keys():
                Xinput[ipname] = dataset[ipname].getData(
                    indices[ib*batchsize:(ib+1)*batchsize], augment).copy()

            yinput = dataset['input_1'].labels[
                    indices[ib*batchsize:(ib+1)*batchsize]].copy()

            sw = sample_weights[indices[ib*batchsize:(ib+1)*batchsize]].copy()

            ib += 1

            if yinput.shape[0] <=0:
                raise Exception("generator produced empty batch")
            yield Xinput, yinput, sw

def generate_predict_data(dataset, indices, batchsize, augment = True):
    while 1:
        ib = 0
        if len(indices) == 0:
            raise Exception("index list is empty")
        while ib < (len(indices)//batchsize + (1 if len(indices)%batchsize > 0 else 0)):
            Xinput = {}
            for ipname in dataset.keys():
                Xinput[ipname] = dataset[ipname].getData(
                    indices[ib*batchsize:(ib+1)*batchsize], augment).copy()

            ib += 1

            yield Xinput

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
        self.batchsize = 100
        #try to use a block instead of random datapoints

        hcode = datadict['input_1'].healthCode

        # determine sample weights

        hcdf = pd.DataFrame(hcode, columns=["healthCode"])
        hcvc = hcdf['healthCode'].value_counts()
        hcdf["nsamples"] = hcdf['healthCode'].map(lambda r: hcvc[r])
        self.sample_weights = 1./hcdf["nsamples"].values

        test_fraction = 0.3
        val_fraction = 0.1

        np.random.seed(1234)

        # split the dataset by participants
        # first select training and test participants
        if sys.version_info[0] < 3:
            individuals = np.asarray(list(set(hcode)))
        else:
            individuals = np.unique(hcode)

        test_individuals = np.random.choice(individuals,
                size=int(test_fraction*len(individuals)),
                replace=False)

        if sys.version_info[0] < 3:
            train_individuals = set(individuals) - set(test_individuals)
        else:
            train_individuals = np.setdiff1d(individuals, test_individuals)

        val_individuals = np.random.choice(np.array(list(train_individuals)),
                size=int(val_fraction*len(train_individuals)),
                replace=False)

        if sys.version_info[0] < 3:
            train_individuals = train_individuals - set(val_individuals)
        else:
            train_individuals = np.setdiff1d(train_individuals, val_individuals)

        # next, obtain the examples for each participant
        def individualStatistics(individ):
            idxs = []
            num_pd = num_nonpd = nex_pd = nex_nonpd = 0
            for indiv in individ:
                samples = np.where(hcode == indiv)[0]
                idxs += list(samples)
                if datadict['input_1'].labels[samples[0]] == 1:
                    num_pd += 1
                    nex_pd += len(samples)
                else:
                    num_nonpd += 1
                    nex_nonpd += len(samples)
            return idxs, num_pd, num_nonpd, nex_pd, nex_nonpd

        self.test_idxs, test_num_pd, test_num_nonpd, \
            test_nex_pd, test_nex_nonpd = individualStatistics(test_individuals)
        self.train_idxs, train_num_pd, train_num_nonpd, \
            train_nex_pd, train_nex_nonpd = individualStatistics(train_individuals)
        self.val_idxs, val_num_pd, val_num_nonpd, \
            val_nex_pd, val_nex_nonpd = individualStatistics(val_individuals)


        self.logger.info("Basic statistics about the dataset (All/PD/Non-PD):")
        self.logger.info("Total number of participants: {}".format(
                len(individuals)))
        self.logger.info("Number of training participants: {}/{}/{}".format(
                len(train_individuals), train_num_pd, train_num_nonpd))
        self.logger.info("Number of test participants: {}/{}/{}".format(
                len(test_individuals), test_num_pd, test_num_nonpd))
        self.logger.info("Number of validation participants: {}/{}/{}".format(
                len(val_individuals), val_num_pd, val_num_nonpd))

        self.logger.info("Total number of exercises: {}".format(
                len(datadict['input_1'])))
        self.logger.info("Number of training exercises: {}/{}/{}".format(
                len(self.train_idxs), train_nex_pd, train_nex_nonpd))
        self.logger.info("Number of test exercises: {}/{}/{}".format(
                len(self.test_idxs), test_nex_pd, test_nex_nonpd))
        self.logger.info("Number of validation exercises: {}/{}/{}".format(
                len(self.val_idxs), val_nex_pd, val_nex_nonpd))

        self.logger.info("Input dimensions:")
        for k in datadict:
            self.logger.info("\t{}: {} x {}".format(k, len(datadict[k]),
                                    datadict[k].shape))

        self.modelfct = model_definition[0]
        self.modelparams = model_definition[1]
        self.epochs = epochs
        self.dnn = self.defineModel()

    def defineModel(self):

        inputs, outputs = self.modelfct(self.data, self.modelparams)
        # this will be our feature predictor
        self.feature_predictor = Model(inputs = inputs, outputs = outputs)

        outputs = Dense(1, activation='sigmoid', name="main_output")(outputs)
        model = Model(inputs = inputs, outputs = outputs)

        model.compile(loss='binary_crossentropy',
                    optimizer='adadelta',
                    metrics=['accuracy'])

        model.summary()
        model.summary(print_fn = self.logger.info)
        return model

    def fit(self, augment = True):
        self.logger.info("Start training ...")

        train_idx = self.train_idxs
        val_idx = self.val_idxs

        bs = self.batchsize

        if sys.version_info[0] < 3:
            use_mp = True
        else:
            use_mp = False

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                      patience=5, min_lr=0.001, verbose=1, cooldown=5)

        history = self.dnn.fit_generator(
            generate_fit_data(self.data, train_idx, self.sample_weights, bs,
                    augment),
            steps_per_epoch = len(train_idx)//bs + \
                (1 if len(train_idx)%bs > 0 else 0),
            epochs = self.epochs,
            validation_data = generate_fit_data(self.data, val_idx,
                self.sample_weights, bs, augment = False),
            validation_steps = len(val_idx)//bs + \
                (1 if len(val_idx)%bs > 0 else 0),
            use_multiprocessing = use_mp,
            callbacks = [reduce_lr])

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

    def loadModel(self):
        filename = outputdir + "/models/" + self.name + ".h5"
        self.logger.info("Load model {}".format(filename))
        self.dnn = load_model(filename)


    def evaluate(self):
        # determine

        yinput = self.data['input_1'].labels

        results = list(self.name.split('.'))

        for idxs, name in zip([self.test_idxs, self.train_idxs], \
                ['test', 'train']):
            y = yinput[idxs]

            rest = 1 if len(idxs)%self.batchsize > 0 else 0
            scores = self.dnn.predict_generator(generate_predict_data(self.data,
                idxs, self.batchsize, False),
                steps = len(idxs)//self.batchsize + rest)

            auc = metrics.roc_auc_score(y, scores)
            prc = metrics.average_precision_score(y, scores)
            f1score = metrics.f1_score(y, scores.round())
            acc = metrics.accuracy_score(y, scores.round())

            results += [auc, prc, f1score, acc]


        perf = pd.DataFrame([results],
            columns=["dataset", "model"] +
            list(itertools.product(['test', 'train'],
                ["auROC", "auPRC", "F1", "Accuracy"])))

        summary_path = os.path.join(outputdir, "perf_summary")
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)
        perf.to_csv(os.path.join(summary_path, self.name + ".csv"),
            header = False, index = False, sep = "\t")
        self.logger.info("Results written to {}".format(self.name + ".csv"))

    def featurize(self):

#        for idxs in range(len(self.data)):
        idxs = np.arange(len(self.data['input_1']))
        rest = 1 if len(idxs)%self.batchsize > 0 else 0

        print("idxs ={}, steps = {}".format(idxs, len(idxs)//self.batchsize + rest))

        scores = self.feature_predictor.predict_generator(
            generate_predict_data(self.data,
            idxs, self.batchsize, False),
            steps = len(idxs)//self.batchsize + rest)
        return scores


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
    parser.add_argument('--augment', dest="augment",
            default=False, action='store_true', help = "Use data augmentation if available")

    args = parser.parse_args()
    name = '.'.join([args.data, args.model])
    print("--augment {}".format(args.augment))
    if args.augment:
        name = '_'.join([name, "rot"])

    da = {}
    for k in dataset[args.data].keys():
        da[k] = dataset[args.data][k]()

    model = Classifier(da,
            modeldefs[args.model], name=name,
                        epochs = args.epochs)

    model.fit(args.augment)
    model.saveModel()
    model.evaluate()
