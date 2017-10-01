from datamanagement.datasets import dataset
from modeldefs import modeldefs
import logging
import itertools
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))

from keras.models import Model, load_model
from keras.layers import Dense, Flatten, Dropout
import keras
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
from sklearn import metrics
import sys
from datamanagement.datasets import dataset
from classifier import Classifier

outputdir = os.getenv('PARKINSON_DREAM_DATA')

# keep rate for the training, matched with test dataset
keeprate = {'input_1':float(36651./36664),
        'input_2': float(16369.*34631/(36664*34270)),
        'input_3': float(9020.*34631/(36664*23114)),
        'input_4': float(36651./36664),
        'input_5': float(16369.*34631/(36664*34270)),
        'input_6': float(9020.*34631/(36664*23114))}

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

            for ipname in dataset.keys():
                Xinput[ipname] = Xinput[ipname] * np.asarray(
                np.random.binomial(1, keeprate[ipname], Xinput[ipname].shape[0]))[:,None,None]
            # implement own dropout version

            yinput = dataset['input_1'].labels[
                    indices[ib*batchsize:(ib+1)*batchsize]].copy()

            sw = sample_weights[indices[ib*batchsize:(ib+1)*batchsize]].copy()

            ib += 1

            if yinput.shape[0] <=0:
                raise Exception("generator produced empty batch")
            yield Xinput, yinput, sw


class MergeClassifier(Classifier):
    def __init__(self, submodels, name, epochs, training = True, trainalllayers = False,
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

        combs = submodels

        datadict = {}
        for i in range(1,7):
            datadict['input_{}'.format(i)] = dataset[combs[i-1][0]]['input_1'](rmnan=False,
                training = training)

        print("Number of datasets: {}".format(len(datadict)))

        for k in datadict.keys():
            datadict[k].transformData = datadict[k].transformDataFlipRotate

        self.name = name
        self.data = datadict
        self.batchsize = 100

        self.logger.info("Input dimensions:")
        for k in datadict:
            self.logger.info("\t{}: {} x {} -> {}".format(k, len(datadict[k]),
                                    datadict[k].shape, datadict[k].labels.shape))

        modelnames = [ '.'.join(comb) for comb in combs]

        # reload the pretrained models
        self.logger.info("loading {}".format("submodels"))
        self.submodels = [load_model(os.path.join(outputdir, "models", mn+".h5")) for mn in modelnames]
        self.trainalllayers = trainalllayers
        if trainalllayers:
            print("train all layers")

        for i in range(len(self.submodels)):
            self.submodels[i].layers[0].name = 'input_{}'.format(i + 1)
            for j in range(1,len(self.submodels[i].layers)):
                self.submodels[i].layers[j].name = '{}_{}'.format(self.submodels[i].layers[j].name,i)
                if trainalllayers:
                    self.submodels[i].layers[j].trainable = True
                else:
                    self.submodels[i].layers[j].trainable = False


        self.logger.info("split training/test")
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

        self.logger.info("Total number of participants: {}".format(
                len(individuals)))
        self.logger.info("Number of training participants: {}".format(
                len(train_individuals)))
        self.logger.info("Number of test participants: {}".format(
                len(test_individuals)))
        self.logger.info("Number of validation participants: {}".format(
                len(val_individuals)))

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


        # just printing some stuff
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



        self.epochs = epochs
        self.dnn = self.defineModel()

    def loadModel(self):
        Classifier.loadModel(self)
        for i in range(len(self.dnn.layers)):
            if self.trainalllayers:
                self.dnn.layers[i].trainable = True

        self.dnn.summary()
        self.dnn.summary(print_fn = self.logger.info)

    def defineModel(self):

        # reload the pretrained models

        merged_layers = keras.layers.concatenate([ \
            model.layers[-2].output for model in self.submodels])
        inputs = [model.layers[0].output for model in self.submodels]


        merged_layers = Dropout(.5)(merged_layers)
        layer = merged_layers
        layer = Dense(100, activation='relu')(layer)
        layer = Dropout(.5)(layer)
        layer = Dense(20, activation='relu')(layer)
        layer = Dropout(.5)(layer)
        outputs = layer

        #inputs, outputs = self.modelfct(self.data, self.modelparams)
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

        early_stopping = EarlyStopping(monitor='val_loss', patience=2)
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
            callbacks = [reduce_lr, early_stopping])

        self.logger.info("Performance after {} epochs: loss {:1.3f}, val-loss {:1.3f}, acc {:1.3f}, val-acc {:1.3f}".format(self.epochs,
                history.history["loss"][-1],
                history.history["val_loss"][-1],
                history.history["acc"][-1],
                history.history["val_acc"][-1]))
        self.logger.info("Finished training ...")


if __name__ == "__main__":

    combs = {'alldata.integration1': [('svdrotout', 'conv2l_30_300_10_40_30_rofl'),
        ('flprotres','conv2l_30_300_10_40_30_rofl'),
        ('rrotret','conv2l_30_300_10_20_30_rofl'),
        ('fhpwcuaout','conv3l_50_300_10_20_30_10_10_rofl'),
        ('fbpwcuares','conv2l_50_300_10_40_30_rofl'),
        ('fhpwcuaret','conv2l_30_300_10_40_30_rofl')] }

    import argparse
    from argparse import RawTextHelpFormatter

    parser = argparse.ArgumentParser(description = \
            "Integration Model for Parkinson's disease predicion", formatter_class = \
            RawTextHelpFormatter)
    helpstr = ""
    for mkey in combs:
        helpstr += '{}\t{}\n'.format(mkey, combs[mkey])
    parser.add_argument('model', choices = [ k for k in combs],
        help = "Selection of Models:" + helpstr)

    parser.add_argument('--epochs', dest="epochs", type=int,
            default=30, help = "Number of epochs")
    parser.add_argument('--trainall', dest='trainall', action='store_true',
            default=False, help = "Use frozen subnets")

    parser.add_argument('--reloadpar', dest='reloadpar', action='store_true',
                default=False, help = "Reload params")

    args = parser.parse_args()

    model = MergeClassifier(combs[args.model], name=args.model,
                        epochs = args.epochs, trainalllayers = args.trainall)

    if args.reloadpar:
        model.loadModel()

    model.fit()
    model.saveModel()
    model.evaluate()
