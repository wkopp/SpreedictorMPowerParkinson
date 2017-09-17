#!/home/wkopp/anaconda2/bin/python
import itertools
import synapseclient
from modeldefs import modeldefs
from datamanagement.datasets import dataset
from classifier import Classifier
import numpy as np

import argparse
from argparse import RawTextHelpFormatter

parser = argparse.ArgumentParser(description = \
        'Run all (or selection) of parkinson`s disease prediciton models.', formatter_class = \
        RawTextHelpFormatter)
parser.add_argument('-df', dest='datafilter', nargs = '*', default = [''],
        help = "Filter for datasets")
parser.add_argument('-mf', dest="modelfilter", nargs = '*',
        default = [''], help = "Filter for model definitions")

parser.add_argument('--epochs', dest="epochs", type=int,
        default = 30, help = "Number of epochs")

parser.add_argument('--noise', dest="noise", action='store_true',
        default=False, help = "Augment with gaussian noise")

parser.add_argument('--rotate', dest="rotate", action='store_true',
        default=False, help = "Augment with random rotations")

parser.add_argument('--flip', dest="flip", action='store_true',
        default=False, help = "Augment by flipping the sign")

args = parser.parse_args()
print(args.datafilter)
print(args.modelfilter)

import re

all_combinations = list(itertools.product(dataset, modeldefs))

for comb in all_combinations:

    x  = [ type(re.search(d, comb[0])) != type(None) for d in args.datafilter]
    if not np.any(x):
        continue
    print(comb)
    x  = [ type(re.search(d, comb[1])) != type(None) for d in args.modelfilter]
    if not np.any(x):
        continue

    #name = '.'.join([args.data, args.model])

    print("Running {}-{}".format(comb[0],comb[1]))
    name = '.'.join(comb)
    print("--noise {}".format(args.noise))

    if args.noise:
        name = '_'.join([name, "aug"])
    print(name)
    print("--rotate {}".format(args.rotate))
    if args.rotate:
        name = '_'.join([name, "rot"])
    print(name)
    print("--flip {}".format(args.flip))
    if args.flip:
        name = '_'.join([name, "flip"])
    print(name)

    da = {}
    for k in dataset[comb[0]].keys():
        da[k] = dataset[comb[0]][k]()
        if args.noise:
            da[k].transformData = da[k].transformDataNoise
        if args.rotate:
            da[k].transformData = da[k].transformDataRotate
        if args.flip:
            da[k].transformData = da[k].transformDataFlipSign

    model = Classifier(da, modeldefs[comb[1]], name=name, epochs = args.epochs)

    model.fit(args.noise|args.rotate|args.flip)
    model.saveModel()
    model.evaluate()
