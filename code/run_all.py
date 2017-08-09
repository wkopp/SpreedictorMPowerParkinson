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
parser.add_argument('--d', dest='datafilter', nargs = '*', default = [''],
        help = "Filter for datasets")
parser.add_argument('--m', dest="modelfilter", nargs = '*', 
        default = [''], help = "Filter for model definitions")

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
    print(name)

    da = {}
    for k in dataset[comb[0]].keys():
        da[k] = dataset[comb[0]][k]()

    model = Classifier(da, modeldefs[comb[1]], name=name, epochs = 30)

    model.fit()
    model.saveModel()
    model.evaluate()
