#!/home/wkopp/anaconda2/bin/python
import itertools
import synapseclient
from modeldefs import modeldefs
from datasets import dataset
from classifier import Classifier


all_combinations = list(itertools.product(dataset, modeldefs))

for comb in all_combinations:
    print("Running {}-{}".format(comb[0],comb[1]))
    name = '.'.join(comb)
    print(name)
    model = Classifier(dataset[comb[0]](), 
            modeldefs[comb[1]], name=name,
                        epochs = 30)
    model.fit()
    model.saveModel()
    model.evaluateModel()
