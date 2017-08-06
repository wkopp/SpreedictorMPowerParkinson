#!/home/wkopp/anaconda2/bin/python
import itertools
import synapseclient
from modeldefs import modeldefs
from dataset.datasets import dataset
from classifier import Classifier


all_combinations = list(itertools.product(dataset, modeldefs))

for comb in all_combinations:
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
