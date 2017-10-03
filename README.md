# mPower feature extraction for Parkinson's disease DREAM challenge
## Requirements
To run the scripts you need the following software requirements:
1. Install [Anaconda2-4.4.0](https://www.continuum.io/downloads). This will install Python 2.7 
and several necessary packages, including numpy and pandas.
2. `conda install -c bioconda synapseclient`
3. `conda install -c anaconda joblib`
4. `conda install -c conda-forge keras`
5. `conda install -c conda-forge tensorflow`

We recommend to install tensorflow with GPU utilization to speed up the training and 
model evaluation (see TensorFlow website).

## Environment variables
You need to set the environment variable `PARKINSON_DREAM_DATA` to point to
the directory where the dataset should be stored. For instance,
on Linux use 

`export PARKINSON_DREAM_DATA=/path/to/data/`


## Training

First the individual models need to be pre-trained on the specified dataset
```
cd <repo_root>/code

# Pre-training for submission_v1.csv
python run_all.py -df svdrotout -mf conv3l_30_300_10_20_30_10_10 --rofl
python run_all.py -df flprotres -mf conv2l_30_300_10_20_30 --rofl
python run_all.py -df rrotret -mf conv2l_30_300_10_20_30 --rofl
python run_all.py -df fbpwcuaout -mf conv3l_30_300_10_40_30_10_10 --rofl
python run_all.py -df fbpwcuaout -mf conv2l_30_300_10_20_30 --rofl
python run_all.py -df svduaret -mf conv2l_30_300_10_20_30 --rofl

# Pre-training for submission_v2.csv
python run_all.py -df svdrotout -mf conv2l_30_300_10_40_30 --rofl
python run_all.py -df flprotres -mf conv2l_30_300_10_40_30 --rofl
python run_all.py -df rrotret -mf conv2l_30_300_10_20_30 --rofl
python run_all.py -df fhpwcuaout -mf conv3l_50_300_10_20_30_10_10 --rofl
python run_all.py -df fbpwcuaout -mf conv2l_50_300_10_40_30 --rofl
python run_all.py -df fhpwcuaout -mf conv2l_30_300_10_40_30 --rofl
```

For submission_v2.csv, an integration model was used that used the top-level
feature activities from the pre-trained models as input for neural network
consisting of two layers. The training was performed according to

```
python merge_classifier.py alldata.integration1
```

## Feature prediction

Finally, the feature predictions were generated using
```
python featurizer.py --genfeat1 --genfeat2
```
