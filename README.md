# ParkinsonDream
## Requirements
To run the scripts you need the following software requirements:
1. Install [Anaconda2-4.4.0](https://www.continuum.io/downloads). This will install Python 2 and several necessary packages, including numpy and pandas.
2. `conda install -c bioconda synapseclient`
3. `conda install -c anaconda joblib`

## Environment variables
You need to set the environment variable `PARKINSON_DREAM_DATA` to point to
the directory where the dataset should be stored. For instance,
on Linux use 

`export PARKINSON_DREAM_DATA=/path/to/data/`

## Accessing the dataset through synapseclient


```
import synapseclient
# If you have your config file set up you can run:
syn = synapseclient.login()
# Otherwise, pass in your username and password:
syn = synapseclient.login(email='me@example.com', password='secret', rememberMe=True)
```
