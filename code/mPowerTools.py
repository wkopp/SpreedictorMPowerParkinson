import pandas as pd
import numpy as np

def meanTkeo(x):
    y = x.as_matrix()
    y = (y ** 2)[1:-1] - (y[2:] * y[:-2])

    return y.mean()


def zeroCrossingRate(x):
    n = len(x)
    y = np.repeat(1, n)
    y[x <= x.mean()] = -1
    return np.sum((y[:-1] * y[1:]) < 0) / (n - 1.0)

def singleAxisFeatures(x):
    x = x.dropna()
    saf = {
        'meanX' : x.mean(skipna=True),
        'sdX' : x.std(skipna=True),
        #'modeX' : x.mode(), # I don't think this is useful with continuous data like the timeseries
        'skewX' : x.skew(skipna=True),
        'kurX' : x.kurtosis(skipna=True),
        'q1X' : x.quantile(0.25),
        'medianX' : x.median(),
        'q3X' : x.quantile(0.75),
        'iqrX' : x.quantile(0.75) - x.quantile(0.25),
        'rangeX' : x.ptp(),
        'acfX' : x.autocorr(),
        'zcrX' : zeroCrossingRate(x),
        #'dfaX' : tryCatch({
        #    fractal::DFA(x, sum.order = 1)[[1]]
        #}, error = function(err)
        #{NA})
        'cvX' : x.std() / x.mean() * 100,
        'tkeoX' : meanTkeo(x)
        #'lspX' : tryCatch({
        #    lomb::lsp(cbind(tmp_time, x), plot=FALSE)
        #}, error = function(err)
        #{NA})
        #'F0X' : tryCatch({
        #    lspX$peak.at[1]
        #}, error = function(err)
        #{NA})
        #'P0X' : tryCatch({
        #    lspX$peak
        #}, error = function(err)
        #{NA})
        #'lspXF' : tryCatch({
        #    lomb::lsp(cbind(tmp_time, x), plot=FALSE,
        #from = 0.2, to = 5)
        #}, error = function(err)
        #{NA})
        #'F0XF' : tryCatch({
        #    lspXF$peak.at[1]
        #}, error = function(err)
        #{NA})
        #'P0XF' : tryCatch({
        #    lspXF$peak
        #}, error = function(err)
        #{NA})
        #'summaryF0X' : tryCatch({
        #as.numeric(getMedianF0(tmp_time, x))
        #}, error = function(err)
        #{c(NA, NA)})
        #'tlagX' : tryCatch({
        #    tmp_time[fractal::timeLag(x, method='acfdecor')]
        #}, error=function(err)
        #{NA})
        }

    return saf