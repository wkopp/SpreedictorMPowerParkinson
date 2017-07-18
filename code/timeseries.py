import numpy as np
from matplotlib import pyplot as plt

def fft(ts, plot=False):

    Ts = ts['timestamp'].diff().mean() # mean sampling interval [s]
    Fs = 1.0 / Ts # sampling rate [Hz]
    t = ts['time_in_task'] # time vector
    y = ts['userAcceleration_y'] # signal

    n = len(y)
    k = np.arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[range(n/2)] # one side frequency range

    Y = np.fft.fft(y) / n
    Y = Y[range(n/2)]

    if plot:
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(t,y)
        ax[0].set_xlabel('Time')
        ax[0].set_ylabel('Amplitude')
        ax[1].plot(frq,abs(Y),'r') # plotting the spectrum
        ax[1].set_xlabel('Freq (Hz)')
        ax[1].set_ylabel('|Y(freq)|')

    return frq, Y
