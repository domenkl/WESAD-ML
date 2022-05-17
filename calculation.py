import numpy as np
import scipy.signal as scs


def mean(x):
    return np.mean(x)


def std(x):
    return np.std(x)


def peaks(x):
    peak_nums, _ = scs.find_peaks(x)
    return len(peak_nums)


def median(x):
    return np.median(x)


def amin(x):
    return np.amin(x)


def amax(x):
    return np.amax(x)


features = {
    'mean': mean,
    'std': std,
    'peaks': peaks,
    'median': median,
    'min': amin,
    'max': amax
}
