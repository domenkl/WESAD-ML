import numpy as np
import scipy.signal as scs


def mean(x):
    return np.mean(x)


def std(x):
    return np.std(x)


def peaks(x):
    return len(scs.find_peaks(x))


def median(x):
    return np.median(x)


features = {
    'mean': mean,
    'std': std,
    'peaks': peaks,
    'median': median
}
