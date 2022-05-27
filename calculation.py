import numpy as np
import scipy.signal as scs
import scipy.stats as stats


def mean(x):
    return np.mean(x)


def std(x):
    return np.std(x)


def peaks(x):
    peak_nums, _ = scs.find_peaks(x)
    peaks_phk = len(peak_nums) * 10000 / len(x)
    return int(peaks_phk)


def median(x):
    return np.median(x)


def amin(x):
    return np.amin(x)


def amax(x):
    return np.amax(x)


def skew(x):
    return stats.skew(x)


features = {
    'mean': mean,
    'std': std,
    'skew': skew,
    # 'peaks': peaks,
    'median': median,
    'min': amin,
    'max': amax
}
