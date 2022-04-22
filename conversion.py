import numpy
import numpy as np


def ecg(samples=None, resolution=16, vcc=3):
    """ Converts ECG sensor data to mV
    [TRANSFER FUNCTION]
    ECG(mV) = (ACC / 2 ** r - 0.5) * VCC

        - ACC   Value sampled from sensor
        - r     Sensor bit resolution
        - VCC   Operating voltage
    """
    samples = _check_samples(samples)
    ecg_samples = [(float(s) / 2 ** resolution - 0.5) * vcc for s in samples]
    return np.asarray(ecg_samples)


def eda(samples=None, resolution=16, vcc=3):
    """ Converts EDA sensor data to µS
    [TRANSFER FUNCTION]
    EDA(µS) = (ACC / 2 ** r) * VCC / 0.12

        - ACC   Value sampled from sensor
        - r     Sensor bit resolution
        - VCC   Operating voltage
    """
    samples = _check_samples(samples)
    eda_samples = [(float(s) / 2 ** resolution) * vcc / 0.12 for s in samples]
    return np.asarray(eda_samples)


def emg(samples=None, resolution=16, vcc=3):
    """ Converts EMG sensor data to mV
    [TRANSFER FUNCTION]
    EMG(mV) = (ACC / 2 ** r - 0.5) * VCC

        - ACC   Value sampled from sensor
        - r     Sensor bit resolution
        - VCC   Operating voltage
    """
    # Same as ECG
    return ecg(samples, resolution, vcc)


def temp(samples=None, resolution=16, vcc=3):
    """ Converts TEMP sensor data to °C
    [TRANSFER FUNCTION]
    TEMP(°C) = (ACC * VCC / (2 ** r - 1)) 

        - ACC   Value sampled from sensor
        - r     Sensor bit resolution
        - VCC   Operating voltage
    """
    samples = _check_samples(samples)
    temp_samples = [(float(s) * vcc / (2 ** resolution - 1)) for s in samples]
    return np.asarray(temp_samples)


def xyz(samples=None, resolution=None, vcc=None):
    """ Converts XYZ sensor data to g
    [TRANSFER FUNCTION]
    XYZ(g) = (ACC - Cmin) / (Cmax - Cmin) * 2.1 

        - ACC   Value sampled from sensor
        - Cmin  28000
        - Cmax  38000
    """
    c_min = 28000
    c_max = 38000
    samples = _check_samples(samples)
    temp_samples = [((float(s) - c_min) / (c_max - c_min)) for s in samples]
    return np.asarray(temp_samples)


def respiration(samples=None, resolution=16, vcc=None):
    """ Converts Respiration sensor data to %
    [TRANSFER FUNCTION]
    RESPIRATION(%) = (ACC / 2 ** r - 0.5) * 100
    
        - ACC   Value sampled from sensor
        - r     Sensor bit resolution
    """
    samples = _check_samples(samples)
    resp_samples = [((float(s) / 2 ** resolution - 0.5) * 100) for s in samples]
    return np.asarray(resp_samples)


def _check_samples(samples=None):
    if samples is None:
        raise TypeError("No samples provided")
    elif type(samples) is list:
        return np.asarray(samples)
    elif type(samples) is numpy.ndarray:
        return np.asarray(samples)
    else:
        raise TypeError("Unsupported sample type")


units = {
    'RAW': '-',
    'ECG': 'mV',
    'EEG': u'µV',
    'EMG': 'mV',
    'EDA': u'µS',
    'ACC': 'g',
    'GLUC': 'mg/dL',
    'Temp': u'°C',
    'NTC': u'°C',
    'LUX': '%',
    'HR': 'bpm',
    'OXI': '%',
    'SYS': 'mmHg',
    'DIA': 'mmHg',
    'EOG': 'mV',
    'EGG': 'mV',
    'Resp': '%',
    'XYZ': 'g'
}

ranges = {
    'RAW': [0, 2 ** 10],
    'ECG': [-1.5, 1.5],
    'EEG': [-41.24, 41.25],
    'EMG': [-1.65, 1.65],
    'EDA': [-4.4, 21],
    'ACC': [-3, 3],
    'GLUC': [20, 600],
    'Temp': [-40, 125],
    'NTC': [0, 50],
    'LUX': [0, 100],
    'HR': [30, 250],
    'OXI': [0, 100],
    'SYS': [0, 300],
    'DIA': [0, 300],
    'EOG': [-0.81, 0.81],
    'EGG': [-0.27, 0.27],
    'Resp': [0, 100],
    'XYZ': [0, 1]
}

transfer_functions = {
    'ECG': ecg,
    'EMG': emg,
    'EDA': eda,
    'XYZ': xyz,
    'Temp': temp,
    'Resp': respiration,
}
