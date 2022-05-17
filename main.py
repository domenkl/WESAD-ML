import datetime

from biosignalspicklereader import BioSignalsReader

if __name__ == '__main__':
    time = str(datetime.datetime.now())
    file_name = time.split('.')[0].replace(' ', '_').replace(':', '-')
    chest_sensors = ['ACC', 'ECG', 'EDA', 'EMG', 'Temp', 'Resp']
    sensor = ['ECG', 'EDA', 'EMG', 'Temp', 'Resp']
    BSR = BioSignalsReader(sensor=sensor)

    BSR.test_all_combinations()
