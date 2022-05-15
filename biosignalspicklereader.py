import glob
import os
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas

from calculation import features as feat
import conversion


class BioSignalsReader:

    def __init__(self, sensor='all', path=None, position='chest', sampling_rate=700):
        """ Reads either separate pickle files or reads every pickle file starting with name S (as subject) in given
        directory. Note: all subject pkl files must be in the same directory. Examples: - BioSignalsReader(
        sensor='ECG', path='data') will loop through all files in data directory, get ECG values for each file and
        save it to ECG_combined.pkl - BioSignalsReader(sensor='ECG' path='data/S1.pkl) will read S1.pkl file

        Params:
            - sensor : Name of sensor (ECG, ACC, ...) or list of sensors
            - path : path to file or directory to read, if none it searches in /data/
            - position : Position of sensor (chest or wrist).
            - sampling_rate : Sampling rate of sensors, defaults to 700Hz.

        File structure for combined files:
        S1: {
            - sensor_signal: [...]  for ECG : [-1.5 mV, 1.5mV]
            - label: [...]  [0, 7] - condition of subject throughout the experiment
            (1 - baseline, 2 - stress, 3 - amusement, 4 - meditation)

        },
        S2: {...}, ...
        Otherwise if multiple sensors are given all_sensor_data structure is following:
        ECG: {
            - S1: [...],
            - S2: [...]
        },
        EDA : {
            - S1: [...],
            - S2: [...]
        }
        subject condition is saved then in self.subject_labels
        """
        self.directory = None
        self.sensor = sensor
        self.position = position
        self.sampling_rate = sampling_rate
        self.subjects = []
        self.num_of_subjects = None
        self.all_sensor_data = dict()
        self.subject_labels = dict()
        self.append = False
        self.multiple_sensors = False
        self.chest_sensors = ['ACC', 'ECG', 'EDA', 'EMG', 'Temp', 'Resp']
        self.units = conversion.units
        self.ranges = conversion.ranges

        if position == 'wrist':
            raise NotImplementedError

        if path is None:
            path = 'data'
            self.directory = 'data'

        if self.sensor == 'all':
            self.sensor = self.chest_sensors

        # if more than one sensor is given
        if type(self.sensor) is list and len(self.sensor) > 1:
            self.multiple_sensors = True
            files = glob.glob(path + '/S*.pkl')
            self.num_of_subjects = len(files)
            if self.num_of_subjects == 0:
                raise FileNotFoundError('Make sure to put all subject pkl files into directory', path)
            self.__read_all_files__(files)
        # if only one sensor and directory are given
        elif os.path.isdir(path):
            if os.path.isfile('%s/%s_combined.pkl' % (path, sensor)):
                self.__read_combined_file__()
                self.append = True

            files = glob.glob(path + '/*.pkl')
            if len(files) == 0:
                raise FileNotFoundError('No subject files in directory,', path)
            self.__read_and_append_files__(files)
        # if only one sensor and exact file are given
        elif os.path.isfile(path):
            subject, data = self.__read_file__(path)
            self.all_sensor_data.update({subject: data})
        else:
            raise ValueError('The path or directory appears to be incorrect.')

    def __read_all_files__(self, files):
        for sensor in self.sensor:
            self.all_sensor_data.update({sensor: dict()})

        for file in files:
            pickle_file = pandas.read_pickle(file)

            for sensor in self.sensor:
                sensor_info = dict()

                subject_name = pickle_file['subject']
                self.subject_labels.update({subject_name: pickle_file['label']})

                sensor_signal = pickle_file['signal'][self.position][sensor]
                sensor_info.update({subject_name: sensor_signal})

                self.all_sensor_data[sensor].update(sensor_info)
        print('Finished reading all subject files.')
        print(self.all_sensor_data)

    def __read_and_append_files__(self, files):
        changes = False
        for file in files:
            file_basename = os.path.basename(file)
            file_name = file_basename.split('.')[0]
            if file_name not in self.subjects and file_name.startswith('S'):
                changes = True
                print('Appending %s subject to %s' % (file_name, self.sensor))
                subject, data = self.__read_file__(file)
                self.all_sensor_data.update({subject: data})
        self.subjects = list(self.all_sensor_data.keys())

        # if new data was added
        if changes:
            self.save_to_pickle()
            print('Saved combined file to pickle')

    def __read_file__(self, file):
        """ Reads given pickle file, takes data for only one sensor
        returns subject_name, sensor_info
        """
        try:
            # Read pickle file
            pickle_file = pandas.read_pickle(file)
            sensor_info = dict()
            subject_name = pickle_file['subject']
            sensor_signal = pickle_file['signal'][self.position][self.sensor]

            sensor_info.update({'sensor_signal': sensor_signal})
            sensor_info.update({'label': pickle_file['label']})
            return subject_name, sensor_info

        except KeyError as e:
            print('No such key found in dictionary', e)
        except BaseException as e:
            print('Error reading file', e)

    def __read_combined_file__(self, path=None):
        """ Reads already combined file, if only one sensor is given
        path : Path to file, has to include whole file name e.g. S2.pkl
        """
        try:
            if path is None or not str(path).endswith('.pkl'):
                path = '%s/%s_combined.pkl' % (self.directory, self.sensor)
            self.all_sensor_data = pandas.read_pickle(path)
            self.subjects = list(self.all_sensor_data.keys())
        except FileNotFoundError as e:
            print('File with such name not found', e)
        except KeyError:
            print('Mismatched file structure.')

    def draw_label_graph(self, subject):
        try:
            y_values = self.all_sensor_data[subject]['label']
            if self.multiple_sensors:
                y_values = self.subject_labels[subject]

            size = len(y_values)
            time_vector = np.linspace(0, float(size) / self.sampling_rate, size)
            plt.plot(time_vector, y_values)
            plt.xlabel('Time (s)')
            plt.ylabel('Condition id')
            plt.title(f'Condition of subject {subject} over time')
            plt.show()
        except KeyError as e:
            print('Subject not found', e)

    def save_all_graphs(self, g_type='sensor', subjects=None):
        """ Saves all specified graphs with multiprocessing pools
        g_type : type of graph to save (sensor, stress, all)
        Throws:
            - RuntimeError: if method is not run from __main__
        """
        try:
            if self.multiple_sensors:
                raise NotImplementedError('Not implemented for multiple sensors')
            self.__check_figure_dir__()
            pool = Pool(2)
            if subjects is None:
                subjects = list(self.subjects)

            if g_type == 'sensor':
                pool.map(self.save_sensor_graph, iter(subjects, ))
            elif g_type == 'all':
                pool.map(self.save_subject_graph, iter(subjects, ))
        except RuntimeError:
            print('Make sure you run *save_graphs* method from if __name__ == \'__main__\'')

    def __prepare_sensor_graph__(self, subject):
        sensor_signal = self.all_sensor_data[subject]['sensor_signal']
        size = len(sensor_signal)
        ranges = self.ranges[self.sensor]
        time_vector = np.linspace(0, float(size) / self.sampling_rate, size)
        interval = [0, time_vector[-1]]
        plt.title(f'{self.sensor} sensor for subject {subject}')
        plt.plot(time_vector, sensor_signal)
        plt.axis([interval[0], interval[1], ranges[0], ranges[1]])
        plt.xlabel('Time (s)')
        plt.ylabel('%s sensor (%s)' % (self.sensor, self.units[self.sensor]))
        plt.grid()

    def __prepare_subject_graph__(self, subject):
        sensor_signal = self.all_sensor_data[subject]['sensor_signal']
        condition_label = self.all_sensor_data[subject]['label']
        size = len(sensor_signal)
        time_vector = np.linspace(0, float(size) / self.sampling_rate, size)
        ranges1 = self.ranges[self.sensor]
        ranges2 = [0, 7]
        interval = [0, time_vector[-1]]

        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('%s sensor (%s)' % (self.sensor, self.units[self.sensor]), color='g')
        ax1.plot(time_vector, sensor_signal, color='g')
        ax1.tick_params(axis='y', labelcolor='g')
        ax1.axis([interval[0], interval[1], ranges1[0], ranges1[1]])

        ax2 = ax1.twinx()

        ax2.set_ylabel('Condition id', color='r')
        ax2.plot(time_vector, condition_label, color='r')
        ax2.axis([interval[0], interval[1], ranges2[0], ranges2[1]])
        ax2.tick_params(axis='y', labelcolor='r')
        plt.grid()
        plt.title('Graph for subject %s' % subject)
        fig.tight_layout()

    def save_subject_graph(self, subject):
        if self.multiple_sensors:
            raise NotImplementedError('Not implemented for multiple sensors')
        self.__prepare_subject_graph__(subject)
        plt.savefig('figures/%s/%s_subject_graph.png' % (self.sensor, subject))
        plt.close()

    def save_sensor_graph(self, subject):
        if self.multiple_sensors:
            raise NotImplementedError('Not implemented for multiple sensors')
        self.__prepare_sensor_graph__(subject)
        plt.savefig('figures/%s/%s_sensor_graph.png' % (self.sensor, subject))
        plt.close()

    def draw_sensor_graph(self, subject):
        if self.multiple_sensors:
            raise NotImplementedError('Not implemented for multiple sensors')
        self.__prepare_sensor_graph__(subject)
        plt.show()

    def draw_subject_graph(self, subject):
        if self.multiple_sensors:
            raise NotImplementedError('Not implemented for multiple sensors')
        self.__prepare_subject_graph__(subject)
        plt.show()

    def save_to_pickle(self, path=None):
        if path is None:
            path = '%s/%s_combined.pkl' % (self.directory, self.sensor)
        pandas.to_pickle(self.all_sensor_data, path)

    def __check_figure_dir__(self):
        if not os.path.isdir('figures'):
            os.mkdir('figures')
        directory = 'figures/%s' % self.sensor
        if not os.path.isdir(directory):
            os.mkdir(directory)

    def prepare_feature_matrix(self):
        """ Prepares feature matrix for given sensor
        features are defined in calculation.features
        returns: - x_matrix
            [[mean_S2, std_S2, peaks_S2, median_S2], - for stress value 1
            [mean_S2, std_S2, peaks_S2, median_S2],  - for stress value 2
            [...]] - ...
                y_matrix stress value corresponding to x_matrix row
        """
        num_of_subjects = len(self.all_sensor_data.keys())
        features = feat
        x_matrix = np.zeros((num_of_subjects * len(features.keys()), 4))
        y_matrix = []

        for i, subject in enumerate(self.all_sensor_data.keys()):
            sensor_signal = np.array(self.all_sensor_data[subject]['sensor_signal'], dtype=float)
            stress_level = np.array(self.all_sensor_data[subject]['stress_level'], np.uint8)
            for j in range(1, 5):
                sig_interval = sensor_signal[stress_level == j].flatten()
                feats_for_sub = []
                for feature in features:
                    feats_for_sub.append(features[feature](sig_interval))
                x_matrix[i * 4 + (j - 1), :] = feats_for_sub
                y_matrix.append(j)
        return x_matrix, np.array(y_matrix)
