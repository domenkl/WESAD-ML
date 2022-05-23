import datetime
import glob
import os
import warnings
from itertools import combinations
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier

from calculation import features as feat
import conversion


class BioSignalsReader:

    def __init__(self, sensor='all', path=None, position='chest', sampling_rate=700):
        """ Reads either separate pickle files or reads every pickle file starting with
        name S (as subject) in given directory.
        Note: all subject pkl files must be in the same directory.
        Examples:
            - BioSignalsReader(sensor='ECG', path='data') will loop through all files
            in data directory, get ECG values for each file and save it to ECG_combined.pkl
            - BioSignalsReader(sensor='ECG' path='data/S1.pkl) will read S1.pkl file

        Params:
            - sensor : Name of sensor (ECG, ACC, ...) or list of sensors
            - path : path to file or directory to read, if none it searches in /data/
            - position : Position of sensor (chest or wrist).
            - sampling_rate : Sampling rate of sensors, defaults to 700Hz.

        File structure for combined files:
        S1: {
            sensor_signal: [...]  for ECG : [-1.5 mV, 1.5mV]
            label: [...]  [0, 7] - condition of subject throughout the experiment
            (1 - baseline, 2 - stress, 3 - amusement, 4 - meditation)
        },
        """
        self.directory = None
        self.sensor = sensor
        self.position = position
        self.sampling_rate = sampling_rate
        self.subjects = []
        self.num_of_subjects = 15
        self.all_sensor_data = dict()
        self.subject_labels = dict()
        self.append = False
        self.multiple_sensors = False
        self.chest_sensors = ['ACC', 'ECG', 'EDA', 'EMG', 'Temp', 'Resp']
        self.files = None
        self.feature_matrices = dict()
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
            self.sensor.sort()
            self.multiple_sensors = True
            combined_files = glob.glob(path + '/*_combined.pkl')
            files = glob.glob(path + '/S*.pkl')
            self.files = files
            self.num_of_subjects = len(files)
            if len(combined_files) != 6:
                self.__read_all_files__(files)
                self.__save_to_combined__()

            for s in self.sensor:
                combined = self.__read_combined_file__(s)
                self.all_sensor_data.update({s: combined})
                matrix = self.__prepare_multiple_sensor_matrix(s)
                self.feature_matrices.update({s: matrix})

        # if only one sensor and directory are given
        elif os.path.isdir(path):
            if os.path.isfile('%s/%s_combined.pkl' % (path, sensor)):
                combined = self.__read_combined_file__(sensor)
                self.all_sensor_data = combined
                self.append = True

            files = glob.glob(path + '/S*.pkl')
            if len(files) == 0:
                raise FileNotFoundError('No subject files in directory,', path)
            # self.__read_and_append_files__(files)
        # if only one sensor and exact file are given
        elif os.path.isfile(path):
            subject, data = self.__read_file__(path)
            self.all_sensor_data.update({subject: data})
        else:
            raise ValueError('The path or directory appears to be incorrect.')

    def __save_to_combined__(self):
        """ Saves loaded subject files into sensor combined files
        """
        for sensor in self.chest_sensors:
            subject_data = dict()
            for subject in self.all_sensor_data.keys():
                subject_sensor = dict()
                subject_sensor.update({'sensor_signal': self.all_sensor_data[subject][sensor]})
                subject_sensor.update({'label': self.all_sensor_data[subject]['label']})
                self.all_sensor_data[subject][sensor] = []
                subject_data.update({subject: subject_sensor})
            path = '%s/%s_combined.pkl' % (self.directory, sensor)
            pandas.to_pickle(subject_data, path)
            print(f'Saved {sensor} to combined')
        self.all_sensor_data = dict()

    def __read_matrix_file__(self, sensor):
        file = f'{self.directory}/matrix_{sensor}.pkl'
        return pandas.read_pickle(file)

    def __read_all_files__(self, files):
        for file in files:
            pickle_file = pandas.read_pickle(file)
            subject_name = pickle_file['subject']
            subject_data = dict()
            for sensor in self.chest_sensors:
                sensor_signal = pickle_file['signal'][self.position][sensor]
                subject_data.update({sensor: sensor_signal})

            subject_data.update({'label': pickle_file['label']})
            self.all_sensor_data.update({subject_name: subject_data})
            print(f'Loaded file {file}')

    def __read_and_append_files__(self, files):
        changes = False
        for file in files:
            file_basename = os.path.basename(file)
            file_name = file_basename.split('.')[0]
            if file_name not in self.subjects:
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

    def __read_combined_file__(self, sensor):
        """ Reads already combined file, if only one sensor is given
        path : Path to file, has to include whole file name e.g. S2.pkl
        saves combined file to self.all_sensor_data
        """
        try:
            path = '%s/%s_combined.pkl' % (self.directory, sensor)
            combined_file = pandas.read_pickle(path)
            if len(self.subjects) == 0:
                self.subjects = list(combined_file.keys())
                self.num_of_subjects = len(self.subjects)
            return combined_file
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
        # ranges1 = self.ranges[self.sensor]
        s_min = np.min(sensor_signal)
        s_max = np.max(sensor_signal)
        perc = (s_max - s_min) * 0.1
        s_max = s_max + perc if s_max > 0 else s_max - perc
        s_min = s_min + perc if s_min > 0 else s_min - perc
        ranges1 = [s_min, s_max]
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

    def prepare_feature_matrix(self, sensors):
        if self.multiple_sensors:
            x_matrix = None
            y_matrix = np.tile(np.array([1, 2, 3, 4]), self.num_of_subjects)
            for sensor in sensors:
                feature_matrix = self.feature_matrices[sensor]
                if x_matrix is None:
                    x_matrix = feature_matrix
                else:
                    x_matrix = np.concatenate((x_matrix, feature_matrix), axis=1)
            return x_matrix, y_matrix
        else:
            return self.__prepare_single_sensor_matrix__()

    def __prepare_multiple_sensor_matrix(self, sensor):
        features = feat
        x_matrix = np.zeros((self.num_of_subjects * 4, len(features.keys())))

        for i, subject in enumerate(self.all_sensor_data[sensor].keys()):
            sensor_signal = np.array(self.all_sensor_data[sensor][subject]['sensor_signal'], dtype=float)
            label = np.array(self.all_sensor_data[sensor][subject]['label'], np.uint8)
            for j in range(1, 5):
                sig_interval = sensor_signal[label == j].flatten()
                feats_for_sub = []
                for feature in features:
                    feats_for_sub.append(features[feature](sig_interval))
                x_matrix[i * 4 + (j - 1), :] = feats_for_sub
        print('Finished preparing feature matrix')
        return x_matrix

    def __prepare_single_sensor_matrix__(self):
        """ Prepares feature matrix for loaded sensor
        features are defined in calculation.features
        returns: - x_matrix
            [[mean_S2, std_S2, peaks_S2, median_S2], - for stress value 1
            [mean_S2, std_S2, peaks_S2, median_S2],  - for stress value 2
            [...]] - ...
                y_matrix stress value corresponding to x_matrix row
        """
        num_of_subjects = len(self.all_sensor_data.keys())
        features = feat
        x_matrix = np.zeros((num_of_subjects * 4, len(features.keys())))
        y_matrix = []

        for i, subject in enumerate(self.all_sensor_data.keys()):
            sensor_signal = np.array(self.all_sensor_data[subject]['sensor_signal'], dtype=float)
            label = np.array(self.all_sensor_data[subject]['label'], np.uint8)
            for j in range(1, 5):
                sig_interval = sensor_signal[label == j].flatten()
                feats_for_sub = []
                for feature in features:
                    feats_for_sub.append(features[feature](sig_interval))
                x_matrix[i * 4 + (j - 1), :] = feats_for_sub
                y_matrix.append(j)
        print('Finished preparing feature matrix')
        return x_matrix, np.array(y_matrix)

    def __save_single_sensor_matrix__(self, sensor):
        x_matrix, _ = self.__prepare_single_sensor_matrix__()
        pandas.to_pickle(x_matrix, f'{self.directory}/matrix_{sensor}.pkl')

    def train_and_test_model(self, sensors=None, knn=False):
        if sensors is None:
            sensors = self.sensor
        time = str(datetime.datetime.now())
        x_mat, y_mat = self.prepare_feature_matrix(sensors)
        rs = ShuffleSplit(n_splits=10, test_size=0.3)
        combined_text = ""
        i = 1
        scores = []
        combined_text += f'Tests using sensors {self.sensor}\n'
        combined_text += f'Time of test {time}\n'
        print(combined_text)
        for train_index, test_index in rs.split(x_mat):
            x_train = x_mat[train_index.astype(int)]
            x_test = x_mat[test_index.astype(int)]
            y_train = y_mat[np.array(train_index).astype(int)]
            y_test = y_mat[test_index.astype(int)]
            if knn:
                clf = KNeighborsClassifier()
            else:
                clf = svm.SVC(kernel='linear', C=1)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            acc_score = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            scores.append(acc_score)
            combined_text += f"Test number {i}\n"
            combined_text += f"Accuracy score: {acc_score}\n"
            combined_text += f"Confusion matrix: \n {conf_matrix}\n" \
                             f"--------------------------------\n"
            i += 1
        avg_score = np.average(scores)
        min_score = np.amin(scores)
        combined_text += f"/**********************************/\n" \
                         f"Average accuracy score: {np.average(scores)}\n"
        print('Finished with tests.')
        if not os.path.isdir("results"):
            os.mkdir("results")
        return avg_score, min_score

    def test_all_combinations(self, knn=False):
        combined = ""
        if type(self.sensor) is not list:
            warnings.warn('Sensor parameter is not a list. Testing only for one sensor')
            self.train_and_test_model(knn)
        elif len(self.sensor) == 1:
            self.train_and_test_model(knn)
        else:
            s_combinations = sum(
                [list(map(list, combinations(self.sensor, i))) for i in range(1, len(self.sensor) + 1)], [])
            for combination in s_combinations:
                self.sensor = combination
                avg, amin = self.train_and_test_model(combination)
                combined += f'{combination}\t{int(avg * 100)}\t{int(amin * 100)}\n'
        print(combined)
