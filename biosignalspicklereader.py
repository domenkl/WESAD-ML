import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas
import conversion
import os
from multiprocessing import Pool


def calculate_stress_avg(label):
    between = np.logical_and(label > 0, label < 5)
    return label[between].sum() / between.sum()


class BioSignalsReader:

    def __init__(self, sensor=None, path=None, position='chest', sampling_rate=700):
        """ Reads either separate pickle files or
        reads every pickle file starting with name S (as subject) in given directory
        Examples:
            - BioSignalsReader(sensor='ECG', path='data') will loop through all files
            in data directory, get ECG values for each file and save it to ECG_combined.pkl
            - BioSignalsReader(sensor='ECG' path='data/S1.pkl) will read S1.pkl file

        Params:
            - sensor : Name of sensor (ECG, ACC, ...).
            - path : path to file or directory to read, if none it searches in /data/
            - position : Position of sensor (chest or wrist).
            - sampling_rate : Sampling rate of sensors, defaults to 700Hz.

        File structure:
        S1: {
            sensor_signal: [...]  for ECG : [-1.5 mV, 1.5mV]
            stress_level: [...]  [0, 7]
            stress_level_avg: average
        },
        S2: {...}, ...
        """
        self.directory = None
        self.sensor = sensor
        self.position = position
        self.sampling_rate = sampling_rate
        self.subjects = None
        self.all_sensor_data = dict()
        self.avg_stress = dict()
        self.append = False
        self.y_train = None
        self.x_train = np.array([1, 2, 3, 4])

        self.transfer_functions = conversion.transfer_functions
        self.units = conversion.units
        self.ranges = conversion.ranges
        if position == 'wrist':
            raise NotImplementedError

        if path is None:
            path = 'data'
            self.directory = 'data'

        # if directory is given
        if os.path.isdir(path):
            if os.path.isfile('%s/%s_combined.pkl' % (path, sensor)):
                self.__read_combined_file__()
                self.append = True

            files = glob.glob(path + '/*.pkl')
            if len(files) == 0:
                raise FileNotFoundError('No subject files in directory,', path)
            self.__read_all_files__(files)
        # if exact file is given
        elif os.path.isfile(path):
            subject, data = self.__read_file__(path)
            self.all_sensor_data.update({subject: data})
            self.__set_average_stress__()
        else:
            raise ValueError('The path or directory appears to be incorrect.')

    def __read_all_files__(self, files):
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
        """
        returns subject_name, sensor_info
        """
        try:
            # Read pickle file
            pickle_file = pandas.read_pickle(file)
            sensor_info = dict()
            subject_name = pickle_file['subject']
            sensor_signal = pickle_file['signal'][self.position][self.sensor]
            #
            # converted_signal = self.convert_data(sensor_signal, self.sensor)
            sensor_info.update({'sensor_signal': sensor_signal})
            sensor_info.update({'stress_level': pickle_file['label']})
            sensor_info.update({'stress_level_avg': calculate_stress_avg(pickle_file['label'])})
            return subject_name, sensor_info

        except KeyError as e:
            print('No such key found in dictionary', e)
        except BaseException as e:
            print('Error reading file', e)

    def __read_combined_file__(self, path=None):
        """ Reads already combined file
        path : Path to file, has to include whole file name e.g. S2.pkl
        """
        try:
            if path is None or not str(path).endswith('.pkl'):
                path = '%s/%s_combined.pkl' % (self.directory, self.sensor)
            self.all_sensor_data = pandas.read_pickle(path)
            self.subjects = list(self.all_sensor_data.keys())
            self.__set_average_stress__()
        except FileNotFoundError as e:
            print('File with such name not found', e)
        except KeyError:
            print('Mismatched file type.')

    def __convert_data__(self, data, sensor, resolution=16):
        if sensor in self.transfer_functions.keys():
            return self.transfer_functions[sensor](data, resolution)
        return data

    def draw_stress_count_graph(self, subject, show_all=False):
        try:
            if show_all:
                y_values = self.all_sensor_data[subject]['stress_level']
                x_values = np.arange(1, 8)
            else:
                y_values = self.get_stress_levels(subject)
                x_values = np.arange(1, 5)
            plt.bar(x_values, y_values)
            plt.xlabel('Stress levels')
            plt.ylabel(f'Stress level count')
            plt.title(f'Stress level count for subject {subject}')
            plt.show()
        except KeyError as e:
            print('Subject not found', e)

    def draw_stress_graph(self, subject):
        try:
            y_values = self.all_sensor_data[subject]['stress_level']
            size = len(y_values)
            time_vector = np.linspace(0, float(size) / self.sampling_rate, size)
            plt.plot(time_vector, y_values)
            plt.xlabel('Time (s)')
            plt.ylabel('Stress level label')
            plt.title(f'Stress level of subject {subject} over time')
            plt.show()
        except KeyError as e:
            print('Subject not found', e)

    def save_all_graphs(self, g_type='sensor'):
        """ Saves all specified graphs with multiprocessing pools
        g_type : type of graph to save (sensor, stress, all)
        Throws:
            - RuntimeError: if method is not run from __main__
        """
        try:
            self.__check_figure_dir__()
            pool = Pool(2)
            subs = list(self.subjects)
            if g_type == 'sensor':
                pool.map(self.save_sensor_graph, iter(subs,))
            if g_type == 'all':
                pool.map(self.save_subject_graph, iter(subs,))
            if g_type == 'stress':
                raise NotImplementedError
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
        stress_signal = self.all_sensor_data[subject]['stress_level']
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

        ax2.set_ylabel('Stress levels', color='r')
        ax2.plot(time_vector, stress_signal, color='r')
        ax2.axis([interval[0], interval[1], ranges2[0], ranges2[1]])
        ax2.tick_params(axis='y', labelcolor='r')
        plt.grid()
        plt.title('Graph for subject %s' % subject)
        fig.tight_layout()

    def save_subject_graph(self, subject):
        self.__prepare_subject_graph__(subject)
        plt.savefig('figures/%s/%s_subject_graph.png' % (self.sensor, subject))
        plt.close()

    def save_sensor_graph(self, subject):
        self.__prepare_sensor_graph__(subject)
        plt.savefig('figures/%s/%s_sensor_graph.png' % (self.sensor, subject))
        plt.close()

    def draw_sensor_graph(self, subject):
        self.__prepare_sensor_graph__(subject)
        plt.show()

    def draw_subject_graph(self, subject):
        self.__prepare_subject_graph__(subject)
        plt.show()

    def save_to_pickle(self, path=None):
        if path is None:
            path = '%s/%s_combined.pkl' % (self.directory, self.sensor)
        pandas.to_pickle(self.all_sensor_data, path)

    def get_stress_levels(self, subject):
        vals = np.zeros(4)
        values = self.all_sensor_data[subject]['stress_level']
        for x in values:
            if 0 < x < 5:
                vals[x - 1] += 1
        return vals

    def __set_average_stress__(self):
        for subject in self.all_sensor_data:
            self.avg_stress.update({subject: self.all_sensor_data[subject]['stress_level_avg']})

    def __check_figure_dir__(self):
        if not os.path.isdir('figures'):
            os.mkdir('figures')
        directory = 'figures/%s' % self.sensor
        if not os.path.isdir(directory):
            os.mkdir(directory)

    def prepare_train_data(self):
        y_train = list()
        x_train = list()
        for subject in self.all_sensor_data.keys():
            sensor_signal = np.array(self.all_sensor_data[subject]['sensor_signal'], dtype=float)
            stress_level = np.array(self.all_sensor_data[subject]['stress_level'], np.uint8)

            for i in range(1, 5):
                y_train.append(sensor_signal[stress_level == i])
                x_train.append(i)

        # self.y_train = y_train
        # self.x_train = x_train
        # [print(y_train[x].shape for x in y_train)]
        return x_train, y_train

    def prepare_train_data2(self):

        y_train = list([])
        for i in range(1, 5):
            elem = np.array([])
            for subject in self.all_sensor_data.keys():
                sensor_signal = np.array(self.all_sensor_data[subject]['sensor_signal'], dtype=float)
                stress_level = np.array(self.all_sensor_data[subject]['stress_level'], np.uint8)
                elem = np.append(elem, sensor_signal[stress_level == i])

            y_train.append(elem)

        self.y_train = np.array(y_train, dtype=object)
        # [print(y_train[x].shape for x in y_train)]
        return self.y_train
