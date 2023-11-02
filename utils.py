import os
import numpy as np
import pandas as pd

from openmovement.load import CwaData
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mode

from Classes.Path import Path_info


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, np.ravel(data))
    return y


def load_cwa(filename):
    with CwaData(filename, include_gyro=True, include_temperature=False) as cwa_data:
        samples = cwa_data.get_samples()
        samples = samples.set_index('time')
        samples = samples.astype('float32')
        # samples = samples.resample('50ms').sum()
    return samples


class Cut_data():

    def __init__(self, subject_ind):

        self.subject_ind = subject_ind
        self.path_info = Path_info(subject_ind=self.subject_ind)
        self.path_imu = self.path_info.data_subject_path_IMU
        self.imu_data_path = os.path.join(self.path_imu, "Participant "+"{0:03}".format((self.subject_ind+1))+ " Arm.CWA")

        self.df1 = load_cwa(self.imu_data_path)
        path2 = self.path_info.exercise_list_file

        self.df2 = pd.read_excel(path2, header=0)

    def cut_exercise(self):
        self.set = {}
        label = {}
        m22 = {}
        for i in range(len(self.df2['start_time'])):
            a = pd.to_datetime(self.df2['start_time'][i], errors='coerce')
            b = pd.to_datetime(self.df2['end_time'][i], errors='coerce')
            c = self.df2['exercise'][i]
            if c == 'bent_row' or c == 'lat_raise' or c == 'sh_press':
                m = self.df1[a:b]

                m2 = m.assign(exercise=c)
                m22[i] = m2
                self.set[i] = m.to_numpy() #[:, 1:]  # only IMU signal, delete time
                label[i] = m2.to_numpy()[:, -1]  # only exercise name


        self.IMU_data = np.concatenate([self.set[i] for i in self.set.keys()], axis=0)
        label_data = np.concatenate([label[i] for i in label.keys()], axis=0)
        le = LabelEncoder()
        le.fit(label_data)
        self.label = le.transform(label_data)

        # filter IMU
        self.IMU_filt = self.IMU_data.copy()
        # Filtering the data
        for col in range(self.IMU_data.shape[1]):
            self.IMU_filt[:, col] = butter_lowpass_filter(data=self.IMU_data[:, col], cutoff=20, fs=100, order=5)

        name = "IMU_segmented.npy"
        np.save(os.path.join(self.path_info.path_IMU_cut, name), self.IMU_filt)
        name = "label_segmented.npy"
        np.save(os.path.join(self.path_info.path_IMU_cut, name), self.label)

        return self.IMU_filt

    def split_sequence_IMU(self, n_steps_in=200, n_steps_out=5, lag_value=10):

        data_to_plot = self.IMU_filt
        # split into samples (e.g.in 300, out 200), lag value 1
        X, y = list(), list()
        input_label = list()
        output_label = list()
        i = np.arange(0, len(data_to_plot), lag_value)
        for i in np.arange(0, len(data_to_plot), lag_value):
            # find the end of this pattern
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            # check if we are beyond the dataset
            if out_end_ix > len(data_to_plot):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = data_to_plot[i:end_ix, :], data_to_plot[end_ix:out_end_ix, :]

            X.append(seq_x)
            y.append(seq_y)

        X = np.array(X)
        y = np.array(y)

        name = "IMU_slided.npy"
        np.save(os.path.join(self.path_info.path_IMU_cut, name), X)

        print("IMU slide data saved ")

        return X, y

    def split_sequence_label(self, n_steps_in=200, n_steps_out=5, lag_value=10):

        data_to_plot = self.label[:,np.newaxis]
        # split into samples (e.g.in 300, out 200), lag value 1
        X, y = list(), list()
        input_label = list()
        output_label = list()
        i = np.arange(0, len(data_to_plot), lag_value)
        for i in np.arange(0, len(data_to_plot), lag_value):
            # find the end of this pattern
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            # check if we are beyond the dataset
            if out_end_ix > len(data_to_plot):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = data_to_plot[i:end_ix, :], data_to_plot[end_ix:out_end_ix, :]

            X.append(seq_x)
            y.append(seq_y)

        X = np.array(X)
        y = np.array(y)
        Y_window = mode(X[:, -1, :], axis=1)[0].squeeze()
        Y_window = Y_window[:, np.newaxis]

        #Y_window = to_categorical(Y_window, num_classes=3)

        name = "label_slided.npy"
        np.save(os.path.join(self.path_info.path_IMU_cut, name), Y_window)

        print("Label slide data saved ")

        return X, y, Y_window