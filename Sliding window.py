from mpl_toolkits import mplot3d
from scipy.signal import butter, filtfilt
import random
from sklearn.preprocessing import LabelEncoder
import matplotlib
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
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
def derivate(data, delta_t):
   d = np.zeros(data.shape[0])
   # Left derivates for the first value
   d[0] = (data[1] - data[0])/delta_t
   # Right derivate for the last value
   d[-1] = (data[-1] - data[-2])/delta_t
   # Centered derivate for everythong else
   for i in range(1, data.shape[0]-1):
       d[i] = (data[i+1] - data[i-1]) / (2*delta_t)
   return d


class Cut_data():

    def __init__(self, subject_ind):

        self.subject_ind = subject_ind


#self = Cut_data(subject_ind=24, imu_data_path=imu_data_path)

class Cut_data():

    def __init__(self, subject_ind,imu_data_path):

        self.subject_ind = subject_ind


        path_info = Path_info(subject_ind=self.subject_ind)

        self.path_imu = path_info.data_subject_path_IMU


        self.imu_data_path0 = os.path.join(self.path_imu, "Participant 004 Arm.csv")
        self.imu_data_path1 = os.path.join(self.path_imu, "Participant 004 Wrist.csv")
        self.imu_data_path2 = os.path.join(self.path_imu, "Participant 004 Trunk.csv")
        self.imu_data_path = imu_data_path

        self.df1 = pd.read_csv(self.imu_data_path, names=['timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'],
                          header=None)
        self.df1['timestamp'] = pd.to_datetime(self.df1['timestamp'], errors='coerce')
        path2 = path_info.exercise_list_file

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
                m = self.df1[(self.df1['timestamp'] >= a) & (self.df1['timestamp'] <= b)]

                m2 = m.assign(exercise=c)
                m22[i] = m2
                self.set[i] = m.to_numpy()[:, 1:]  # only IMU signal, delete time
                label[i] = m2.to_numpy()[:, -1]  # only exercise name
                # name = "IMU_set_"+ str(i) + ".npy"
                # np.save(os.path.join(path_info.path_IMU_cut, name), self.set[i])


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

        # name = "IMU_segmented.npy"
        # np.save(os.path.join(path_info.path_IMU_cut, name), self.IMU_filt)
        name = "label_segmented.npy"
        np.save(os.path.join(path_info.path_IMU_cut, name), self.label)

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

        # name = "IMU_slided.npy"
        # np.save(os.path.join(path_info.path_IMU_cut, name), X)

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
        np.save(os.path.join(path_info.path_IMU_cut, name), Y_window)

        print("Label slide data saved ")

        return X, y, Y_window



##################
# sliding window #
##################
subjects_ind = [i for i in range(36, 36)]
subject_ind = 35

for subject_ind in subjects_ind:
    path_info = Path_info(subject_ind=subject_ind)
    path_imu = path_info.data_subject_path_IMU

    imu_data_path = os.path.join(path_imu, "Participant "+"{0:03}".format((subject_ind+1))+ " Arm.csv")

    cut_data = Cut_data(subject_ind=subject_ind, imu_data_path=imu_data_path)
    IMU_filt=cut_data.cut_exercise()  # save IMU segmented data
    name = "IMU_segmented_Arm.npy"
    np.save(os.path.join(path_info.path_IMU_cut, name), IMU_filt)

    X, X2 = cut_data.split_sequence_IMU(n_steps_in=200, n_steps_out=5, lag_value=200)  # X2 is next window(no used)
    name = "IMU_slided_Arm.npy"
    np.save(os.path.join(path_info.path_IMU_cut, name), X)

    Y, Y2, Y_window = cut_data.split_sequence_label(n_steps_in=200, n_steps_out=5,
                                                    lag_value=200)  # Y2 is next window(no used)
    name = "label_slided_Arm.npy"
    np.save(os.path.join(path_info.path_IMU_cut, name), Y_window)

    #Y_window = mode(Y[:, -1, :], axis=1)[0].squeeze()
    #Y_window = Y_window[:, np.newaxis]

for subject_ind in subjects_ind:
    path_info = Path_info(subject_ind=subject_ind)
    path_imu = path_info.data_subject_path_IMU

    imu_data_path = os.path.join(path_imu, "Participant "+"{0:03}".format((subject_ind+1))+" Wrist.csv")

    cut_data = Cut_data(subject_ind=subject_ind, imu_data_path=imu_data_path)
    IMU_filt=cut_data.cut_exercise()  # save IMU segmented data
    name = "IMU_segmented_Wrist.npy"
    np.save(os.path.join(path_info.path_IMU_cut, name), IMU_filt)

    X, X2 = cut_data.split_sequence_IMU(n_steps_in=200, n_steps_out=5, lag_value=200)  # X2 is next window(no used)
    name = "IMU_slided_Wrist.npy"
    np.save(os.path.join(path_info.path_IMU_cut, name), X)

    Y, Y2, Y_window = cut_data.split_sequence_label(n_steps_in=200, n_steps_out=5,
                                                    lag_value=200)  # Y2 is next window(no used)
    name = "label_slided_Wrist.npy"
    np.save(os.path.join(path_info.path_IMU_cut, name), Y_window)


for subject_ind in subjects_ind:
    path_info = Path_info(subject_ind=subject_ind)
    path_imu = path_info.data_subject_path_IMU

    imu_data_path = os.path.join(path_imu, "Participant "+"{0:03}".format((subject_ind+1))+" Trunk.csv")

    cut_data = Cut_data(subject_ind=subject_ind, imu_data_path=imu_data_path)
    IMU_filt=cut_data.cut_exercise()  # save IMU segmented data
    name = "IMU_segmented_Trunk.npy"
    np.save(os.path.join(path_info.path_IMU_cut, name), IMU_filt)

    X, X2 = cut_data.split_sequence_IMU(n_steps_in=200, n_steps_out=5, lag_value=200)  # X2 is next window(no used)
    name = "IMU_slided_Trunk.npy"
    np.save(os.path.join(path_info.path_IMU_cut, name), X)

    Y, Y2, Y_window = cut_data.split_sequence_label(n_steps_in=200, n_steps_out=5,
                                                    lag_value=200)  # Y2 is next window(no used)
    name = "label_slided_Trunk.npy"
    np.save(os.path.join(path_info.path_IMU_cut, name), Y_window)




##########################################
# get and save all segmented IMU sensor  #
##########################################
imu_data_path = os.path.join(path_imu, "Participant 008 Wrist.csv")
X3, X4 = cut_data.split_sequence_IMU(n_steps_in=200,n_steps_out=5,lag_value =200) #X2 is next window(no used)

imu_data_path = os.path.join(path_imu, "Participant 008 Trunk.csv")
X5, X6 = cut_data.split_sequence_IMU(n_steps_in=200,n_steps_out=5,lag_value =200) #X2 is next window(no used)

X7 = np.concatenate([X,X3],axis=2)
X8 = np.concatenate([X7,X5],axis=2)

name = "IMU_slided.npy"
np.save(os.path.join(path_info.path_IMU_cut, name), X8)

plt.figure()
plt.plot(X[:,:,0])
plt.plot(Y[:,:,0])

