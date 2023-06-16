import matplotlib
matplotlib.use('TkAgg')
import os
import matplotlib.pylab as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import warnings
from Classes.Path import Path_info
from Classes.Plotter import Plotter
from scipy.signal import butter, filtfilt
# This code is performing different things
# 1) Load the data from forces plate
# 2) Cut each set
# 3) Cut each rep inside each set by using the markers data (vertical movement)

class Cut_data():

    def __init__(self, subject_ind,imu_data_path):

        self.subject_ind = subject_ind


        path_info = Path_info(subject_ind=self.subject_ind)

        self.path_imu = path_info.data_subject_path_IMU

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

        new_dict_set = {}
        i = 1
        for key, value in zip(self.set.keys(), self.set.values()):
            new_key = i
            new_dict_set[new_key] = self.set[key]
            i=i+1

        for i in new_dict_set.keys():
            name = "IMU_set_" + str(i) + ".npy"
            np.save(os.path.join(path_info.path_IMU_cut, name), new_dict_set[i])

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
        # name = "label_segmented.npy"
        # np.save(os.path.join(path_info.path_IMU_cut, name), self.label)


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

class Get_sets_and_reps():

    def __init__(self, subject_ind,imu_data_path):

        self.subject_ind = subject_ind


        path_info = Path_info(subject_ind=self.subject_ind)

        self.path_IMU = path_info.data_subject_path_IMU
        self.path_points = path_info.path_points
        self.path_markers = path_info.path_markers

        name = "IMU_segmented_arm.npy"  # obtained from sliding window.py, removed rest
        self.IMU_data = np.load(os.path.join(path_info.path_IMU_cut, name))

        self.subject_ind = subject_ind

        path_info = Path_info(subject_ind=self.subject_ind)

        self.path_imu = path_info.data_subject_path_IMU

        self.imu_data_path = imu_data_path

        self.df1 = pd.read_csv(self.imu_data_path,
                               names=['timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'],
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

        self.new_dict_set = {}
        i = 1
        for key, value in zip(self.set.keys(), self.set.values()):
            new_key = i
            self.new_dict_set[new_key] = self.set[key]
            i=i+1

        #for i in new_dict_set.keys():
            #name = "IMU_set_" + str(i) + ".npy"
            #np.save(os.path.join(path_info.path_IMU_cut, name), new_dict_set[i])

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
        # name = "label_segmented.npy"
        # np.save(os.path.join(path_info.path_IMU_cut, name), self.label)
        return self.new_dict_set


    def save_point_rep(self, points,set_ind):

        points = np.array(points)
        points = points.reshape(points.shape[0] // 2, -1)
        print(points)

        name ="_reps_points_set_"+str(set_ind)+".npy"
        np.save(os.path.join(self.path_points, name), points)

        print("Sets points for " + " saved: " + self.path_points + name)

    def load_point_rep(self,set_ind):

        name ="_reps_points_set_"+str(set_ind)+".npy"
        points = np.load(os.path.join(self.path_points, name))

        return points

    def save_point_ecc(self, points,set_ind):

        points = np.array(points)
        points = points.reshape(points.shape[0] // 2, -1)
        print(points)

        name = "_eccs_points_set_" + str(set_ind)+ ".npy"
        np.save(os.path.join(self.path_points, name), points)

        print("Reps points for " + " saved: " + self.path_points + name)

    def save_point_conc(self, points,set_ind):

        points = np.array(points)
        points = points.reshape(points.shape[0] // 2, -1)
        print(points)

        name = "_concs_points_set_" + str(set_ind)+ ".npy"
        np.save(os.path.join(self.path_points, name), points)

        print("Reps points for " + " saved: " + self.path_points + name)

    def save_point_ecc_conc_rep(self, points,set_ind):

        points = np.array(points)
        pointsecc= np.delete(points, -1)
        pointsecc = pointsecc.reshape(pointsecc.shape[0] // 2, -1)
        print(pointsecc)

        name = "_eccs_points_set_" + str(set_ind)+ ".npy"
        np.save(os.path.join(self.path_points, name), pointsecc)

        print("Reps ecc for " + " saved: " + self.path_points + name)

        pointsconc= np.delete(points, 0)
        pointsconc = pointsconc.reshape(pointsconc.shape[0] // 2, -1)
        print(pointsconc)

        name = "_concs_points_set_" + str(set_ind) + ".npy"
        np.save(os.path.join(self.path_points, name), pointsconc)

        print("Reps conc for " + " saved: " + self.path_points + name)

        points = np.array(points)
        middle = np.delete(points,0)
        middle = np.delete(middle, -1)
        middle=middle[1::2]
        m=np.concatenate([middle,middle])
        m=np.sort(m)

        pointsrep = np.insert(m,0,points[0])
        pointsrep = np.insert(pointsrep, -1, points[-1])
        pointsrep=np.sort(pointsrep)

        pointsrep = pointsrep.reshape(pointsrep.shape[0] // 2, -1)
        print(pointsrep)

        name = "_reps_points_set_" + str(set_ind) + ".npy"
        np.save(os.path.join(self.path_points, name), pointsrep)

        print("Sets points for " + " saved: " + self.path_points + name)


######################################
# Plot and click to separate the rep #
######################################

####################
subject_ind = 2
####################

path_info = Path_info(subject_ind=subject_ind)
path_imu = path_info.data_subject_path_IMU
imu_data_path = os.path.join(path_imu, "Participant 00"+str(subject_ind+1)+" Arm.csv")
get_sets_and_reps = Get_sets_and_reps(subject_ind, imu_data_path)

IMU_data =get_sets_and_reps.cut_exercise()
########################
set_ind=27
##########################
IMU_data2 = IMU_data[set_ind]
set_filt = np.empty(IMU_data2.shape)

for col in range(IMU_data2.shape[1]):
     set_filt[:, col] = butter_lowpass_filter(data=IMU_data2[:, col], cutoff=3, fs=100, order=5)
data_to_plot=set_filt

plotter = Plotter(data_to_plot)

get_sets_and_reps.save_point_ecc_conc_rep(points=plotter.points,set_ind=set_ind)




#
# get_sets_and_reps.save_point_rep(points=plotter.points,set_ind=set_ind)
#
# #########################################
# # Load the subset point, plot and click #
# #########################################
#
#
# subject_ind = 0
# #set_ind=4
#
# # get_sets_and_reps = Get_sets_and_reps(subject_ind, imu_data_path)
# plotter = Plotter(data_to_plot)
#
# get_sets_and_reps.save_point_ecc_conc(points=plotter.points,set_ind=set_ind)
#
# self = Get_sets_and_reps(subject_ind, imu_data_path)
# #
# # get_sets_and_reps.save_point_ecc(points=plotter.points,set_ind=set_ind)
# #
# #
# # plotter = Plotter(data_to_plot)
# #
# #
# #
# #
# # get_sets_and_reps.save_point_conc(points=plotter.points,set_ind=set_ind)
# #
