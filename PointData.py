# This code is performing different things
# 1) Load the data from forces plate
# 2) Cut each set
# 3) Cut each rep inside each set by using the markers data (vertical movement)

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pylab as plt

from scipy.signal import butter, filtfilt
from sklearn.preprocessing import LabelEncoder

from Classes.Path import Path_info
from Classes.Plotter import Plotter
from utils import butter_lowpass_filter

class Get_sets_and_reps():

    def __init__(self, subject_ind):

        self.subject_ind = subject_ind
        self.path_info = Path_info(subject_ind=self.subject_ind)

        self.path_IMU = self.path_info.data_subject_path_IMU
        self.path_points = self.path_info.path_points
        self.path_markers = self.path_info.path_markers
        self.imu_data_path = os.path.join(self.path_IMU, "Participant "+"{0:03}".format((self.subject_ind+1))+ " Arm.CWA")

        name = "IMU_segmented.npy"  # obtained from sliding window.py, removed rest
        self.IMU_data = np.load(os.path.join(self.path_info.path_IMU_cut, name))

        self.subject_ind = subject_ind

        self.path_imu = self.path_info.data_subject_path_IMU

        # self.df1 = pd.read_csv(self.imu_data_path,
        #                        names=['timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'],
        #                        header=None)
        # self.df1['timestamp'] = pd.to_datetime(self.df1['timestamp'], errors='coerce')
        # path2 = path_info.exercise_list_file

        # self.df2 = pd.read_excel(path2, header=0)

    def get_segment(self):
        name = "IMU_segmented.npy"
        return np.load(os.path.join(self.path_info.path_IMU_cut, name))

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

    def save_point_ecc_conc_rep(self, points, set_ind):

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
subject_ind = 0
####################

get_sets_and_reps = Get_sets_and_reps(subject_ind)

# IMU_data =get_sets_and_reps.cut_exercise()
IMU_data = get_sets_and_reps.get_segment()
########################
set_ind=27
##########################
#IMU_data2 = IMU_data[set_ind]
set_filt = np.empty(IMU_data.shape)

for col in range(IMU_data.shape[1]):
     set_filt[:, col] = butter_lowpass_filter(data=IMU_data[:, col], cutoff=3, fs=100, order=5)
data_to_plot=set_filt

fig = plt.figure(1)
ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)

plotter = Plotter(data_to_plot, fig, ax1, ax2)

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
