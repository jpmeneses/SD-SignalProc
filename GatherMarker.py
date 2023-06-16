from mpl_toolkits import mplot3d
from scipy.signal import butter, filtfilt
from Classes.Path import Path_info
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd


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
def standardize(data, inds):
    std = []
    mean = []
    dataOut = data.copy()

    for ind in inds:
        std_temp = np.std(data[:, :, ind], ddof=1)
        std.extend([std_temp for i in range(len(ind))])
        mean_temp = np.mean(data[:, :, ind])
        mean.extend([mean_temp for i in range(len(ind))])

    for i in range(dataOut.shape[2]):
        dataOut[:, :, i] = (data[:, :, i] - mean[i]) / std[i]

    dataOut[np.isnan(dataOut.astype(int))] = 0

    return dataOut, mean, std
def standardize_with_mean_and_sd(data, mean, std):
    dataOut = data.copy()

    for i in range(dataOut.shape[2]):
        dataOut[:, :, i] = (data[:, :, i] - mean[i]) / std[i]

    dataOut[np.isnan(dataOut.astype(int))] = 0

    return dataOut

class Gather_data():

    def __init__(self, subject_ind):

        self.subject_ind = subject_ind


        path_info = Path_info(subject_ind=self.subject_ind)

        self.path_marker = path_info.data_subject_path_markers
        name = "Participant "+"{0:03}".format((subject_ind+1))+".csv"
        self.marker_data_path = os.path.join(self.path_marker, name)
        self.df1 = pd.read_csv(self.marker_data_path, skiprows=3)



    def l_arm(self):
        x1 = self.df1['Skeleton 001:LWristOut.3'][3:].to_numpy().astype(np.float)
        y1 = self.df1['Skeleton 001:LWristOut.4'][3:].to_numpy().astype(np.float)
        z1 = self.df1['Skeleton 001:LWristOut.5'][3:].to_numpy().astype(np.float)

        x2 = self.df1['Skeleton 001:LElbowOut.3'][3:].to_numpy().astype(np.float)
        y2 = self.df1['Skeleton 001:LElbowOut.4'][3:].to_numpy().astype(np.float)
        z2 = self.df1['Skeleton 001:LElbowOut.5'][3:].to_numpy().astype(np.float)

        x3 = self.df1['Skeleton 001:LShoulder.4'][3:].to_numpy().astype(np.float)
        y3 = self.df1['Skeleton 001:LShoulder.5'][3:].to_numpy().astype(np.float)
        z3 = self.df1['Skeleton 001:LShoulder.6'][3:].to_numpy().astype(np.float)

        l_lowerarm = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
        l_upperarm = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2 + (z3 - z2) ** 2)

        radius_lowerarm= np.vstack((x1 - x2, y1 - y2, z1 - z2)).T
        radius_upperarm = np.vstack((x2 - x3, y2 - y3, z2 - z3)).T

        path_info = Path_info(subject_ind=self.subject_ind)
        # self.path_markers= path_info.path_markers
        name = "radius_lowerarm_l.npy"
        np.save(os.path.join(path_info.path_markers, name), radius_lowerarm)
        name = "radius_upperarm_l.npy"
        np.save(os.path.join(path_info.path_markers, name), radius_upperarm)

        return radius_lowerarm,radius_upperarm

    def r_arm(self):
        x1 = self.df1['Skeleton 001:RWristOut.3'][3:].to_numpy().astype(np.float)
        y1 = self.df1['Skeleton 001:RWristOut.4'][3:].to_numpy().astype(np.float)
        z1 = self.df1['Skeleton 001:RWristOut.5'][3:].to_numpy().astype(np.float)



        x2 = self.df1['Skeleton 001:RElbowOut.3'][3:].to_numpy().astype(np.float)
        y2 = self.df1['Skeleton 001:RElbowOut.4'][3:].to_numpy().astype(np.float)
        z2 = self.df1['Skeleton 001:RElbowOut.5'][3:].to_numpy().astype(np.float)

        x3 = self.df1['Skeleton 001:RShoulder.4'][3:].to_numpy().astype(np.float)
        y3 = self.df1['Skeleton 001:RShoulder.5'][3:].to_numpy().astype(np.float)
        z3 = self.df1['Skeleton 001:RShoulder.6'][3:].to_numpy().astype(np.float)


        r_lowerarm = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
        r_upperarm = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2 + (z3 - z2) ** 2)

        radius_lowerarm= np.vstack((x1 - x2, y1 - y2, z1 - z2)).T
        radius_upperarm = np.vstack((x2 - x3, y2 - y3, z2 - z3)).T

        # plt.figure()
        # plt.plot(radius_upperarm[:,0])
        # plt.plot(x2)
        # plt.plot(x3)
        # a=x2-x3
        # plt.plot(r_lowerarm)
        # plt.plot(r_upperarm)
        # plt.show()

        path_info = Path_info(subject_ind=self.subject_ind)
        # self.path_markers= path_info.path_markers
        name = "radius_lowerarm_r.npy"
        np.save(os.path.join(path_info.path_markers, name), radius_lowerarm)
        name = "radius_upperarm_r.npy"
        np.save(os.path.join(path_info.path_markers, name), radius_upperarm)

        return radius_lowerarm,radius_upperarm

class Gather_data():

    def __init__(self, subject_ind):

        self.subject_ind = subject_ind


self = Gather_data(subject_ind=7)

class Gather_data():

    def __init__(self, subject_ind):

        self.subject_ind = subject_ind


        path_info = Path_info(subject_ind=self.subject_ind)

        self.path_marker = path_info.data_subject_path_markers
        name = "Participant "+"{0:03}".format((subject_ind+1))+".csv"
        self.marker_data_path = os.path.join(self.path_marker, name)
        self.df1 = pd.read_csv(self.marker_data_path, skiprows=3)



    def l_arm(self):
        x1 = self.df1['Skeleton 001:LWristOut.3'][3:].to_numpy().astype(np.float)
        y1 = self.df1['Skeleton 001:LWristOut.4'][3:].to_numpy().astype(np.float)
        z1 = self.df1['Skeleton 001:LWristOut.5'][3:].to_numpy().astype(np.float)

        x2 = self.df1['Skeleton 001:LElbowOut.3'][3:].to_numpy().astype(np.float)
        y2 = self.df1['Skeleton 001:LElbowOut.4'][3:].to_numpy().astype(np.float)
        z2 = self.df1['Skeleton 001:LElbowOut.5'][3:].to_numpy().astype(np.float)

        x3 = self.df1['Skeleton 001:LShoulder.4'][3:].to_numpy().astype(np.float)
        y3 = self.df1['Skeleton 001:LShoulder.5'][3:].to_numpy().astype(np.float)
        z3 = self.df1['Skeleton 001:LShoulder.6'][3:].to_numpy().astype(np.float)

        l_lowerarm = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
        l_upperarm = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2 + (z3 - z2) ** 2)

        radius_lowerarm= np.vstack((x1 - x2, y1 - y2, z1 - z2)).T
        radius_upperarm = np.vstack((x2 - x3, y2 - y3, z2 - z3)).T

        path_info = Path_info(subject_ind=self.subject_ind)
        # self.path_markers= path_info.path_markers
        name = "radius_lowerarm_l.npy"
        np.save(os.path.join(path_info.path_markers, name), radius_lowerarm)
        name = "radius_upperarm_l.npy"
        np.save(os.path.join(path_info.path_markers, name), radius_upperarm)

        return radius_lowerarm,radius_upperarm

    def r_arm(self):
        x1 = self.df1['Skeleton 001:RWristOut.3'][3:].to_numpy().astype(np.float)
        y1 = self.df1['Skeleton 001:RWristOut.4'][3:].to_numpy().astype(np.float)
        z1 = self.df1['Skeleton 001:RWristOut.5'][3:].to_numpy().astype(np.float)



        x2 = self.df1['Skeleton 001:RElbowOut.3'][3:].to_numpy().astype(np.float)
        y2 = self.df1['Skeleton 001:RElbowOut.4'][3:].to_numpy().astype(np.float)
        z2 = self.df1['Skeleton 001:RElbowOut.5'][3:].to_numpy().astype(np.float)

        x3 = self.df1['Skeleton 001:RShoulder.4'][3:].to_numpy().astype(np.float)
        y3 = self.df1['Skeleton 001:RShoulder.5'][3:].to_numpy().astype(np.float)
        z3 = self.df1['Skeleton 001:RShoulder.6'][3:].to_numpy().astype(np.float)


        r_lowerarm = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
        r_upperarm = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2 + (z3 - z2) ** 2)

        radius_lowerarm= np.vstack((x1 - x2, y1 - y2, z1 - z2)).T
        radius_upperarm = np.vstack((x2 - x3, y2 - y3, z2 - z3)).T

        # plt.figure()
        # plt.plot(radius_upperarm[:,0])
        # plt.plot(x2)
        # plt.plot(x3)
        # a=x2-x3
        # plt.plot(r_lowerarm)
        # plt.plot(r_upperarm)
        # plt.show()

        path_info = Path_info(subject_ind=self.subject_ind)
        # self.path_markers= path_info.path_markers
        name = "radius_lowerarm_r.npy"
        np.save(os.path.join(path_info.path_markers, name), radius_lowerarm)
        name = "radius_upperarm_r.npy"
        np.save(os.path.join(path_info.path_markers, name), radius_upperarm)

        return radius_lowerarm,radius_upperarm


subjects_ind = [i for i in range(33, 36)]
subject_ind = 0

path_info = Path_info(subject_ind=subject_ind)

gather_data = Gather_data(subject_ind=subject_ind)
#radius_lowerarm_l,radius_upperarm_l= gather_data.l_arm()
radius_lowerarm_r,radius_upperarm_r= gather_data.r_arm()


for subject_ind in subjects_ind:
    path_info = Path_info(subject_ind=subject_ind)

    gather_data = Gather_data(subject_ind=subject_ind)
    # radius_lowerarm_l,radius_upperarm_l= gather_data.l_arm()
    radius_lowerarm_r, radius_upperarm_r = gather_data.r_arm()












