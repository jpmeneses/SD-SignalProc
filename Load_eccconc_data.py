
from Classes.Path import Path_info
from Classes.Segment import Seg_data
from scipy.signal import butter, filtfilt,lfilter
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
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
import matplotlib.pyplot as plt
import warnings


def ecc_conc_set(y, name):
    peak_vel_ecc_set_arm = {}

    for k in y.keys():
        temp = np.concatenate([y[k][i][:, np.newaxis] for i in y[k].keys()], axis=1)
        max = np.max(temp, axis=1)
        peak_vel_ecc_set_arm[k] = max

    peak_vel_ecc_set_arm_data = np.concatenate(
        [peak_vel_ecc_set_arm[i][:, np.newaxis] for i in peak_vel_ecc_set_arm.keys()], axis=1)
    #name = "peak_vel_ecc_set_arm.npy"
    name = name+ '.npy'
    np.save(os.path.join(path_info.path_IMU_cut, name), peak_vel_ecc_set_arm)
    plt.figure()
    plt.plot(peak_vel_ecc_set_arm_data[0, :])
    plt.title(name)
    plt.show()

subjects_ind = [i for i in range(1, 36)]
subject_ind = 3

for subject_ind in subjects_ind:

    path_info = Path_info(subject_ind=subject_ind )
    try:
        name = "peak_vel_ecc_arm.npy"
        peak_vel_ecc_arm = np.load(os.path.join(path_info.path_IMU_cut, name)).item()
        name = "peak_vel_conc_arm.npy"
        peak_vel_conc_arm = np.load(os.path.join(path_info.path_IMU_cut, name)).item()
        name = "mean_vel_ecc_arm.npy"
        mean_vel_ecc_arm = np.load(os.path.join(path_info.path_IMU_cut, name)).item()
        name = "mean_vel_conc_arm.npy"
        mean_vel_conc_arm = np.load(os.path.join(path_info.path_IMU_cut, name)).item()

        name = "peak_vel_ecc_wrist.npy"
        peak_vel_ecc_wrist = np.load(os.path.join(path_info.path_IMU_cut, name)).item()
        name = "peak_vel_conc_wrist.npy"
        peak_vel_conc_wrist = np.load(os.path.join(path_info.path_IMU_cut, name)).item()
        name = "mean_vel_ecc_wrist.npy"
        mean_vel_ecc_wrist = np.load(os.path.join(path_info.path_IMU_cut, name)).item()
        name = "mean_vel_conc_wrist.npy"
        mean_vel_conc_wrist = np.load(os.path.join(path_info.path_IMU_cut, name)).item()

        ecc_conc_set(y=peak_vel_ecc_arm, name='peak_vel_ecc_arm_set')
        ecc_conc_set(y=peak_vel_conc_arm, name='peak_vel_conc_arm_set')
        ecc_conc_set(y=mean_vel_ecc_arm, name='mean_vel_ecc_arm_set')
        ecc_conc_set(y=mean_vel_conc_arm, name='mean_vel_conc_arm_set')
        ecc_conc_set(y=peak_vel_ecc_wrist, name='peak_vel_ecc_wrist_set')
        ecc_conc_set(y=peak_vel_conc_wrist, name='peak_vel_conc_wrist_set')
        ecc_conc_set(y=mean_vel_ecc_wrist, name='mean_vel_ecc_wrist_set')
        ecc_conc_set(y=mean_vel_conc_wrist, name='mean_vel_conc_wrist_set')
    except:
        pass

################################
import matplotlib
matplotlib.use('TkAgg')

path_info = Path_info(subject_ind=3)

name = "peak_vel_ecc_arm.npy"
peak_vel_ecc_arm= np.load(os.path.join(path_info.path_IMU_cut, name)).item()
peak_vel_ecc_arm_data = np.concatenate([peak_vel_ecc_arm[i][k][:,np.newaxis] for i in peak_vel_ecc_arm.keys() for k in peak_vel_ecc_arm[i].keys()], axis=1)
plt.figure()
plt.plot(peak_vel_ecc_arm_data[0,:])# 0: x-axis' 1: y-axis; 2-z-axis;
plt.plot(peak_vel_ecc_arm_data[1,:])
plt.plot(peak_vel_ecc_arm_data[2,:])
a= (peak_vel_ecc_arm_data[0,1]-peak_vel_ecc_arm_data[0,-1])/peak_vel_ecc_arm_data[0,1]
plt.title('peak_vel_ecc_arm_data ')
plt.show()


name = "peak_vel_conc_arm.npy"
peak_vel_conc_arm= np.load(os.path.join(path_info.path_IMU_cut, name)).item()
peak_vel_conc_arm_data = np.concatenate([peak_vel_conc_arm[i][k][:,np.newaxis] for i in peak_vel_conc_arm.keys() for k in peak_vel_conc_arm[i].keys()], axis=1)
plt.figure()
plt.plot(peak_vel_conc_arm_data[0,:])
plt.title('peak_vel_conc_arm_data ')
plt.show()

name = "mean_vel_ecc_arm.npy"
mean_vel_ecc_arm=np.load(os.path.join(path_info.path_IMU_cut, name)).item()
mean_vel_ecc_arm_data = np.concatenate([mean_vel_ecc_arm[i][k][:,np.newaxis] for i in mean_vel_ecc_arm.keys() for k in mean_vel_ecc_arm[i].keys()], axis=1)
plt.figure()
plt.plot(mean_vel_ecc_arm_data[0,:])
plt.title('mean_vel_ecc_arm_data')
plt.show()


name = "mean_vel_conc_arm.npy"
mean_vel_conc_arm=np.load(os.path.join(path_info.path_IMU_cut, name)).item()
mean_vel_conc_arm_data = np.concatenate([mean_vel_conc_arm[i][k][:,np.newaxis] for i in mean_vel_conc_arm.keys() for k in mean_vel_conc_arm[i].keys()], axis=1)
plt.figure()
plt.plot(mean_vel_conc_arm_data[0,:])
plt.title('mean_vel_conc_arm_data')
plt.show()


name = "peak_vel_rep_arm.npy"
peak_vel_rep_arm=np.load(os.path.join(path_info.path_IMU_cut, name)).item()
peak_vel_rep_arm_data = np.concatenate([peak_vel_rep_arm[i][k][:,np.newaxis] for i in peak_vel_rep_arm.keys() for k in peak_vel_rep_arm[i].keys()], axis=1)
plt.figure()
plt.plot(peak_vel_rep_arm_data[0,:])
plt.title('peak_vel_rep_arm_data')
plt.show()

name = "mean_vel_rep_arm.npy"
mean_vel_rep_arm= np.load(os.path.join(path_info.path_IMU_cut, name)).item()
mean_vel_rep_arm_data = np.concatenate([mean_vel_rep_arm[i][k][:,np.newaxis] for i in mean_vel_rep_arm.keys() for k in mean_vel_rep_arm[i].keys()], axis=1)
plt.figure()
plt.plot(mean_vel_rep_arm_data[0,:])
plt.title('mean_vel_rep_arm_data')
plt.show()


name = "peak_vel_set_arm.npy"
peak_vel_set_arm= np.load(os.path.join(path_info.path_IMU_cut, name)).item()
peak_vel_set_arm_data = np.concatenate([peak_vel_set_arm[i][:,np.newaxis] for i in peak_vel_set_arm.keys()], axis=1)
plt.figure()
plt.plot(peak_vel_set_arm_data[0,:])
plt.title('peak_vel_set_arm_data')
plt.show()


name = "mean_vel_set_arm.npy"
mean_vel_set_arm=np.load(os.path.join(path_info.path_IMU_cut, name)).item()
mean_vel_set_arm_data = np.concatenate([mean_vel_set_arm[i][:,np.newaxis] for i in mean_vel_set_arm.keys()], axis=1)
plt.figure()
plt.plot(mean_vel_set_arm_data[0,:])
plt.title('mean_vel_set_arm_data')
plt.show()


    ################# wrist ##############

name = "peak_vel_ecc_wrist.npy"
peak_vel_ecc_wrist=np.load(os.path.join(path_info.path_IMU_cut, name)).item()
peak_vel_ecc_wrist_data = np.concatenate([peak_vel_ecc_wrist[i][k][:,np.newaxis] for i in peak_vel_ecc_wrist.keys() for k in peak_vel_ecc_wrist[i].keys()], axis=1)
plt.figure()
plt.plot(peak_vel_ecc_wrist_data[0,:])
plt.title('peak_vel_ecc_wrist_data')
plt.show()

name = "peak_vel_conc_wrist.npy"
peak_vel_conc_wrist=np.load(os.path.join(path_info.path_IMU_cut, name)).item()
peak_vel_conc_wrist_data = np.concatenate([peak_vel_conc_wrist[i][k][:,np.newaxis] for i in peak_vel_conc_wrist.keys() for k in peak_vel_conc_wrist[i].keys()], axis=1)
plt.figure()
plt.plot(peak_vel_conc_wrist_data[0,:])
plt.title('peak_vel_conc_wrist_data')
plt.show()

name = "mean_vel_ecc_wrist.npy"
mean_vel_ecc_wrist=np.load(os.path.join(path_info.path_IMU_cut, name)).item()
mean_vel_ecc_wrist_data = np.concatenate([mean_vel_ecc_wrist[i][k][:,np.newaxis] for i in mean_vel_ecc_wrist.keys() for k in mean_vel_ecc_wrist[i].keys()], axis=1)
plt.figure()
plt.plot(mean_vel_ecc_wrist_data[0,:])
plt.title('mean_vel_ecc_wrist_data')
plt.show()

name = "mean_vel_conc_wrist.npy"
mean_vel_conc_wrist=np.load(os.path.join(path_info.path_IMU_cut, name)).item()
mean_vel_conc_wrist_data = np.concatenate([mean_vel_conc_wrist[i][k][:,np.newaxis] for i in mean_vel_conc_wrist.keys() for k in mean_vel_conc_wrist[i].keys()], axis=1)
plt.figure()
plt.plot(mean_vel_conc_wrist_data[0,:])
plt.title('mean_vel_conc_wrist_data')
plt.show()

name = "peak_vel_rep_wrist.npy"
peak_vel_rep_wrist=np.load(os.path.join(path_info.path_IMU_cut, name)).item()
peak_vel_rep_wrist_data = np.concatenate([peak_vel_rep_wrist[i][k][:,np.newaxis] for i in peak_vel_rep_wrist.keys() for k in peak_vel_rep_wrist[i].keys()], axis=1)
plt.figure()
plt.plot(peak_vel_rep_wrist_data[0,:])
plt.title('peak_vel_rep_wrist_data')
plt.show()

name = "mean_vel_rep_wrist.npy"
mean_vel_rep_wrist=np.load(os.path.join(path_info.path_IMU_cut, name)).item()
mean_vel_rep_wrist_data = np.concatenate([mean_vel_rep_wrist[i][k][:,np.newaxis] for i in mean_vel_rep_wrist.keys() for k in mean_vel_rep_wrist[i].keys()], axis=1)
plt.figure()
plt.plot(mean_vel_rep_wrist_data[0,:])
plt.title('mean_vel_rep_wrist_data')
plt.show()


name = "peak_vel_set_wrist.npy"
peak_vel_set_wrist=np.load(os.path.join(path_info.path_IMU_cut, name)).item()
peak_vel_set_wrist_data = np.concatenate([peak_vel_set_wrist[i][:,np.newaxis] for i in peak_vel_set_wrist.keys()], axis=1)
plt.figure()
plt.plot(peak_vel_set_wrist_data[0,:])
plt.title('peak_vel_set_wrist_data')
plt.show()


name = "mean_vel_set_wrist.npy"
mean_vel_set_wrist = np.load(os.path.join(path_info.path_IMU_cut, name)).item()
mean_vel_set_wrist_data = np.concatenate([mean_vel_set_wrist[i][:,np.newaxis] for i in mean_vel_set_wrist.keys()], axis=1)
plt.figure()
plt.plot(mean_vel_set_wrist_data[0,:])
plt.title('mean_vel_set_wrist_data')
plt.show()