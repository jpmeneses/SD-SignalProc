from scipy.signal import butter, filtfilt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks


subjects_ind = [i for i in range(1, 7)]
subject_ind = 3

path_info = Path_info(subject_ind=subject_ind)
path_imu = path_info.data_subject_path_IMU


name = "linear_velocity.npy"
vel = np.load(os.path.join(path_info.path_IMU, name))
np.savetxt("Participant_004_linear_velocity.csv", vel, delimiter=",")

# save rep data
subject_ind = 3
path_info = Path_info(subject_ind=subject_ind)
path_imu = path_info.data_subject_path_IMU

name = "IMU_data_rep_linear_vel.npy"
IMU_data_rep=np.load(os.path.join(path_info.path_IMU_cut, name)).item()

for k in IMU_data_rep.keys():
    a = IMU_data_rep[k]
    for i in a.keys():
        rep = a[i]
        name = os.path.join("Participant_004_linear_vel_set_"+str(k)+"_rep_"+str(i)+".csv")
        np.savetxt(name, rep, delimiter=",")




a=pd.DataFrame(IMU_data_rep[1], index =IMU_data_rep[1].keys())
pd.DataFrame(IMU_data_rep).to_csv('Participant_004_linear_velocity_repetitions.csv', index=False)


