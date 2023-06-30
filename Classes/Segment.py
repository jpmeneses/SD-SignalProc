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
from scipy.misc import electrocardiogram
from scipy.stats import mode
from scipy.signal import find_peaks

class Seg_data():

    def __init__(self, subject_ind,imu_data_path):

        self.subject_ind = subject_ind


        path_info = Path_info(subject_ind=self.subject_ind)

        self.path_imu = path_info.data_subject_path_IMU
        self.path_points = path_info.path_points


        self.imu_data_path0 = os.path.join(self.path_imu, "Participant 00"+str(self.subject_ind+1)+" Arm.csv")
        self.imu_data_path1 = os.path.join(self.path_imu, "Participant 00"+str(self.subject_ind+1)+" Wrist.csv")
        self.imu_data_path2 = os.path.join(self.path_imu, "Participant 00"+str(self.subject_ind+1)+" Trunk.csv")
        self.imu_data_path = imu_data_path

        self.df1 = pd.read_csv(self.imu_data_path, names=['timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'],
                          header=None)
        self.df1['timestamp'] = pd.to_datetime(self.df1['timestamp'], errors='coerce')
        path2 = path_info.exercise_list_file
        self.df2 = pd.read_excel(path2, header=0)

        # self.path22=os.path.join(self.path_imu, "Timing_sheet_conc_ecc.xlsx")
        # self.df22 = pd.read_excel(self.path22, header=0)
        # self.df22['start_time']=pd.to_datetime(self.df22['start_time'], format = '%f %p')
        # self.df22['end_time'] = pd.to_datetime(self.df22['end_time'], format='%f %p')

        name = "linear_velocity_arm.npy"
        self.vel = np.load(os.path.join(path_info.path_IMU, name))
        name = "linear_velocity_wrist.npy"
        self.vel2 = np.load(os.path.join(path_info.path_IMU, name))
        #plt.figure()
        #plt.plot(self.vel[:,0])
        #plt.show()

    def filter_data(self, freq):
        # segmentation set and add exercise label
        self.set = {}
        label = {}
        m22 = {}
        path_info = Path_info(subject_ind=self.subject_ind)
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
                # name = "IMU_set_" + str(i) + ".npy"
                # np.save(os.path.join(path_info.path_IMU_cut, name), self.set[i])

        new_dict_set = {}
        i = 1
        for key, value in zip(self.set.keys(), self.set.values()):
            new_key = i
            new_dict_set[new_key] = self.set[key]
            i = i + 1
        self.set=new_dict_set

        self.IMU_data = np.concatenate([self.set[i] for i in self.set.keys()], axis=0)
        # plt.figure()
        # plt.plot(self.set[1][:,0])
        # plt.show()

        self.set_filt = {}
        for k in self.set.keys():
            self.set_filt[k] = self.set[k].copy()

            try:
                # Filtering the data
                for col in range(self.set[k].shape[1]):
                    self.set_filt[k][:, col] = butter_lowpass_filter(data=self.set[k][:, col], cutoff=freq, fs=100,
                                                                     order=5)
            except:
                 pass



        #plt.figure()
        #plt.plot(self.set_filt[4][:,1])
        #plt.show()

        return self.set_filt, self.set

    def seg_rep_arm(self, y):
        # cut each set into reps
        #####################
        # find peak values  #
        #####################
        lag_value = 1

        n_steps_in = 10
        data_to_plot = y
        seq_x = {}
        peak_ix = {}
        i = 0
        seq_x[0] = data_to_plot[0:n_steps_in]
        for i in np.arange(1, len(data_to_plot), lag_value):
            # find the end of this pattern
            end_ix = i + n_steps_in
            # check if we are beyond the dataset
            if end_ix > len(data_to_plot):
                break
            # gather parts of the pattern
            seq_x[i] = data_to_plot[0:end_ix]
            mean = np.mean(data_to_plot)
            # if reach the peak value
            if seq_x[i][-1] > seq_x[i - 1][-1]:
                peak_ix[i - 1] = end_ix
                continue

        ix = {}
        key = [*peak_ix]
        k=80
        for k in range(len(key)):
            if k == len(key) - 1:
                break
            a = key[k + 1] - key[k]
            if a > 1:
                ix[k] = key[k]
        k = len(key) - 1
        ix[k]=key[-1]

        fig, ax = plt.subplots()
        ax.plot(y)
        end = {}
        for kk in ix.keys():
            a = ix[kk]
            end[kk] = peak_ix[a]
            ax.axvline(x=end[kk], color='orange')

        ax.axvline(x=len(data_to_plot), color='orange')
        ##ax.axvline(x=peak_ix[119], color='orange')
        ##plt.show()

        #####################
        # fake peak removal #
        #####################
        rep_point = list(end.values())
        rep=rep_point
        w = data_to_plot.shape[0]
        w = np.int32(w)
        rep.append(w)
        rep = [0] + rep
        rep = np.unique(rep)
        rep = rep.tolist()


        repp=rep.copy()

        for i in range(len(rep)):
            if i == len(rep) - 1:
                break
            beg = rep[i]
            end = rep[i + 1]
            p = len(rep) * 1.5
            if end - beg < len(data_to_plot) / p:
                repp.remove(end)

        reppp = repp.copy()
        for i in range(len(repp)):
            if i == len(repp) - 1:
                break
            beg = repp[i]
            end = repp[i + 1]
            if data_to_plot[end-1]< 0:
                reppp.remove(end)

        repppp = reppp.copy()
        for i in range(len(reppp)):
            if i == len(reppp) - 1:
                break
            beg = reppp[i]
            end = reppp[i + 1]
            p = len(reppp) * 1.5
            if end - beg < len(data_to_plot) / p:
                repppp.remove(end)



        repp=repppp
        beg = {}
        end = {}
        fig, ax = plt.subplots()
        ax.plot(y)
        for i in range(len(repp)):
            if i == len(repp) - 1:
                break
            beg[i] = repp[i]
            end[i] = repp[i + 1]
            plt.axvline(beg[i],color='red')
            plt.axvline(end[i],color='orange')

        plt.show()

        return beg, end

    def seg_rep22(self, y):
        # cut each set into phrases

        ################################
        # find peak and valley values  #
        ################################

        lag_value = 1

        n_steps_in = 10
        data_to_plot = y
        seq_x = {}
        peak_ix = {}
        low_ix={}
        i = 0
        seq_x[0] = data_to_plot[0:n_steps_in]
        for i in np.arange(1, len(data_to_plot), lag_value):
            # find the end of this pattern
            end_ix = i + n_steps_in
            # check if we are beyond the dataset
            if end_ix > len(data_to_plot):
                break
            # gather parts of the pattern
            seq_x[i] = data_to_plot[0:end_ix]
            mean = np.mean(data_to_plot)
            # if reach the peak value
            if seq_x[i][-1] > seq_x[i - 1][-1]:
                peak_ix[i - 1] = end_ix
                continue
            if seq_x[i][-1] < seq_x[i - 1][-1]:
                low_ix[i - 1] = end_ix
                continue

        ix = {}
        key = [*peak_ix]
        for k in range(len(key)):
            if k == len(key) - 1:
                break
            a = key[k + 1] - key[k]
            if a > 1:
                ix[k] = key[k]
        k = len(key)-1
        ix[k]=key[-1]

        ix_low = {}
        key_low = [*low_ix]
        for k in range(len(key_low)):
            if k == len(key_low) - 1:
                break
            a = key_low[k + 1] - key_low[k]
            if a > 1:
                ix_low[k] = key_low[k]
        k = len(key_low) - 1
        ix_low[k] = key_low[-1]

        fig, ax = plt.subplots()
        ax.plot(y)
        end = {}
        for kk in ix.keys():
            a = ix[kk]
            end[kk] = peak_ix[a]
            ax.axvline(x=end[kk], color='orange')

        #ax.axvline(x=len(data_to_plot), color='orange')
        ##ax.axvline(x=peak_ix[119], color='orange')
        ##plt.show()

        ################################
        # fake peak and valley removal #
        ################################

        rep_point = list(end.values())
        rep = rep_point
        repp = rep.copy()
        for i in range(len(rep)):
            if i == len(rep) - 1:
                break
            beg = rep[i]
            end = rep[i + 1]
            p = len(rep) * 1.5
            if end - beg < len(data_to_plot) / p:
                repp.remove(end)
        rep_peak = repp


        fig, ax = plt.subplots()
        ax.plot(y)
        end_low = {}
        for kk in ix_low.keys():
            a = ix_low[kk]
            end_low[kk] = low_ix[a]
            ax.axvline(x=end_low[kk], color='orange')

        # ax.axvline(x=len(data_to_plot), color='orange')
        ##ax.axvline(x=peak_ix[119], color='orange')
        ##plt.show()

        rep_point = list(end_low.values())
        rep = rep_point
        repp = rep.copy()

        for i in range(len(rep)):
            if i == len(rep) - 1:
                break
            beg = rep[i]
            end = rep[i + 1]
            p = len(rep) * 1.5
            if end - beg < len(data_to_plot) / p:
                repp.remove(end)

        rep_low=repp
        rep=rep_peak+rep_low
        rep=sorted(rep)


        w = data_to_plot.shape[0]
        w = np.int32(w)
        rep.append(w)
        rep = [0] + rep
        rep = np.unique(rep)
        rep=rep.tolist()
        repp = rep.copy()
        for i in range(len(rep)):
            if i == len(rep) - 1:
                break
            beg = rep[i]
            end = rep[i + 1]
            p = len(rep) * 1.5
            if end - beg < len(data_to_plot) / p:
                repp.remove(end)

        beg = {}
        end = {}
        plt.plot(y)
        for i in range(len(repp)):
            if i == len(repp) - 1:
                break
            beg[i] = repp[i]
            end[i] = repp[i + 1]
            plt.axvline(beg[i], color='red')
            plt.axvline(end[i], color='orange')

        plt.show()

        return beg, end

    def set_velocity(self, freq):
        # divide continuous velocity into sets
        self.set_vel={}
        k=1
        start = 0
        fig, ax = plt.subplots()
        ax.plot(self.vel[:,0])
        for k in self.set.keys():
            rep = len(self.set[k])
            end = start+rep
            self.set_vel[k] = self.vel[start:end]
            ax.axvline(x=start,color="orange")
            ax.axvline(x=end, color="green")
            plt.show()
            start = end

        self.set_filt2 = {}
        for k in self.set_vel.keys():
            self.set_filt2[k] = self.set_vel[k].copy()
            try:
                # Filtering the data
                for col in range(self.set_vel[k].shape[1]):
                    self.set_filt2[k][:, col] = butter_lowpass_filter(data=self.set_vel[k][:, col], cutoff=freq, fs=100,
                                                                      order=5)
            except:
                pass
        return self.set_vel,self.set_filt2

    def set_velocity2(self, freq):
        # divide continuous velocity into sets
        self.set_vel={}
        k=1
        start = 0
        fig, ax = plt.subplots()
        ax.plot(self.vel2[:,0])
        for k in self.set.keys():
            rep = len(self.set[k])
            end = start+rep
            self.set_vel[k] = self.vel2[start:end]
            ax.axvline(x=start,color="orange")
            ax.axvline(x=end, color="green")
            plt.show()
            start = end

        self.set_filt2 = {}
        for k in self.set_vel.keys():
            self.set_filt2[k] = self.set_vel[k].copy()
            try:
                # Filtering the data
                for col in range(self.set_vel[k].shape[1]):
                    self.set_filt2[k][:, col] = butter_lowpass_filter(data=self.set_vel[k][:, col], cutoff=freq, fs=100,
                                                                      order=5)
            except:
                 pass

        return self.set_vel,self.set_filt2



'''
    def align_sensor(self):
        self.temp0 = pd.read_csv(self.imu_data_path0,
                               names=['timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'],
                               header=None)
        self.temp0['timestamp'] = pd.to_datetime(self.temp0['timestamp'], errors='coerce')
        self.temp1 = pd.read_csv(self.imu_data_path1,
                                 names=['timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'],
                                 header=None)
        self.temp1['timestamp'] = pd.to_datetime(self.temp1['timestamp'], errors='coerce')
        self.temp2 = pd.read_csv(self.imu_data_path2,
                                 names=['timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'],
                                 header=None)
        self.temp2['timestamp'] = pd.to_datetime(self.temp2['timestamp'], errors='coerce')

        a = max(self.temp0['timestamp'].iloc[0],self.temp1['timestamp'].iloc[0],self.temp2['timestamp'].iloc[0])
        b = min(self.temp0['timestamp'].iloc[-1], self.temp1['timestamp'].iloc[-1], self.temp2['timestamp'].iloc[-1])
        self.temp_arm = self.temp0[(self.temp0['timestamp']>=a) & (self.temp0['timestamp']<=b)]
        self.temp_wrist = self.temp1[(self.temp1['timestamp'] >=a) & (self.temp1['timestamp'] <= b)]
        self.temp_trunk = self.temp2[(self.temp2['timestamp'] >=a) & (self.temp2['timestamp'] <= b)]

    def filter_data2(self):
        # just used to check conc/ecc
        # segmentation set and add exercise label
        self.set = {}
        label = {}
        m22 = {}
        plt.figure()
        plt.plot(self.df1['acc_x'])
        #plt.plot(self.df1['gyro_x'])
        for i in range(len(self.df22['start_time'])):
            a = pd.to_datetime(self.df22['start_time'][i], errors='coerce')
            b = pd.to_datetime(self.df22['end_time'][i], errors='coerce')
            c = self.df22['exercise'][i]


            if c == 'bent_row_conc' or c == 'lat_raise_conc' or c == 'sh_press_conc' \
                    or c == 'bent_row_ecc' or c == 'lat_raise_ecc' or c == 'sh_press_ecc':
                m = self.df1[(self.df1['timestamp'] >= a) & (self.df1['timestamp'] <= b)]
                m2 = m.assign(exercise=c)
                m22[i] = m2
                self.set[i] = m.to_numpy()[:, 1:]  # only IMU signal, delete time
                label[i] = m2.to_numpy()[:, -1]  # only exercise name
                plt.axvline(x=m.index[0], color='orange')
                plt.axvline(x=m.index[-1], color='green')

        self.IMU_data = np.concatenate([self.set[i] for i in self.set.keys()], axis=0)

        # plt.figure()
        # plt.plot(self.set[1][:,0])
        plt.show()

        self.set_filt = {}
        for k in self.set.keys():
            self.set_filt[k] = self.set[k].copy()
            # Filtering the data
            for col in range(self.set[k].shape[1]):
                self.set_filt[k][:, col] = butter_lowpass_filter(data=self.set[k][:, col], cutoff=1, fs=100, order=5)

        return self.set_filt, self.set





    def seg_rep(self, y):
        # cut each set into reps

        lag_value = 1

        n_steps_in = 10
        data_to_plot = y
        seq_x = {}
        peak_ix = {}
        i = 0
        seq_x[0] = data_to_plot[0:n_steps_in]
        for i in np.arange(1, len(data_to_plot), lag_value):
            # find the end of this pattern
            end_ix = i + n_steps_in
            # check if we are beyond the dataset
            if end_ix > len(data_to_plot):
                break
            # gather parts of the pattern
            seq_x[i] = data_to_plot[0:end_ix]
            mean = np.mean(data_to_plot)
            # if reach the peak value
            if seq_x[i][-1] < seq_x[i - 1][-1]:
                peak_ix[i - 1] = end_ix
                continue

        ix = {}
        key = [*peak_ix]
        for k in range(len(key)):
            if k == len(key) - 1:
                break
            a = key[k + 1] - key[k]
            if a > 1:
                ix[k] = key[k]

        fig, ax = plt.subplots()
        ax.plot(y)
        end = {}
        for kk in ix.keys():
            a = ix[kk]
            end[kk] = peak_ix[a]
            ax.axvline(x=end[kk], color='orange')

        ax.axvline(x=len(data_to_plot), color='orange')
        ##ax.axvline(x=peak_ix[119], color='orange')
        ##plt.show()


        rep_point = list(end.values())
        w = data_to_plot.shape[0]
        w = np.int32(w)
        rep_point.append(w)
        rep = [0] + rep_point


        for i in range(len(rep)):
            if i == len(rep) - 1:
                break
            beg = rep[i]
            end = rep[i + 1]
            p = len(rep) * 1.5
            if end - beg < len(data_to_plot) / p:
                rep.remove(end)

        # for i in range(len(rep)):
        #     if i == len(rep) - 1:
        #         break
        #     beg = rep[i]
        #     end = rep[i + 1]
        #     temp = y[beg:end]

            # plt.figure()
            # plt.plot(temp)

        beg = {}
        end = {}
        for i in range(len(rep)):
            if i == len(rep) - 1:
                break
            beg[i] = rep[i]
            end[i] = rep[i + 1]

        return beg, end

    def seg_rep2(self, y):
        # cut each set into phrases

        lag_value = 1

        n_steps_in = 10
        data_to_plot = y
        seq_x = {}
        peak_ix = {}
        i = 0
        seq_x[0] = data_to_plot[0:n_steps_in]
        for i in np.arange(1, len(data_to_plot), lag_value):
            # find the end of this pattern
            end_ix = i + n_steps_in
            # check if we are beyond the dataset
            if end_ix > len(data_to_plot):
                break
            # gather parts of the pattern
            seq_x[i] = data_to_plot[0:end_ix]
            mean = np.mean(data_to_plot)
            # if reach the peak value
            if seq_x[i][-1] < seq_x[i - 1][-1]:
                peak_ix[i - 1] = end_ix
                continue

        ix = {}
        key = [*peak_ix]
        for k in range(len(key)):
            if k == len(key) - 1:
                break
            a = key[k + 1] - key[k]
            if a > 1:
                ix[k] = key[k]

        fig, ax = plt.subplots()
        ax.plot(y)
        end = {}
        for kk in ix.keys():
            a = ix[kk]
            end[kk] = peak_ix[a]
            ax.axvline(x=end[kk], color='orange')

        #ax.axvline(x=len(data_to_plot), color='orange')
        ##ax.axvline(x=peak_ix[119], color='orange')
        ##plt.show()

        rep_point = list(end.values())
        w = data_to_plot.shape[0]
        w = np.int32(w)
        rep_point.append(w)
        rep = [0] + rep_point


        for i in range(len(rep)):
            if i == len(rep) - 1:
                break
            beg = rep[i]
            end = rep[i + 1]
            temp = y[beg:end]

            # plt.figure()
            # plt.plot(temp)

        beg = {}
        end = {}
        for i in range(len(rep)):
            if i == len(rep) - 1:
                break
            beg[i] = rep[i]
            end[i] = rep[i + 1]

        return beg, end
        
'''