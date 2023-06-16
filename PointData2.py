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


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


class Seg_data():

    def __init__(self, subject_ind, imu_data_path):

        self.subject_ind = subject_ind

        path_info = Path_info(subject_ind=self.subject_ind)

        self.path_imu = path_info.data_subject_path_IMU
        self.path_points = path_info.path_points

        self.imu_data_path0 = os.path.join(self.path_imu, "Participant 00" + str(self.subject_ind + 1) + " Arm.csv")
        self.imu_data_path1 = os.path.join(self.path_imu, "Participant 00" + str(self.subject_ind + 1) + " Wrist.csv")
        self.imu_data_path2 = os.path.join(self.path_imu, "Participant 00" + str(self.subject_ind + 1) + " Trunk.csv")
        self.imu_data_path = imu_data_path

        self.df1 = pd.read_csv(self.imu_data_path,
                               names=['timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'],
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
        self.vel = np.concatenate([self.vel, self.vel2], axis=1)
        # plt.figure()
        # plt.plot(self.vel[:,0])
        # plt.show()

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
        self.set = new_dict_set

        self.IMU_data = np.concatenate([self.set[i] for i in self.set.keys()], axis=0)
        # plt.figure()
        # plt.plot(self.set[1][:,0])
        # plt.show()

        self.set_filt = {}
        for k in self.set.keys():
            self.set_filt[k] = self.set[k].copy()
            # Filtering the data
            for col in range(self.set[k].shape[1]):
                self.set_filt[k][:, col] = butter_lowpass_filter(data=self.set[k][:, col], cutoff=freq, fs=100, order=5)

        # plt.figure()
        # plt.plot(self.set_filt[4][:,1])
        # plt.show()

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
        k = 80
        for k in range(len(key)):
            if k == len(key) - 1:
                break
            a = key[k + 1] - key[k]
            if a > 1:
                ix[k] = key[k]
        k = len(key) - 1
        ix[k] = key[-1]

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
        rep = rep_point
        w = data_to_plot.shape[0]
        w = np.int32(w)
        rep.append(w)
        rep = [0] + rep
        rep = np.unique(rep)
        rep = rep.tolist()

        repp = rep.copy()

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
            if data_to_plot[end - 1] < 0:
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

        repp = repppp
        beg = {}
        end = {}
        fig, ax = plt.subplots()
        ax.plot(y)
        for i in range(len(repp)):
            if i == len(repp) - 1:
                break
            beg[i] = repp[i]
            end[i] = repp[i + 1]
            plt.axvline(beg[i], color='red')
            plt.axvline(end[i], color='orange')

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
        low_ix = {}
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
        k = len(key) - 1
        ix[k] = key[-1]

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

        # ax.axvline(x=len(data_to_plot), color='orange')
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

        rep_low = repp
        rep = rep_peak + rep_low
        rep = sorted(rep)

        w = data_to_plot.shape[0]
        w = np.int32(w)
        rep.append(w)
        rep = [0] + rep
        rep = np.unique(rep)
        rep = rep.tolist()
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
        self.set_vel = {}
        k = 1
        start = 0
        fig, ax = plt.subplots()
        ax.plot(self.vel[:, 0])
        for k in self.set.keys():
            rep = len(self.set[k])
            end = start + rep
            self.set_vel[k] = self.vel[start:end]
            ax.axvline(x=start, color="orange")
            ax.axvline(x=end, color="green")
            plt.show()
            start = end

        self.set_filt2 = {}
        for k in self.set_vel.keys():
            self.set_filt2[k] = self.set_vel[k].copy()
            # Filtering the data
            for col in range(self.set_vel[k].shape[1]):
                self.set_filt2[k][:, col] = butter_lowpass_filter(data=self.set_vel[k][:, col], cutoff=freq, fs=100,
                                                                  order=5)

        new_dict_set = {}
        i = 1
        for key, value in zip(self.set_vel.keys(), self.set_vel.values()):
            new_key = i
            new_dict_set[new_key] = self.set_vel[key]
            i = i + 1

        # for i in new_dict_set.keys():
        #     name = "IMU_set_vel_" + str(i) + ".npy"
        #     np.save(os.path.join(path_info.path_IMU_cut, name), new_dict_set[i])

        return self.set_vel, self.set_filt2, new_dict_set


class Cut_data_2():

    def __init__(self, subject_ind):
        self.subject_ind = subject_ind


self = Cut_data_2(subject_ind=2)


class Cut_data_2():

    def __init__(self, subject_ind):

        self.subject_ind = subject_ind

        path_info = Path_info(subject_ind=self.subject_ind)

        self.path_points = path_info.path_points

        self.path_imu = path_info.path_IMU
        self.path_imu_cut = path_info.path_IMU_cut

        path_info = Path_info(subject_ind=subject_ind)
        path_imu = path_info.data_subject_path_IMU
        imu_data_path = os.path.join(path_imu, "Participant " + "{0:03}".format((subject_ind + 1)) + " Arm.csv")
        seg_data = Seg_data(subject_ind=subject_ind, imu_data_path=imu_data_path)
        set_filt, set = seg_data.filter_data(freq=1)

        # get sets of linear velocity
        set_vel, set_filt2, self.new_dict_set = seg_data.set_velocity(freq=1)

    def cut_imu_data_eccs(self, set_ind=1, plotIt=False):
        name = "_eccs_points_set_" + str(set_ind) + ".npy"
        self.eccs_points_list = np.load(os.path.join(self.path_points, name))

        imu_data_cut = {}

        # name = "IMU_set_vel_" + str(set_ind) + ".npy"
        # self.imu_data = np.load(os.path.join(self.path_imu_cut, name))
        self.imu_data= self.new_dict_set[set_ind]

        plt.figure()
        plt.plot(self.imu_data[:, 0])
        path_info = Path_info(subject_ind=self.subject_ind)

        for i, p in enumerate(self.eccs_points_list):
            imu_data_cut[i] = {}

            beg = p[0]
            end = p[1]

            # Remove the initial condition and normalize the force with subject's weight
            temp = self.imu_data[beg:end].copy()

            imu_data_cut[i] = temp

            plt.axvline(x=beg, color='r')
            plt.axvline(x=end, color='b')
            plt.show()

        path_info = Path_info(subject_ind=self.subject_ind)

        arm = {}
        wrist = {}
        for k in imu_data_cut.keys():
            arm[k] = imu_data_cut[k][:, :3]
            wrist[k] = imu_data_cut[k][:, 3:]

        return arm, wrist

    def cut_imu_data_reps(self, set_ind=1, plotIt=False):
        name = "_reps_points_set_" + str(set_ind) + ".npy"
        self.reps_points_list = np.load(os.path.join(self.path_points, name))

        imu_data_cut = {}

        # name = "IMU_set_" + str(set_ind) + ".npy"
        # self.imu_data = np.load(os.path.join(self.path_imu_cut, name))
        self.imu_data = self.new_dict_set[set_ind]

        plt.figure()
        plt.plot(self.imu_data[:, 0])
        path_info = Path_info(subject_ind=self.subject_ind)

        for i, p in enumerate(self.reps_points_list):
            imu_data_cut[i] = {}

            beg = p[0]
            end = p[1]

            # Remove the initial condition and normalize the force with subject's weight
            temp = self.imu_data[beg:end].copy()

            imu_data_cut[i] = temp

            plt.axvline(x=beg, color='r')
            plt.axvline(x=end, color='b')
            plt.show()

        path_info = Path_info(subject_ind=self.subject_ind)
        arm = {}
        wrist = {}
        for k in imu_data_cut.keys():
            arm[k] = imu_data_cut[k][:, :3]
            wrist[k] = imu_data_cut[k][:, 3:]

        return arm, wrist

    def cut_imu_data_concs(self, set_ind=1, plotIt=False):

        name = "_concs_points_set_" + str(set_ind) + ".npy"
        self.concs_points_list = np.load(os.path.join(self.path_points, name))

        imu_data_cut = {}

        # name = "IMU_set_" + str(set_ind) + ".npy"
        # self.imu_data = np.load(os.path.join(self.path_imu_cut, name))
        self.imu_data = self.new_dict_set[set_ind]

        plt.figure()
        plt.plot(self.imu_data[:, 0])
        path_info = Path_info(subject_ind=self.subject_ind)

        for i, p in enumerate(self.concs_points_list):
            imu_data_cut[i] = {}

            beg = p[0]
            end = p[1]

            # Remove the initial condition and normalize the force with subject's weight
            temp = self.imu_data[beg:end].copy()

            imu_data_cut[i] = temp

            plt.axvline(x=beg, color='r')
            plt.axvline(x=end, color='b')
            plt.show()

        path_info = Path_info(subject_ind=self.subject_ind)

        arm = {}
        wrist = {}
        for k in imu_data_cut.keys():
            arm[k] = imu_data_cut[k][:, :3]
            wrist[k] = imu_data_cut[k][:, 3:]

        return arm, wrist

    def cut_imu_data_phrases1(self, plotIt=False):

        imu_data_cut = {}

        name = "IMU_segmented_arm.npy"  # obtained from sliding window.py, removed rest
        self.imu_data = np.load(os.path.join(self.path_imu_cut, name))

        path_info = Path_info(subject_ind=self.subject_ind)

        for i, p in enumerate(self.phrases_points_list):

            imu_data_cut[i] = {}
            beg = p[0][0]
            end = p[-1][-1]

            self.imu_data = self.imu_data[end:]
            self.imu_rep = self.imu_data[beg:end]
            plt.figure()
            plt.plot(self.imu_rep[:, 0])

            for j, p2 in enumerate(p):
                beg = p2[0]
                end = p2[1]

                # Remove the initial condition and normalize the force with subject's weight
                temp = self.imu_rep[beg:end].copy()

                imu_data_cut[i][j] = temp

                plt.axvline(x=beg, color='r')
                plt.axvline(x=end, color='b')
                plt.show()

        path_info = Path_info(subject_ind=self.subject_ind)

        arm = {}
        wrist = {}
        for k in imu_data_cut.keys():
            arm[k] = imu_data_cut[:, :3]
            wrist[k] = imu_data_cut[:, 3:]

        return arm, wrist

    def cut_imu_data_reps1(self, plotIt=False):

        # TODO: let's do something with the marker later. I guess we don't really need to cut them for now
        # TODO: we will use the marker to run Inverse Kinematics in the future and then cut and spline the IK results

        imu_data_cut = {}

        name = "IMU_segmented_arm.npy"  # obtained from sliding window.py, removed rest
        self.imu_data = np.load(os.path.join(self.path_imu_cut, name))

        path_info = Path_info(subject_ind=self.subject_ind)

        for i, p in enumerate(self.reps_points_list):

            imu_data_cut[i] = {}
            beg = p[0][0]
            end = p[-1][-1]

            self.imu_data = self.imu_data[end:]
            self.imu_set = self.imu_data[beg:end]
            plt.figure()
            plt.plot(self.imu_set[:, 0])

            for j, p2 in enumerate(p):
                beg = p2[0]
                end = p2[1]

                # Remove the initial condition and normalize the force with subject's weight
                temp = self.imu_set[beg:end].copy()

                imu_data_cut[i][j] = temp
                plt.axvline(x=beg, color='r')
                plt.axvline(x=end, color='b')
                plt.show()

        path_info = Path_info(subject_ind=self.subject_ind)

        # name = "_cut.npy"
        # np.save(os.path.join(path_info.path_IMU_cut, name), imu_data_cut)
        return imu_data_cut


subject_ind = 0
set_ind = 1

def corret_data(subject_ind, set_ind):

    path_info = Path_info(subject_ind=subject_ind)
    path_imu = path_info.data_subject_path_IMU
    imu_data_path = os.path.join(path_imu, "Participant " + "{0:03}".format((subject_ind + 1)) + " Arm.csv")
    seg_data = Seg_data(subject_ind=subject_ind, imu_data_path=imu_data_path)

    cut_data = Cut_data_2(subject_ind=subject_ind)

    IMU_cut_reps_arm, IMU_cut_reps_wrist = cut_data.cut_imu_data_reps(set_ind=set_ind, plotIt=False)
    IMU_cut_eccs_arm, IMU_cut_eccs_wrist = cut_data.cut_imu_data_eccs(set_ind=set_ind, plotIt=False)
    IMU_cut_concs_arm, IMU_cut_concs_wrist = cut_data.cut_imu_data_concs(set_ind=set_ind, plotIt=False)

    name = "IMU_vel_rep_arm.npy"
    IMU_vel_rep_arm = np.load(os.path.join(path_info.path_IMU_cut, name)).item()
    IMU_vel_rep_arm[set_ind] = IMU_cut_reps_arm

    name = "IMU_vel_rep_wrist.npy"
    IMU_vel_rep_wrist = np.load(os.path.join(path_info.path_IMU_cut, name)).item()
    IMU_vel_rep_wrist[set_ind] = IMU_cut_reps_wrist

    name = "IMU_vel_ecc_arm.npy"
    IMU_vel_ecc_arm = np.load(os.path.join(path_info.path_IMU_cut, name)).item()
    IMU_vel_ecc_arm[set_ind] = IMU_cut_eccs_arm

    name = "IMU_vel_ecc_wrist.npy"
    IMU_vel_ecc_wrist = np.load(os.path.join(path_info.path_IMU_cut, name)).item()
    IMU_vel_ecc_wrist[set_ind] = IMU_cut_eccs_wrist

    name = "IMU_vel_conc_arm.npy"
    IMU_vel_conc_arm = np.load(os.path.join(path_info.path_IMU_cut, name)).item()
    IMU_vel_conc_arm[set_ind] = IMU_cut_concs_arm

    name = "IMU_vel_conc_wrist.npy"
    IMU_vel_conc_wrist = np.load(os.path.join(path_info.path_IMU_cut, name)).item()
    IMU_vel_conc_wrist[set_ind] = IMU_cut_concs_wrist

    # peak velocity of each phase
    k = 1
    peak_vel_ecc = {}
    for k in IMU_vel_ecc_arm.keys():
        temp = IMU_vel_ecc_arm[k]
        peak_vel_ecc[k] = {}
        for i in temp.keys():
            temp2 = temp[i]
            max = np.max(temp2, axis=0)
            peak_vel_ecc[k][i] = max
    peak_vel_conc = {}
    for k in IMU_vel_conc_arm.keys():
        temp = IMU_vel_conc_arm[k]
        peak_vel_conc[k] = {}
        for i in temp.keys():
            temp2 = temp[i]
            max = np.max(temp2, axis=0)
            peak_vel_conc[k][i] = max

    # mean velocity of each phase
    k = 1
    mean_vel_ecc = {}
    for k in IMU_vel_ecc_arm.keys():
        temp = IMU_vel_ecc_arm[k]
        mean_vel_ecc[k] = {}
        for i in temp.keys():
            temp2 = temp[i]
            mean = np.mean(temp2, axis=0)
            mean_vel_ecc[k][i] = mean

    mean_vel_conc = {}
    for k in IMU_vel_conc_arm.keys():
        temp = IMU_vel_conc_arm[k]
        mean_vel_conc[k] = {}
        for i in temp.keys():
            temp2 = temp[i]
            mean = np.mean(temp2, axis=0)
            mean_vel_conc[k][i] = mean

    ##### wrist ####
    k = 1
    peak_vel_ecc_wrist = {}
    for k in IMU_vel_ecc_wrist.keys():
        temp = IMU_vel_ecc_wrist[k]
        peak_vel_ecc_wrist[k] = {}
        for i in temp.keys():
            temp2 = temp[i]
            max = np.max(temp2, axis=0)
            peak_vel_ecc_wrist[k][i] = max
    peak_vel_conc_wrist = {}
    for k in IMU_vel_conc_wrist.keys():
        temp = IMU_vel_conc_wrist[k]
        peak_vel_conc_wrist[k] = {}
        for i in temp.keys():
            temp2 = temp[i]
            max = np.max(temp2, axis=0)
            peak_vel_conc_wrist[k][i] = max

    # mean velocity of each phase
    k = 1
    mean_vel_ecc_wrist = {}
    for k in IMU_vel_ecc_wrist.keys():
        temp = IMU_vel_ecc_wrist[k]
        mean_vel_ecc_wrist[k] = {}
        for i in temp.keys():
            temp2 = temp[i]
            mean = np.mean(temp2, axis=0)
            mean_vel_ecc_wrist[k][i] = mean

    mean_vel_conc_wrist = {}
    for k in IMU_vel_conc_wrist.keys():
        temp = IMU_vel_conc_wrist[k]
        mean_vel_conc_wrist[k] = {}
        for i in temp.keys():
            temp2 = temp[i]
            mean = np.mean(temp2, axis=0)
            mean_vel_conc_wrist[k][i] = mean
    ###############################################
    # peak velocity of each rep
    k = 1
    peak_vel_rep = {}
    for k in IMU_vel_rep_arm.keys():
        temp = IMU_vel_rep_arm[k]
        peak_vel_rep[k] = {}
        for i in temp.keys():
            temp2 = temp[i]
            max = np.max(temp2, axis=0)
            peak_vel_rep[k][i] = max

    # mean velocity of each rep
    k = 1
    mean_vel_rep = {}
    for k in IMU_vel_rep_arm.keys():
        temp = IMU_vel_rep_arm[k]
        mean_vel_rep[k] = {}
        for i in temp.keys():
            temp2 = temp[i]
            mean = np.mean(temp2, axis=0)
            mean_vel_rep[k][i] = mean


    ############### wrist #############

    # peak velocity of each rep
    k = 1
    peak_vel_rep_wrist = {}
    for k in IMU_vel_rep_wrist.keys():
        temp = IMU_vel_rep_wrist[k]
        peak_vel_rep_wrist[k] = {}
        for i in temp.keys():
            temp2 = temp[i]
            max = np.max(temp2, axis=0)
            peak_vel_rep_wrist[k][i] = max

    # mean velocity of each rep
    k = 1
    mean_vel_rep_wrist = {}
    for k in IMU_vel_rep_wrist.keys():
        temp = IMU_vel_rep_wrist[k]
        mean_vel_rep_wrist[k] = {}
        for i in temp.keys():
            temp2 = temp[i]
            mean = np.mean(temp2, axis=0)
            mean_vel_rep_wrist[k][i] = mean

    ################################
    name = "peak_vel_ecc_arm.npy"
    np.save(os.path.join(path_info.path_IMU_cut, name), peak_vel_ecc)

    print("IMU data cut saved: " + path_info.path_IMU_cut + name)
    name = "peak_vel_conc_arm.npy"
    np.save(os.path.join(path_info.path_IMU_cut, name), peak_vel_conc)
    print("IMU data cut saved: " + path_info.path_IMU_cut + name)

    name = "mean_vel_ecc_arm.npy"
    np.save(os.path.join(path_info.path_IMU_cut, name), mean_vel_ecc)
    print("IMU data cut saved: " + path_info.path_IMU_cut + name)
    name = "mean_vel_conc_arm.npy"
    np.save(os.path.join(path_info.path_IMU_cut, name), mean_vel_conc)
    print("IMU data cut saved: " + path_info.path_IMU_cut + name)

    name = "peak_vel_rep_arm.npy"
    np.save(os.path.join(path_info.path_IMU_cut, name), peak_vel_rep)

    print("IMU data cut saved: " + path_info.path_IMU_cut + name)

    name = "mean_vel_rep_arm.npy"
    np.save(os.path.join(path_info.path_IMU_cut, name), mean_vel_rep)

    print("IMU data cut saved: " + path_info.path_IMU_cut + name)


    ################# wrist ##############

    name = "peak_vel_ecc_wrist.npy"
    np.save(os.path.join(path_info.path_IMU_cut, name), peak_vel_ecc_wrist)

    print("IMU data cut saved: " + path_info.path_IMU_cut + name)
    name = "peak_vel_conc_wrist.npy"
    np.save(os.path.join(path_info.path_IMU_cut, name), peak_vel_conc_wrist)
    print("IMU data cut saved: " + path_info.path_IMU_cut + name)

    name = "mean_vel_ecc_wrist.npy"
    np.save(os.path.join(path_info.path_IMU_cut, name), mean_vel_ecc_wrist)
    print("IMU data cut saved: " + path_info.path_IMU_cut + name)
    name = "mean_vel_conc_wrist.npy"
    np.save(os.path.join(path_info.path_IMU_cut, name), mean_vel_conc_wrist)
    print("IMU data cut saved: " + path_info.path_IMU_cut + name)

    name = "peak_vel_rep_wrist.npy"
    np.save(os.path.join(path_info.path_IMU_cut, name), peak_vel_rep_wrist)

    print("IMU data cut saved: " + path_info.path_IMU_cut + name)

    name = "mean_vel_rep_wrist.npy"
    np.save(os.path.join(path_info.path_IMU_cut, name), mean_vel_rep_wrist)

    print("IMU data cut saved: " + path_info.path_IMU_cut + name)


corret_data(subject_ind, set_ind)
