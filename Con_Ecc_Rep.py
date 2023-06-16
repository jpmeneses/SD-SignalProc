# cut and save conc_ecc data

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

class Cut_data():

    def __init__(self, subject_ind,set_ind):

        self.subject_ind = subject_ind


        path_info = Path_info(subject_ind=self.subject_ind)

        self.path_points = path_info.path_points

        self.path_imu = path_info.path_IMU
        self.path_imu_cut=path_info.path_IMU_cut

        name ="_reps_points_set_"+str(set_ind)+".npy"

        # Load sets points
        is_exist = os.path.isfile(os.path.join(self.path_points, name))

        if is_exist:
            self.reps_points = np.load(os.path.join(self.path_points, name))
        else:
            warnings.warn("\nFile " + self.path_points + name + " does not exist\n")

        # Load all reps point at once
        list_files = os.listdir(self.path_points)

        # Remove files that don't integrate 'reps_points_set'
        list_files = [f for f in list_files if 'phrases_points_rep' in f]
        list_files = [os.path.splitext(f)[0] for f in list_files]  # Remove extension
        list_files = sorted(list_files, key=lambda x: int(x.split("_")[-1]))  # Sorted the files 0, 1, 2 ... 10, 11, 12 ...

        phrases_points_list = []
        for i, f in enumerate(list_files):
            phrases_points_list.append(np.load(os.path.join(self.path_points, f + '.npy')))

        self.phrases_points_list = phrases_points_list

    def plot_cut_points(self, set_ind):

        name = "IMU_set_" + str(set_ind) + ".npy"
        self.imu_data = np.load(os.path.join(self.path_imu_cut, name))

        data_to_plot = self.imu_data

        # This will also plot the fatigue information on the graph
        fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
        plt.tick_params(axis='x', labelsize=15)
        plt.tick_params(axis='y', labelsize=15)
        axes[1].tick_params(axis='y', labelsize=15)
        axes[2].tick_params(axis='y', labelsize=15)
        axes[0].tick_params(axis='y', labelsize=15)

        n_set = len(self.phrases_points_list)



        axes[0].plot(self.imu_data[:, 5], 'r', label='IMU')
        axes[1].plot(data_to_plot, 'b', label='Marker')
        axes[2].plot(self.imu_data[:, 0], 'g', label='IMUc')

        fig.suptitle("Subject " + str(subject_ind) , fontsize=15)
        # fig.legend()

        box_props = dict(boxstyle='round', alpha=1, facecolor='white')

        for i, p in enumerate(self.phrases_points_list):

            for j, p2 in enumerate(p):
                beg = p2[0]
                end = p2[1]


                axes[0].axvline(x=beg, color='r')
                axes[0].axvline(x=end, color='b')

                axes[1].axvline(x=beg, color='r')
                axes[1].axvline(x=end, color='b')

                axes[2].axvline(x=beg, color='r', linestyle='--')
                axes[2].axvline(x=end, color='b', linestyle='--')



        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.9, wspace=0.1, hspace=0.1)

        plt.xlabel('Time Step', fontsize=15)

    def cut_imu_data(self,set_ind, plotIt=False):

        # TODO: let's do something with the marker later. I guess we don't really need to cut them for now
        # TODO: we will use the marker to run Inverse Kinematics in the future and then cut and spline the IK results

        imu_data_cut = {}

        name = "IMU_set_" + str(set_ind) + ".npy"
        self.imu_data = np.load(os.path.join(self.path_imu_cut, name))


        path_info = Path_info(subject_ind=self.subject_ind)

        for i, p in enumerate(self.phrases_points_list):

            imu_data_cut[i] = {}


            for j, p2 in enumerate(p):

                beg = p2[0]
                end = p2[1]

                # Remove the initial condition and normalize the force with subject's weight
                temp = self.imu_data[beg:end].copy()


                imu_data_cut[i][j] = temp


        path_info = Path_info(subject_ind=self.subject_ind)

        name = "_cut.npy"
        np.save(os.path.join(path_info.path_IMU_cut, name), imu_data_cut)


        print("Forces data cut saved: " + path_info.path_IMU_cut + name)

def run(subject_ind = 3):
    path_info = Path_info(subject_ind=subject_ind)
    path_imu = path_info.data_subject_path_IMU

    ####################################################
    # IMU acc/gyro set and reps; trunk, arm, wrist   ###
    ####################################################
    def obtain(set_filt, IMU_set):
        IMU_data_rep = {}
        # cut linear velocity rep
        i = 7
        for i in set_filt.keys():
            y = set_filt[i][:, 1]
            beg, end = seg_data.seg_rep_arm(y=y)  # get beg, end point by using acceleration x
            IMU_data = IMU_set[i]  # use beg, end point cut everything
            IMU_data_rep[i] = {}
            fig, ax = plt.subplots()
            ax.plot(IMU_data[:, 1])
            for j in beg.keys():
                beg_point = beg[j]
                end_point = end[j]
                temp = IMU_data[beg_point:end_point, :]
                IMU_data_rep[i][j] = temp
                ax.axvline(x=end_point, color='orange')
                ax.axvline(x=beg_point, color='orange')
        plt.show()

        return IMU_data_rep

    # use arm axx_y to cut rep(easy to recognize)

    imu_data_path = os.path.join(path_imu, "Participant " + "{0:03}".format((subject_ind + 1)) + " Arm.csv")
    seg_data = Seg_data(subject_ind=subject_ind, imu_data_path=imu_data_path)
    self = Seg_data(subject_ind=subject_ind, imu_data_path=imu_data_path)
    set_filt_arm, set_arm = seg_data.filter_data(freq=5)
    IMU_data_rep_arm = obtain(set_filt=set_filt_arm, IMU_set=set_arm)


    imu_data_path = os.path.join(path_imu, "Participant " + "{0:03}".format((subject_ind + 1)) + " Trunk.csv")
    self = Seg_data(subject_ind=subject_ind, imu_data_path=imu_data_path)
    seg_data = Seg_data(subject_ind=subject_ind, imu_data_path=imu_data_path)
    # get sets of linear acceleration and angular velocity
    set_filt_trunk, set_trunk = seg_data.filter_data(freq=5)
    IMU_data_rep_trunk = obtain(set_filt=set_filt_arm, IMU_set=set_trunk)

    imu_data_path = os.path.join(path_imu, "Participant " + "{0:03}".format((subject_ind + 1)) + " Wrist.csv")
    seg_data = Seg_data(subject_ind=subject_ind, imu_data_path=imu_data_path)
    self = Seg_data(subject_ind=subject_ind, imu_data_path=imu_data_path)
    set_filt_wrist, set_wrist = seg_data.filter_data(freq=5)
    IMU_data_rep_wrist = obtain(set_filt_arm, set_wrist)

    ####################################################
    # IMU acc/gyro conc and ecc; arm, wrist          ###
    ####################################################
    def obtain2(set_filt, IMU_set):
        IMU_data_rep = {}
        # cut linear velocity rep
        i = 7
        for i in set_filt.keys():
            y = set_filt[i][:, 1]
            beg, end = seg_data.seg_rep22(y=y)  # get beg, end point by using acceleration x
            IMU_data = IMU_set[i]  # use beg, end point cut everything
            IMU_data_rep[i] = {}
            fig, ax = plt.subplots()
            ax.plot(IMU_data[:, 1])

            for j in beg.keys():
                beg_point = beg[j]
                end_point = end[j]
                temp = IMU_data[beg_point:end_point, :]
                IMU_data_rep[i][j] = temp
                ax.axvline(x=end_point, color='orange')
                ax.axvline(x=beg_point, color='orange')
        plt.show()

        return IMU_data_rep

    imu_data_path = os.path.join(path_imu, "Participant " + "{0:03}".format((subject_ind + 1)) + " Arm.csv")
    seg_data = Seg_data(subject_ind=subject_ind, imu_data_path=imu_data_path)
    self = Seg_data(subject_ind=subject_ind, imu_data_path=imu_data_path)
    set_filt_arm, set_arm = seg_data.filter_data(freq=3)
    IMU_conc_ecc_arm = obtain2(set_filt=set_filt_arm, IMU_set=set_arm)

    imu_data_path = os.path.join(path_imu, "Participant " + "{0:03}".format((subject_ind + 1)) + " Wrist.csv")
    seg_data = Seg_data(subject_ind=subject_ind, imu_data_path=imu_data_path)
    self = Seg_data(subject_ind=subject_ind, imu_data_path=imu_data_path)
    set_filt_wrist, set_wrist = seg_data.filter_data(freq=2)
    IMU_conc_ecc_wrist = obtain2(set_filt_arm, set_wrist)

    ####################################################
    # IMU vel set and reps; conc and ecc; arm, wrist ###
    ####################################################
    imu_data_path = os.path.join(path_imu, "Participant " + "{0:03}".format((subject_ind + 1)) + " Arm.csv")
    seg_data = Seg_data(subject_ind=subject_ind, imu_data_path=imu_data_path)
    self = Seg_data(subject_ind=subject_ind, imu_data_path=imu_data_path)
    set_filt, set = seg_data.filter_data(freq=1)
    # get sets of linear velocity
    set_vel, set_filt2 = seg_data.set_velocity(freq=1)

    def obtain(set_filt, IMU_set):
        IMU_data_rep = {}
        # cut linear velocity rep
        i = 1
        for i in set_filt.keys():
            y = set_filt[i][:, 1]
            beg, end = seg_data.seg_rep_arm(y=y)  # get beg, end point by using acceleration x
            IMU_data = IMU_set[i]  # use beg, end point cut everything
            IMU_data_rep[i] = {}
            fig, ax = plt.subplots()
            ax.plot(IMU_data[:, 1])
            for j in beg.keys():
                beg_point = beg[j]
                end_point = end[j]
                temp = IMU_data[beg_point:end_point, :]
                IMU_data_rep[i][j] = temp
                ax.axvline(x=end_point, color='orange')
                ax.axvline(x=beg_point, color='orange')
        plt.show()

        return IMU_data_rep

    IMU_vel_rep_arm = obtain(set_filt_arm, set_vel)

    def obtain2(set_filt, IMU_set):
        IMU_data_rep = {}
        # cut linear velocity rep
        i = 1
        for i in set_filt.keys():
            y = set_filt[i][:, 1]
            beg, end = seg_data.seg_rep22(y=y)  # get beg, end point by using acceleration x
            IMU_data = IMU_set[i]  # use beg, end point cut everything
            IMU_data_rep[i] = {}
            fig, ax = plt.subplots()
            ax.plot(IMU_data[:, 1])
            for j in beg.keys():
                beg_point = beg[j]
                end_point = end[j]
                temp = IMU_data[beg_point:end_point, :]
                IMU_data_rep[i][j] = temp
                ax.axvline(x=end_point, color='orange')
                ax.axvline(x=beg_point, color='orange')
        plt.show()

        return IMU_data_rep

    IMU_vel_conc_ecc_arm = obtain2(set_filt_arm, set_vel)

    set_vel_wrist, set_filt2 = seg_data.set_velocity2(freq=1)

    def obtain(set_filt, IMU_set):
        IMU_data_rep = {}
        # cut linear velocity rep
        i = 1
        for i in set_filt.keys():
            y = set_filt[i][:, 1]
            beg, end = seg_data.seg_rep_arm(y=y)  # get beg, end point by using acceleration x
            IMU_data = IMU_set[i]  # use beg, end point cut everything
            IMU_data_rep[i] = {}
            fig, ax = plt.subplots()
            ax.plot(IMU_data[:, 1])
            for j in beg.keys():
                beg_point = beg[j]
                end_point = end[j]
                temp = IMU_data[beg_point:end_point, :]
                IMU_data_rep[i][j] = temp
                ax.axvline(x=end_point, color='orange')
                ax.axvline(x=beg_point, color='orange')
        plt.show()

        return IMU_data_rep

    IMU_vel_rep_wrist = obtain(set_filt_arm, set_vel)

    def obtain2(set_filt, IMU_set):
        IMU_data_rep = {}
        # cut linear velocity rep
        i = 1
        for i in set_filt.keys():
            y = set_filt[i][:, 1]
            beg, end = seg_data.seg_rep22(y=y)  # get beg, end point by using acceleration x
            IMU_data = IMU_set[i]  # use beg, end point cut everything
            IMU_data_rep[i] = {}
            fig, ax = plt.subplots()
            ax.plot(IMU_data[:, 1])
            for j in beg.keys():
                beg_point = beg[j]
                end_point = end[j]
                temp = IMU_data[beg_point:end_point, :]
                IMU_data_rep[i][j] = temp
                ax.axvline(x=end_point, color='orange')
                ax.axvline(x=beg_point, color='orange')
        plt.show()

        return IMU_data_rep

    IMU_vel_conc_ecc_wrist = obtain2(set_filt_arm, set_vel)
    ##########################################################################################################


    # save rep data

    name = "IMU_data_rep_trunk.npy"
    np.save(os.path.join(path_info.path_IMU_cut, name), IMU_data_rep_trunk)

    print("IMU data cut saved: " + path_info.path_IMU_cut + name)

    name = "IMU_data_rep_arm.npy"
    np.save(os.path.join(path_info.path_IMU_cut, name), IMU_data_rep_arm)

    print("IMU data cut saved: " + path_info.path_IMU_cut + name)

    name = "IMU_data_rep_wrist.npy"
    np.save(os.path.join(path_info.path_IMU_cut, name), IMU_data_rep_wrist)

    print("IMU data cut saved: " + path_info.path_IMU_cut + name)

    IMU_ecc_arm = {}
    IMU_conc_arm = {}
    for k in IMU_conc_ecc_arm.keys():
        IMU_ecc_arm[k] = {}
        IMU_conc_arm[k] = {}
        for num in IMU_conc_ecc_arm[k].keys():
            if num % 2 != 0:
                IMU_conc_arm[k][num] = IMU_conc_ecc_arm[k][num]
            else:
                IMU_ecc_arm[k][num] = IMU_conc_ecc_arm[k][num]

    name = "IMU_ecc_arm.npy"
    np.save(os.path.join(path_info.path_IMU_cut, name), IMU_ecc_arm)
    print("IMU data cut saved: " + path_info.path_IMU_cut + name)
    name = "IMU_conc_arm.npy"
    np.save(os.path.join(path_info.path_IMU_cut, name), IMU_conc_arm)
    print("IMU data cut saved: " + path_info.path_IMU_cut + name)

    IMU_ecc_wrist = {}
    IMU_conc_wrist = {}
    for k in IMU_conc_ecc_wrist.keys():
        IMU_ecc_wrist[k] = {}
        IMU_conc_wrist[k] = {}
        for num in IMU_conc_ecc_wrist[k].keys():
            if num % 2 != 0:
                IMU_conc_wrist[k][num] = IMU_conc_ecc_wrist[k][num]
            else:
                IMU_ecc_wrist[k][num] = IMU_conc_ecc_wrist[k][num]

    name = "IMU_ecc_wrist.npy"
    np.save(os.path.join(path_info.path_IMU_cut, name), IMU_ecc_wrist)
    print("IMU data cut saved: " + path_info.path_IMU_cut + name)
    name = "IMU_conc_wrist.npy"
    np.save(os.path.join(path_info.path_IMU_cut, name), IMU_conc_wrist)
    print("IMU data cut saved: " + path_info.path_IMU_cut + name)

    name = "IMU_vel_rep_arm.npy"
    np.save(os.path.join(path_info.path_IMU_cut, name), IMU_vel_rep_arm)

    print("IMU data cut saved: " + path_info.path_IMU_cut + name)

    name = "IMU_vel_rep_wrist.npy"
    np.save(os.path.join(path_info.path_IMU_cut, name), IMU_vel_rep_wrist)

    print("IMU data cut saved: " + path_info.path_IMU_cut + name)

    IMU_vel_ecc_arm = {}
    IMU_vel_conc_arm = {}
    for k in IMU_vel_conc_ecc_arm.keys():
        IMU_vel_ecc_arm[k] = {}
        IMU_vel_conc_arm[k] = {}
        for num in IMU_vel_conc_ecc_arm[k].keys():
            if num % 2 != 0:
                IMU_vel_conc_arm[k][num] = IMU_vel_conc_ecc_arm[k][num]
            else:
                IMU_vel_ecc_arm[k][num] = IMU_vel_conc_ecc_arm[k][num]

    name = "IMU_vel_ecc_arm.npy"
    np.save(os.path.join(path_info.path_IMU_cut, name), IMU_vel_ecc_arm)
    print("IMU data cut saved: " + path_info.path_IMU_cut + name)
    name = "IMU_vel_conc_arm.npy"
    np.save(os.path.join(path_info.path_IMU_cut, name), IMU_vel_conc_arm)
    print("IMU data cut saved: " + path_info.path_IMU_cut + name)

    IMU_vel_ecc_wrist = {}
    IMU_vel_conc_wrist = {}
    for k in IMU_vel_conc_ecc_wrist.keys():
        IMU_vel_ecc_wrist[k] = {}
        IMU_vel_conc_wrist[k] = {}
        for num in IMU_vel_conc_ecc_wrist[k].keys():
            if num % 2 != 0:
                IMU_vel_conc_wrist[k][num] = IMU_vel_conc_ecc_wrist[k][num]
            else:
                IMU_vel_ecc_wrist[k][num] = IMU_vel_conc_ecc_wrist[k][num]

    name = "IMU_vel_ecc_wrist.npy"
    np.save(os.path.join(path_info.path_IMU_cut, name), IMU_vel_ecc_wrist)
    print("IMU data cut saved: " + path_info.path_IMU_cut + name)
    name = "IMU_vel_conc_wrist.npy"
    np.save(os.path.join(path_info.path_IMU_cut, name), IMU_vel_conc_wrist)
    print("IMU data cut saved: " + path_info.path_IMU_cut + name)
    #name = "IMU_vel_rep_wrist.npy"
    #m=np.load(os.path.join(path_info.path_IMU_cut, name)).item()
    ##########################################################################
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

    # peak velocity of each set
    k = 1
    peak_vel_set = {}
    for k in set_vel.keys():
        temp = set_vel[k]
        max = np.max(temp, axis=0)
        peak_vel_set[k] = max

    # mean velocity of each set
    k = 1
    mean_vel_set = {}
    for k in set_vel.keys():
        temp = set_vel[k]
        mean = np.mean(temp, axis=0)
        mean_vel_set[k] = mean

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

    # peak velocity of each set
    k = 1
    peak_vel_set_wrist = {}
    for k in set_vel_wrist.keys():
        temp = set_vel_wrist[k]
        max = np.max(temp, axis=0)
        peak_vel_set_wrist[k] = max

    # mean velocity of each set
    k = 1
    mean_vel_set_wrist = {}
    for k in set_vel_wrist.keys():
        temp = set_vel_wrist[k]
        mean = np.mean(temp, axis=0)
        mean_vel_set_wrist[k] = mean

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

    name = "peak_vel_set_arm.npy"
    np.save(os.path.join(path_info.path_IMU_cut, name), peak_vel_set)

    print("IMU data cut saved: " + path_info.path_IMU_cut + name)

    name = "mean_vel_set_arm.npy"
    np.save(os.path.join(path_info.path_IMU_cut, name), mean_vel_set)

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

    name = "peak_vel_set_wrist.npy"
    np.save(os.path.join(path_info.path_IMU_cut, name), peak_vel_set_wrist)

    print("IMU data cut saved: " + path_info.path_IMU_cut + name)

    name = "mean_vel_set_wrist.npy"
    np.save(os.path.join(path_info.path_IMU_cut, name), mean_vel_set_wrist)

    print("IMU data cut saved: " + path_info.path_IMU_cut + name)


############################################################################################
subjects_ind = [i for i in range(30, 36)]
subject_ind = 0
a=[]
for subject_ind in subjects_ind:
    #run(subject_ind=subject_ind)
    try:
       run(subject_ind = subject_ind)
    except:
        a.append(subject_ind)
        pass
print(a)




'''
################################
import matplotlib
matplotlib.use('TkAgg')

path_info = Path_info(subject_ind=3)

name = "peak_vel_ecc_arm.npy"
peak_vel_ecc_arm= np.load(os.path.join(path_info.path_IMU_cut, name)).item()
peak_vel_ecc_arm_data = np.concatenate([peak_vel_ecc_arm[i][k][:,np.newaxis] for i in peak_vel_ecc_arm.keys() for k in peak_vel_ecc_arm[i].keys()], axis=1)
plt.figure()
plt.plot(peak_vel_ecc_arm_data[0,:])
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
'''
name = "linear_velocity_wrist.npy"
vel2 = np.load(os.path.join(path_info.path_IMU, name))
plt.figure()
plt.plot(vel2[:,0])
plt.title('mean_vel_set_wrist_data')
plt.show()