# user-independent classification
# sliding window

import os
import random
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline

from Classes.Path import Path_info


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


def change_fatigue_level(Y, fatigue_level='2'):
    # Part to change the fatigue level
    for i in range(len(Y)):

        if fatigue_level == '2':

            if Y[i] < 7:
                Y[i] = 0
            if Y[i] >= 7:
                Y[i] = 1

        if fatigue_level == '3':

            if Y[i] < 4:
                Y[i] = 0
            if Y[i] < 7 and Y[i] > 3:
                Y[i] = 1
            if Y[i] > 6:
                Y[i] = 2

        if fatigue_level == '4':

            if Y[i] < 3:
                Y[i] = 0
            if Y[i] < 5 and Y[i] > 2:
                Y[i] = 1
            if Y[i] == 5 or Y[i] == 6:
                Y[i] = 2
            if Y[i] > 6:
                Y[i] = 3

        if fatigue_level == '7':

            if Y[i] < 3:
                Y[i] = 0
            if Y[i] == 3:
                Y[i] = 1
            if Y[i] == 4:
                Y[i] = 2
            if Y[i] == 5:
                Y[i] = 3
            if Y[i] == 6:
                Y[i] = 4
            if Y[i] == 7:
                Y[i] = 5
            if Y[i] == 8:
                Y[i] = 6

    return Y


def orientation_matrix(theta):
    Rx = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
    Ry = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

    return Rx, Ry, Rz


class Up_sampling():

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.num_label = len(np.unique(self.Y, axis=0))
        self.unique_label = np.unique(self.Y, axis=0)
        self.sample = self.X.shape[0]
        self.height = self.X.shape[1]
        #self.width = self.X.shape[2]
        b = []
        for uniq in self.unique_label:
            a = np.asscalar(uniq)
            b.append(a)
        self.unique_label = b

    def upsampling(self):
        n_label = {}
        index_label = {}
        X_new_dict = {}
        Y_new_dict = {}
        for uniq in self.unique_label:
            index_label[uniq] = np.where(self.Y == uniq)[0]
            n_label[uniq] = len(np.where(self.Y == uniq)[0])
            # print('Label ' + str(uniq) + " has " + str(n_label[uniq]) + ' samples')
        n_majority_label = np.max([n_label[k] for k in n_label.keys()])
        for uniq in self.unique_label:
            if n_label[uniq] < n_majority_label:
                p = np.random.choice(index_label[uniq], size=n_majority_label - n_label[uniq])
                temp1 = self.X[p]
                temp2 = self.X[index_label[uniq]]
                temp_X = np.concatenate([temp1, temp2], axis=0)
                temp1 = self.Y[p]
                temp2 = self.Y[index_label[uniq]]
                temp_Y = np.concatenate([temp1, temp2], axis=0)
            else:
                p = index_label[uniq]
                temp_X = self.X[p]
                temp_Y = self.Y[p]
            X_new_dict[uniq] = temp_X
            Y_new_dict[uniq] = temp_Y
        X_aug = np.concatenate([X_new_dict[k] for k in self.unique_label], axis=0)
        Y_aug = np.concatenate([Y_new_dict[k] for k in self.unique_label], axis=0)
        print(' ')
        n_label_new = {}
        index_label_new = {}
        for uniq in self.unique_label:
            index_label_new[uniq] = np.where(Y_aug == uniq)[0]
            n_label_new[uniq] = len(np.where(Y_aug == uniq)[0])
            # print('Label ' + str(uniq) + " has " + str(n_label_new[uniq]) + ' samples after upsampling')
        return X_aug, Y_aug


class Data_augmentation():

    def __init__(self, X, Y, hyper_DA):

       self.X = X.copy()
       self.Y = Y.copy()
       self.X_orig = X.copy()
       self.Y_orig = Y.copy()
       self.hyper_DA = hyper_DA

       self.augmentation_types = self.hyper_DA['augmentation_types']
       self.n = self.hyper_DA['n']
       #self.degree = self.hyper_DA['degree']
       if self.augmentation_types == 'rn':
           self.random_noise_factor = self.hyper_DA['random_noise_factor']
       elif self.augmentation_types == 'tw':
          self.sigma_tw = self.hyper_DA['sigma_tw']
          self.knot_tw = self.hyper_DA['knot_tw']
       elif self.augmentation_types == 'aw':
           self.sigma_aw = self.hyper_DA['sigma_aw']
           self.knot_aw = self.hyper_DA['knot_aw']
       elif self.augmentation_types == 'ro':
           self.degree = self.hyper_DA['degree']
       elif self.augmentation_types == 'sc':
           self.sigma_sc = self.hyper_DA['sigma_sc']

       self.sample = self.X.shape[0]
       self.height = self.X.shape[1]
       self.width = self.X.shape[2]
       self.sample_new = self.sample + self.sample * self.n

    def run(self):

        if self.augmentation_types == 'rn':
             X_aug, Y_aug = self.add_random_noise(n=self.n, boundaries=[-1, 1])
             self.X_aug = X_aug
             self.Y_aug = Y_aug

        elif self.augmentation_types == 'tw':
            X_aug, Y_aug = self.time_warp(n=self.n)
            self.X_aug = X_aug
            self.Y_aug = Y_aug

        elif self.augmentation_types == 'aw':
            X_aug, Y_aug = self.amplitude_warp(n=self.n)
            self.X_aug = X_aug
            self.Y_aug = Y_aug

        elif self.augmentation_types == 'ro':
            X_aug, Y_aug = self.rotation(n=self.n)
            self.X_aug = X_aug
            self.Y_aug = Y_aug

        elif self.augmentation_types == 'sc':
            X_aug, Y_aug = self.scaling(n=self.n, sigma=self.sigma_sc)
            self.X_aug = X_aug
            self.Y_aug = Y_aug

        else:
            self.X_aug = self.X
            self.Y_aug = self.Y

    def add_random_noise(self, n, boundaries=[-1, 1]):

        self.sample = self.X.shape[0]
        X_aug = np.empty([self.sample * n, self.height, self.width])
        Y_aug = np.empty([self.sample * n])
        for i in range(self.sample):
            for k in range(n):
                amplitude_current_signals = np.max(self.X[i, :, :], axis=0) - np.min(self.X[i, :, :], axis=0)
                noise = np.random.uniform(boundaries[0], boundaries[1], size=(self.height, self.width)) * self.random_noise_factor * amplitude_current_signals
                X_aug[i * n + k, :, :] = self.X[i, :, :] + noise
                Y_aug[i * n + k] = self.Y[i]
        return X_aug, Y_aug

    def generate_random_curves(self, n=10, loc=1, sigma=0.5, knot=2):

       xx = (np.ones((n, 1)) * (np.arange(0, self.height, (self.height - 1) / (knot + 1)))).transpose()
       yy = np.random.normal(loc=loc, scale=sigma, size=(knot + 2, n))
       x_range = np.arange(self.height)
       tt = CubicSpline(xx[:, 0], yy, axis=0)(x_range)
       return tt

    def distort_time_steps(self, n=10, sigma=0.5, knot=2):

       tt = self.generate_random_curves(n=n, loc=1, sigma=sigma, knot=knot)
       tt_new = np.cumsum(tt, axis=0)
       t_scale = (self.height - 1) / tt_new[-1, :]
       tt_new = tt_new * t_scale
       return tt_new

    def time_warp(self, n=10):

       self.sample = self.X.shape[0]
       X_aug = np.empty([self.sample * n, self.height, self.width])
       Y_aug = np.empty([self.sample * n])
       tt_old = np.arange(self.height)
       for i in range(self.sample):
           tt_new = self.distort_time_steps(n, sigma=self.sigma_tw, knot=self.knot_tw)
           for k in range(n):
               X_aug[i * n + k, :, :] = interp1d(tt_old, self.X[i, :, :], axis=0, kind='cubic', fill_value="extrapolate")(tt_new[:, k])
               Y_aug[i * n + k] = self.Y[i]
       return X_aug, Y_aug

    def amplitude_warp(self, n=10):

       self.sample = self.X.shape[0]
       X_aug = np.empty([self.sample * n, self.height, self.width])
       Y_aug = np.empty([self.sample * n])
       for i in range(self.sample):
           for k in range(n):
               add = self.generate_random_curves(n=self.width, loc=0, sigma=self.sigma_aw, knot=self.knot_aw)
               amplitude_current_signals = np.max(self.X[i, :, :], axis=0) - np.min(self.X[i, :, :], axis=0)
               X_aug[i * n + k, :, :] = self.X[i, :, :] + (add[:, :] * amplitude_current_signals)
               Y_aug[i * n + k] = self.Y[i]
       return X_aug, Y_aug

    def orientation_matrix(theta=5):
        Rx = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
        Ry = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        Rz = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

        return Rx, Ry, Rz

    def DA_Rotation(self):
        axis = np.random.uniform(low=-1, high=1, size=self.X.shape[2])
        theta = self.degree * (np.pi / 180)
        angle = np.random.uniform(low=-theta, high=theta)
        return axangle2mat(axis, angle)

    def rotation(self, n=10):

        self.sample = self.X.shape[0]
        X_aug = np.empty([self.sample * n, self.height, self.width])
        Y_aug = np.empty([self.sample * n])
        R_acc = {}

        axis = np.random.uniform(low=-1, high=1, size=3)
        theta = self.degree * (np.pi / 180)
        angle = np.random.uniform(low=-theta, high=theta)
        # for col in Opensim_a.keys():

        theta = self.degree * (np.pi / 180)
        Rx, Ry, Rz = orientation_matrix(theta=theta)
        for i in range(self.sample):
            for k in range(n):
                for m in range(self.width):
                    if m % 3 == 0:
                        #X_aug[i * n + k, :,  m:m + 3] = np.dot(Ry, self.X[i, :, m:m + 3].T).T
                        X_aug[i * n + k, :, m:m + 3] = np.matmul(self.X[i, :, m:m + 3], axangle2mat(axis, angle))
                Y_aug[i * n + k] = self.Y[i]

        return X_aug, Y_aug

    def scaling(self, n=10, sigma=0.1):

       self.sample = self.X.shape[0]
       X_aug = np.empty([self.sample * n, self.height, self.width])
       Y_aug = np.empty([self.sample * n])

       for i in range(self.sample):
           for k in range(n):
               scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1, self.X.shape[2]))  # shape=(1,3)
               myNoise = np.matmul(np.ones((self.X.shape[1], 1)), scalingFactor)
               X_aug[i * n + k, :, :] = self.X[i, :, :] * myNoise
               Y_aug[i * n + k] = self.Y[i]
       return X_aug, Y_aug


class Create_DB():

    def __init__(self, DB_N, hyper_DA):
        self.path_info = Path_info(subject_ind=0)
        self.path_info.get_DB_path(DB_N=DB_N, delete_folder=False)
        self.path_info.get_DB_info(DB_N=DB_N)

        self.hyper_DA = hyper_DA
        self.n = self.hyper_DA['n']

        self.augmentation_types = self.hyper_DA['augmentation_types']
        if self.augmentation_types == 'rn':
            self.random_noise_factor = self.hyper_DA['random_noise_factor']
        elif self.augmentation_types == 'tw':
            self.sigma_tw = self.hyper_DA['sigma_tw']
            self.knot_tw = self.hyper_DA['knot_tw']
        elif self.augmentation_types == 'aw':
            self.sigma_aw = self.hyper_DA['sigma_aw']
            self.knot_aw = self.hyper_DA['knot_aw']

        # p_CV = np.concatenate([self.path_info.subjects, self.path_info.subjects])
        # # n = int(self.X_train_valid.shape[0] * self.percentage_data_to_use)
        # self.K_fold = 3 # outer loop
        # self.K_fold_I = np.array([p_CV[i * self.n_subjects_test:i * self.n_subjects_test + self.n_subjects_test] for i in
        #      range(0, self.K_fold)])
        # # KK=3
        # self.K = KK
        # #self.Test_I = np.array(self.subjects_test_with_data)
        # self.current_K = KK
        # I_train_valid = self.K_fold_I[[x for x in range(self.K_fold) if x != self.current_K]].reshape([-1])
        # I_test = self.K_fold_I[self.current_K]
        # self.subjects_train_valid=I_train_valid
        # self.subjects_test=I_test

        self.data_X = {}
        self.data_Y = {}
        self.data_Y_orig = {}
        self.data_Y_subject = {}

        for subject in self.path_info.subjects:

            path_info_subj = Path_info(subject_ind=subject)

            if self.path_info.which_data == ['IMU']:
                path = path_info_subj.path_IMU_cut
                self.dof_to_train_ind = [i for i in range(0, 6)]

            X = np.load(os.path.join(path_info_subj.path_IMU_cut, "IMU_slided.npy"), allow_pickle=True)
            Y = np.load(os.path.join(path_info_subj.path_IMU_cut, "label_slided.npy"), allow_pickle=True)

            self.data_X[subject] = X
            self.data_Y[subject] = Y

            # data_augmentation = Data_augmentation(X, Y, hyper_DA=self.hyper_DA)
            # data_augmentation.run()


    def create_database(self, plotIt=False):

        subjects_with_data = list(self.data_X.keys())
        subjects_train_valid_with_data = self.path_info.subjects_train_valid
        subjects_test_with_data = self.path_info.subjects_test

        np.save(os.path.join(self.path_info.path_DB_N, "subjects_train_valid_with_data.npy"), subjects_train_valid_with_data)
        np.save(os.path.join(self.path_info.path_DB_N, "subjects_test_with_data.npy"), subjects_test_with_data)
        # data_X = self.data_X
        X_train_valid = np.concatenate(
            [self.data_X[subject] for subject in subjects_train_valid_with_data], axis=0)
        Y_train_valid = np.concatenate(
            [self.data_Y[subject] for subject in subjects_train_valid_with_data], axis=0)

        if self.path_info.which_data == ['IMU']:
            # TODO: for now we don't normalize the output, it is a regression task for now
            inds_X = [[i] for i in range(X_train_valid.shape[2])]
            # Compute global mean and std for train/validation set
            X_data_std, X_mean, X_std = standardize(data=X_train_valid, inds=inds_X)


        # Apply mean and std to train/validation/test set with mean/std from train/validation set
        data_X_std = {}
        n_samples = {}

        for subject in subjects_with_data:

            data_X_std[subject] = {}
            n_samples[subject] = {}

            data_X_std[subject] = standardize_with_mean_and_sd(self.data_X[subject], X_mean, X_std)
            n_samples[subject] = self.data_X[subject].shape[0]

        np.save(os.path.join(self.path_info.path_DB_N, "X_Mean.npy"), X_mean)
        np.save(os.path.join(self.path_info.path_DB_N, "X_sd.npy"), X_std)

        data_Y = {}

        for subject in subjects_train_valid_with_data:

            X = np.concatenate([data_X_std[subject]], axis=0)
            Y = np.concatenate([self.data_Y[subject]], axis=0)

            # Y=Y[:X.shape[0]]

            p = np.array([i for i in range(X.shape[0])])
            random.shuffle(p)
            n = int(X.shape[0] * self.path_info.percentage_data_to_use)
            X = X[p][:n]
            Y = Y[p][:n]


            name = "X_" + str(subject).zfill(2) + ".npy"
            np.save(os.path.join(self.path_info.path_DB_base_train_valid_test, name), X)
            name = "Y_" + str(subject).zfill(2) + ".npy"
            np.save(os.path.join(self.path_info.path_DB_base_train_valid_test, name), Y)

            data_X_std[subject] = X
            data_Y[subject] = Y


        X_new = np.concatenate([data_X_std[subject] for subject in subjects_train_valid_with_data], axis=0)
        Y_new = np.concatenate([data_Y[subject] for subject in subjects_train_valid_with_data], axis=0)

        print("There are " + str(X_new.shape[0]) + " samples in X for subjects_train_valid_with_data")
        print("There are " + str(Y_new.shape[0]) + " samples in Y for subjects_train_valid_with_data")

        '''
        name = "X_subjects_new_train_valid" + ".npy"
        np.save(self.path_DB_base_train_valid_test + "" + name, X_new)
        name = "Y_subjects_new_train_valid" + ".npy"
        np.save(self.path_DB_base_train_valid_test + name, Y_new)
        name = "Y_subjects_train_valid_subjects" + ".npy"
        np.save(self.path_DB_base_train_valid_test + name, Y_subject)
        '''

        if plotIt:

            self.fig = plt.figure()
            self.ax1 = plt.subplot(1, 1, 1)

            data_forces_to_plot = X_new[:, :, 2]

            for i in range(data_forces_to_plot.shape[0]):

                self.ax1.plot(data_forces_to_plot[i])

            self.fig.suptitle('Spline - Set ')

        # Apply mean and std to test set
        for subject in subjects_test_with_data:

            X = np.concatenate([data_X_std[subject]], axis=0)
            Y = np.concatenate([self.data_Y[subject]], axis=0)


            name = "X_" + str(subject).zfill(2) + ".npy"
            np.save(os.path.join(self.path_info.path_DB_base_train_valid_test, name), X)
            name = "Y_" + str(subject).zfill(2) + ".npy"
            np.save(os.path.join(self.path_info.path_DB_base_train_valid_test, name), Y)


            data_X_std[subject] = X
            data_Y[subject] = Y


        X_new = np.concatenate([data_X_std[subject] for subject in subjects_test_with_data], axis=0)
        Y_new = np.concatenate([data_Y[subject] for subject in subjects_test_with_data], axis=0)


        print("There are " + str(X_new.shape[0]) + " sample in X for subjects_test_with_data")
        print("There are " + str(Y_new.shape[0]) + " sample in Y for subjects_test_with_data")

        '''
        name = "X_subjects_new_test" + ".npy"
        np.save(self.path_DB_base_train_valid_test + "" + name, X_new)
        name = "Y_subjects_new_test" + ".npy"
        np.save(self.path_DB_base_train_valid_test + name, Y_new)
        name = "Y_subjects_test_subjects" + ".npy"
        np.save(self.path_DB_base_train_valid_test + name, Y_subject)
        '''

        name = "n_samples.npy"
        np.save(os.path.join(self.path_info.path_DB_N, name), n_samples)


class Load_data():

    def __init__(self, DB_N, which_model='MLP', K_tot=3):

        self.DB_N = DB_N
        self.which_model = which_model
        self.K_tot = K_tot

        self.path_info = Path_info(subject_ind=0)
        self.path_info.get_DB_path(DB_N=DB_N, delete_folder=False)
        self.path_info.get_DB_info(DB_N=DB_N)

        if self.path_info.what_kind_of_prediction == 'classification':
            self.num_classes = 3

        X_Mean = np.load(os.path.join(self.path_info.path_DB_N, "X_Mean.npy"), allow_pickle=True)
        X_sd = np.load(os.path.join(self.path_info.path_DB_N, "X_sd.npy"), allow_pickle=True)

        n_samples = np.load(os.path.join(self.path_info.path_DB_N, "n_samples.npy"), allow_pickle=True)

        self.subjects_train_valid_with_data = np.load(os.path.join(self.path_info.path_DB_N,
                                                "subjects_train_valid_with_data.npy"), allow_pickle=True)
        self.subjects_test_with_data = np.load(os.path.join(self.path_info.path_DB_N,
                                                "subjects_test_with_data.npy"), allow_pickle=True)

        self.X_train_valid_dict = {}
        self.Y_train_valid_dict = {}
        self.Y_orig_train_valid_dict = {}

        for s in self.subjects_train_valid_with_data:
            name = "X_" + str(s).zfill(2) + ".npy"
            X = np.load(os.path.join(self.path_info.path_DB_base_train_valid_test, name), allow_pickle=True)

            name = "Y_" + str(s).zfill(2) + ".npy"
            Y = np.load(os.path.join(self.path_info.path_DB_base_train_valid_test, name), allow_pickle=True)

            # self.X_train_valid_dict[s] = np.concatenate([X[:, :, i] for i in range(X.shape[2])], axis=1)
            self.Y_train_valid_dict[s] = Y
            self.X_train_valid_dict[s] = X

        self.X_test_dict = {}
        self.Y_test_dict = {}

        for s in self.subjects_test_with_data:

            name = "X_" + str(s).zfill(2) + ".npy"
            X = np.load(os.path.join(self.path_info.path_DB_base_train_valid_test, name), allow_pickle=True)

            name = "Y_" + str(s).zfill(2) + ".npy"
            Y = np.load(os.path.join(self.path_info.path_DB_base_train_valid_test, name), allow_pickle=True)

            # if self.what_kind_of_prediction == 'classification':
            #
            #     Y = to_categorical(Y, num_classes=self.num_classes)

            self.X_test_dict[s] = X
            self.Y_test_dict[s] = Y

        p_CV = np.concatenate([self.subjects_train_valid_with_data, self.subjects_train_valid_with_data])

        # n = int(self.X_train_valid.shape[0] * self.percentage_data_to_use)
        nsv = self.path_info.n_subjects_valid
        self.K_fold_I = np.array([p_CV[i*nsv:i*nsv+nsv] for i in range(K_tot)])

        self.sample_size = {}
        for s in self.subjects_test_with_data:
            self.sample_size[s] = self.Y_test_dict[s].shape[0]

    def K_fold_data(self, K=1):
        #self.K = K
        self.current_K = K
        self.Test_I = np.array(self.subjects_test_with_data)

        I_train = self.K_fold_I[[x for x in range(self.K_tot) if x != self.current_K]].reshape([-1])
        I_valid = self.K_fold_I[self.current_K]

        X_train = np.concatenate([self.X_train_valid_dict[s] for s in I_train], axis=0)
        X_valid = np.concatenate([self.X_train_valid_dict[s] for s in I_valid], axis=0)

        Y_train = np.concatenate([self.Y_train_valid_dict[s] for s in I_train], axis=0)
        Y_valid = np.concatenate([self.Y_train_valid_dict[s] for s in I_valid], axis=0)

        nsamples_train, nx_train, ny_train = X_train.shape
        nsamples_valid, nx_valid, ny_valid = X_valid.shape

        if self.which_model == 'MLP':

            X_train = X_train.reshape((nsamples_train, nx_train * ny_train))
            X_valid = X_valid.reshape((nsamples_valid, nx_valid * ny_valid))

        if self.which_model == 'CNN':

            X_train = X_train[:, :, :, np.newaxis]
            X_valid = X_valid[:, :, :, np.newaxis]


            X_train =np.concatenate([X_train[:,:,k, :] for k in [1,4,7,10,13]],axis=2)
            X_valid = np.concatenate([X_valid[:, :, k,:] for k in [1, 4, 7, 10, 13]], axis=2)

            X_train = X_train[:, :, :, np.newaxis]
            X_valid = X_valid[:, :, :, np.newaxis]

            #X_train = np.concatenate([X_train[:, :, 6:9, np.newaxis], X_train[:, :, 6:9, np.newaxis]], axis=2)

        if self.which_model == 'LSTM':

            X_train = X_train
            X_valid = X_valid

        if self.which_model == 'CNNLSTM':

            X_train = X_train[:, :, :, np.newaxis]
            X_valid = X_valid[:, :, :, np.newaxis]

        if self.which_model == 'ConvLSTM2D':
            #X_train=X_train[:nsamples_train//25*25,:,:]
            #X_valid = X_valid[:nsamples_valid // 25 * 25, :, :]
            #X_train = X_train.reshape((X_train.shape[0]//25, 25, nx_train,  ny_train,1))
            #X_valid = X_valid.reshape((X_valid.shape[0]//25, 25, nx_valid,  ny_valid, 1))

            X_train = X_train[:, :, :, np.newaxis]
            X_valid =  X_valid[:, :, :, np.newaxis]

        return X_train, X_valid, Y_train, Y_valid

    def test_data(self):

        I_test = self.subjects_test_with_data

        X_test = np.concatenate([self.X_test_dict[s] for s in I_test], axis=0)
        Y_test = np.concatenate([self.Y_test_dict[s] for s in I_test], axis=0)


        nsamples_test, nx_test, ny_test = X_test.shape

        if self.which_model == 'MLP':

            X_test = X_test.reshape((nsamples_test, nx_test * ny_test))

        if self.which_model == 'CNN':

            X_test = X_test[:, :, :, np.newaxis]

        if self.which_model == 'LSTM':

            X_test = X_test

        if self.which_model == 'CNNLSTM':

            X_test = X_test[:, :, :, np.newaxis]

        if self.which_model == 'ConvLSTM2D':
            X_test=X_test[:nsamples_test//25*25,:,:]

            X_test = X_test.reshape((X_test.shape[0]//25, 25, nx_test,  ny_test, 1))


        # print('X_test size: ' + str(X_test.shape))

        return X_test, Y_test

    def leave_one_out_data(self, K):

        self.K_fold = len(self.subjects_train_valid_with_data)

        self.K = K

        p_CV = np.concatenate([self.subjects_train_valid_with_data, self.subjects_train_valid_with_data])

        self.K_fold_I = np.array([p_CV[i] for i in range(0, 12)])
        self.Test_I = np.array(self.subjects_test_with_data)

        self.current_K = K

        I_train = self.K_fold_I[[x for x in range(self.K_fold) if x != self.current_K]].reshape([-1])
        I_valid = self.K_fold_I[self.current_K]

        X_train = np.concatenate([self.X_train_valid_dict[s] for s in I_train], axis=0)
        X_valid = self.X_train_valid_dict[I_valid]

        Y_train = np.concatenate([self.Y_train_valid_dict[s] for s in I_train], axis=0)
        Y_valid = self.Y_train_valid_dict[I_valid]

        nsamples_train, nx_train, ny_train = X_train.shape
        nsamples_valid, nx_valid, ny_valid = X_valid.shape

        if self.which_model == 'MLP':

            X_train = X_train.reshape((nsamples_train, nx_train * ny_train))
            X_valid = X_valid.reshape((nsamples_valid, nx_valid * ny_valid))

        if self.which_model == 'CNN':

            X_train = X_train[:, :, :, np.newaxis]
            X_valid = X_valid[:, :, :, np.newaxis]

        if self.which_model == 'RNN':

            X_train = X_train
            X_valid = X_valid

        return X_train, X_valid, Y_train, Y_valid

