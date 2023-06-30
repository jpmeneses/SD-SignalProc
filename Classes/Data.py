# user-dependent classification
# sliding window
import numpy as np
import matplotlib.pyplot as plt
from Classes.Path import Path_info
#from tensorflow.keras.utils import to_categorical
import os
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
import random
#from transforms3d.axangles import axangle2mat
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

    dataOut[np.isnan(dataOut)] = 0

    return dataOut, mean, std
def standardize_with_mean_and_sd(data, mean, std):
    dataOut = data.copy()

    for i in range(dataOut.shape[2]):
        dataOut[:, :, i] = (data[:, :, i] - mean[i]) / std[i]

    dataOut[np.isnan(dataOut)] = 0

    return dataOut
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
class Up_sampling():

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        #self.Y.squeeze()
        self.num_label = len(np.unique(self.Y, axis=0))
        self.unique_label = np.unique(self.Y, axis=0)
        self.sample = self.X.shape[0]
        self.height = self.X.shape[1]
        #self.width = self.X.shape[2]
        b = []
        for uniq in self.unique_label:
            a = np.asscalar(uniq)
            b.append(a)
        self.unique_label=b

    def upsampling(self):
        n_label = {}
        index_label = {}
        X_new_dict = {}
        Y_new_dict = {}
        for uniq in self.unique_label:
            #uniq=np.asscalar(uniq)
            index_label[uniq] = np.where(self.Y == uniq)[0]
            n_label[uniq] = len(np.where(self.Y == uniq)[0])
            print('Label ' + str(uniq) + " has " + str(n_label[uniq]) + ' samples')
        n_majority_label = np.max([n_label[k] for k in n_label.keys()])
        for uniq in self.unique_label:
            #uniq = np.asscalar(uniq)
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
            print('Label ' + str(uniq) + " has " + str(n_label_new[uniq]) + ' samples after upsampling')
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

        path_info = Path_info(subject_ind=0)
        path_info.get_DB_path_2(DB_N=DB_N, delete_folder=False)
        path_info.get_DB_info_2(DB_N=DB_N)

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


        self.window_size = path_info.window_size
        self.which_data = path_info.which_data
        self.exercises = path_info.exercises

        self.percentage_data_to_train = path_info.percentage_data_to_train
        self.percentage_data_to_valid = path_info.percentage_data_to_valid
        self.percentage_data_to_test = path_info.percentage_data_to_test
        self.what_kind_of_prediction = path_info.what_kind_of_prediction

        self.path_DB_N = path_info.path_DB_N_2
        self.path_DB_base_train_valid_test = path_info.path_DB_base_train_valid_test_2
        self.path_DB = path_info.path_DB_2

        self.subjects = path_info.subjects

        # subject = 2
        self.data_X = {}
        self.data_Y = {}

        for subject in self.subjects:

            path_info = Path_info(subject_ind=subject)

            if self.which_data == ['IMU']:
                path = path_info.path_IMU_cut
                self.dof_to_train_ind = [i for i in range(0, 6)]

            name = "IMU_slided_Wrist.npy"
            X=np.load(os.path.join(path_info.path_IMU_cut, name),allow_pickle=True)
            name = "label_slided_Wrist.npy"
            Y=np.load(os.path.join(path_info.path_IMU_cut, name),allow_pickle=True)


            self.data_X[subject] = X
            self.data_Y[subject] = Y


            data_augmentation = Data_augmentation(X, Y, hyper_DA=self.hyper_DA)
            data_augmentation.run()

            #if type(data_augmentation.X_aug) is np.ndarray:
                #self.data_X[subject] = np.concatenate([X, data_augmentation.X_aug], axis=0)
                #self.data_Y[subject] = np.concatenate([Y, data_augmentation.Y_aug], axis=0)

            #else:
                #self.data_X[subject] = X
                #self.data_Y[subject] = Y


    def create_database(self,KK, plotIt=False):

        subjects_with_data = list(self.data_X.keys())

        data_X = {}
        data_Y = {}

        X_data_std= {}
        X_mean= {}
        X_std= {}

        for subject in subjects_with_data:

            X_data_std[subject] = {}
            X_mean[subject] = {}
            X_std[subject] = {}
            data_X[subject] = {}
            data_Y[subject] = {}

            data_X[subject] = self.data_X[subject]
            data_Y[subject] = self.data_Y[subject]

            if self.which_data == ['IMU']:
                # TODO: for now we don't normalize the output, it is a regression task for now
                inds_X = [[i] for i in range(data_X[subject].shape[2])]
                # Compute global mean and std for train/validation set
                X_data_std[subject], X_mean[subject], X_std[subject] = standardize(data=data_X[subject], inds=inds_X)

        # Apply mean and std to train/validation/test set with mean/std from train/validation set
        data_X_std = {}
        n_samples = {}

        for subject in subjects_with_data:

            data_X_std[subject] = {}
            n_samples[subject] = {}


            data_X_std[subject] = standardize_with_mean_and_sd(self.data_X[subject], X_mean[subject], X_std[subject])
            n_samples[subject] = self.data_X[subject].shape[0]

        name = "X_Mean.npy"
        np.save(os.path.join(self.path_DB_N, name), X_mean)
        name = "X_sd.npy"
        np.save(os.path.join(self.path_DB_N, name), X_std)

        data_Y = {}


        for subject in subjects_with_data:

            X = data_X_std[subject]
            Y = self.data_Y[subject]

            p = np.array([i for i in range(X.shape[0])])
            random.shuffle(p)
            self.p = p
            n = int(X.shape[0] * (self.percentage_data_to_train+self.percentage_data_to_valid))

            self.K_fold=2 # outer loop

            self.percentage_data_to_train_valid = self.percentage_data_to_train + self.percentage_data_to_valid
            n_X_train_valid = list(range(X.shape[0]))
            self.percentage_data_to_valid = (X.shape[0] / self.K_fold) / X.shape[0]

            p_CV = np.concatenate([n_X_train_valid, n_X_train_valid])
            n = int(X.shape[0] * self.percentage_data_to_valid)

            self.K_fold_I = np.array([p_CV[i * n:i * n + n] for i in range(0, self.K_fold)])

            self.K = KK

            # self.percentage_data_to_train_valid = self.percentage_data_to_train+self.percentage_data_to_valid
            # n_X_train_valid=list(range(self.X_train_valid.shape[0]))
            # self.percentage_data_to_valid = (self.X_train_valid.shape[0]/self.K_fold)/self.X_train_valid.shape[0]
            #
            # p_CV = np.concatenate([n_X_train_valid, n_X_train_valid])
            # n = int(self.X_train_valid.shape[0] * self.percentage_data_to_valid)
            #
            # self.K_fold_I = np.array([p_CV[i*n:i*n+n] for i in range(0, self.K_fold)])

            self.current_K = KK

            I_train_valid = self.K_fold_I[[x for x in range(self.K_fold) if x != self.current_K]].reshape([-1])
            I_test = self.K_fold_I[self.current_K]

            X_train_valid = X[p][I_train_valid]
            X_test = X[p][I_test]

            Y_train_valid = Y[p][I_train_valid]
            Y_test = Y[p][I_test]

            Y_train_valid_orig = Y[p][I_train_valid]
            Y_test_orig = Y[p][I_test]

            #nsamples_train, nx_train, ny_train = X_train.shape
            #nsamples_valid, nx_valid, ny_valid = X_valid.shape

            name = "n_tv_" + str(subject).zfill(2) + ".npy"
            np.save(os.path.join(self.path_DB_base_train_valid_test,  name), I_train_valid)
            name = "X_train_valid_" + str(subject).zfill(2) + ".npy"
            np.save(os.path.join(self.path_DB_base_train_valid_test , name), X_train_valid)
            name = "Y_train_valid_" + str(subject).zfill(2) + ".npy"
            np.save(os.path.join(self.path_DB_base_train_valid_test, name), Y_train_valid)
            name = "Y_train_valid_orig_" + str(subject).zfill(2) + ".npy"
            np.save(os.path.join(self.path_DB_base_train_valid_test, name), Y_train_valid_orig)

            name = "n_test_" + str(subject).zfill(2) + ".npy"
            np.save(os.path.join(self.path_DB_base_train_valid_test, name), I_test)

            name = "X_test_" + str(subject).zfill(2) + ".npy"
            np.save(os.path.join(self.path_DB_base_train_valid_test, name), X_test)
            name = "Y_test_" + str(subject).zfill(2) + ".npy"
            np.save(os.path.join(self.path_DB_base_train_valid_test, name), Y_test)
            name = "Y_test_orig_" + str(subject).zfill(2) + ".npy"
            np.save(os.path.join(self.path_DB_base_train_valid_test, name), Y_test_orig)

            data_X_std[subject] = X
            data_Y[subject] = Y

        name = "n_samples.npy"
        np.save(os.path.join(self.path_DB_N, name), n_samples)

        if plotIt:

            self.fig = plt.figure()
            self.ax1 = plt.subplot(1, 1, 1)

            data_forces_to_plot = X[:, :, :]

            for i in range(data_forces_to_plot.shape[0]):

                self.ax1.plot(data_forces_to_plot[i])

            self.fig.suptitle('Spline - Set ')


class Load_data():

    def __init__(self, DB_N, which_model='MLP', subject=1):

        self.DB_N = DB_N
        self.which_model = which_model
        self.subject = subject

        path_info = Path_info(subject_ind=1)
        path_info.get_DB_path_2(DB_N=DB_N, delete_folder=False)
        path_info.get_DB_info_2(DB_N=DB_N)

        self.percentage_data_to_train = path_info.percentage_data_to_train
        self.percentage_data_to_valid = path_info.percentage_data_to_valid
        self.percentage_data_to_test = path_info.percentage_data_to_test

        self.which_data = path_info.which_data

        self.K_fold = 2

        self.what_kind_of_prediction = path_info.what_kind_of_prediction

        if self.what_kind_of_prediction == 'classification':

            self.num_classes = 3


        self.path_DB_N = path_info.path_DB_N_2
        self.path_DB_base_train_valid_test = path_info.path_DB_base_train_valid_test_2

        name = "X_Mean.npy"
        X_Mean = np.load(os.path.join(self.path_DB_N, name), allow_pickle=True)
        name = "X_sd.npy"
        X_sd = np.load(os.path.join(self.path_DB_N, name), allow_pickle=True)

        name = "n_samples.npy"
        n_samples = np.load(os.path.join(self.path_DB_N, name), allow_pickle=True)

        s = self.subject
        name = "X_train_valid_" + str(s).zfill(2) + ".npy"
        self.X_train_valid = np.load(os.path.join(self.path_DB_base_train_valid_test, name), allow_pickle=True)

        name = "Y_train_valid_" + str(s).zfill(2) + ".npy"
        self.Y_train_valid = np.load(os.path.join(self.path_DB_base_train_valid_test, name), allow_pickle=True)

        name = "Y_train_valid_orig_" + str(s).zfill(2) + ".npy"
        self.Y_train_valid_orig = np.load(os.path.join(self.path_DB_base_train_valid_test, name), allow_pickle=True)

        name = "X_test_" + str(s).zfill(2) + ".npy"
        self.X_test = np.load(os.path.join(self.path_DB_base_train_valid_test, name), allow_pickle=True)

        name = "Y_test_" + str(s).zfill(2) + ".npy"
        self.Y_test = np.load(os.path.join(self.path_DB_base_train_valid_test, name), allow_pickle=True)

        name = "Y_test_orig_" + str(s).zfill(2) + ".npy"
        self.Y_test_orig = np.load(os.path.join(self.path_DB_base_train_valid_test, name), allow_pickle=True)


        #if self.what_kind_of_prediction == 'classification':

            #self.Y_test = to_categorical(self.Y_test, num_classes=self.num_classes)


        self.percentage_data_to_train_valid = self.percentage_data_to_train + self.percentage_data_to_valid
        n_X_train_valid = list(range(self.X_train_valid.shape[0]))
        self.percentage_data_to_valid = (self.X_train_valid.shape[0] / self.K_fold) / self.X_train_valid.shape[0]

        p_CV = np.concatenate([n_X_train_valid, n_X_train_valid])
        n = int(self.X_train_valid.shape[0] * self.percentage_data_to_valid)

        self.K_fold_I = np.array([p_CV[i * n:i * n + n] for i in range(0, self.K_fold)])

    def K_fold_data(self, K):
        # K=0

        self.K = K

        # self.percentage_data_to_train_valid = self.percentage_data_to_train+self.percentage_data_to_valid
        # n_X_train_valid=list(range(self.X_train_valid.shape[0]))
        # self.percentage_data_to_valid = (self.X_train_valid.shape[0]/self.K_fold)/self.X_train_valid.shape[0]
        #
        # p_CV = np.concatenate([n_X_train_valid, n_X_train_valid])
        # n = int(self.X_train_valid.shape[0] * self.percentage_data_to_valid)
        #
        # self.K_fold_I = np.array([p_CV[i*n:i*n+n] for i in range(0, self.K_fold)])

        self.current_K = K

        I_train = self.K_fold_I[[x for x in range(self.K_fold) if x != self.current_K]].reshape([-1])
        I_valid = self.K_fold_I[self.current_K]

        X_train = self.X_train_valid[I_train]
        X_valid = self.X_train_valid[I_valid]

        Y_train = self.Y_train_valid[I_train]
        Y_valid = self.Y_train_valid[I_valid]

        Y_orig_train = self.Y_train_valid_orig[I_train]
        Y_orig_valid = self.Y_train_valid_orig[I_valid]

        nsamples_train, nx_train, ny_train = X_train.shape
        nsamples_valid, nx_valid, ny_valid = X_valid.shape
        #nsamples_trainY, nx_trainY, ny_trainY = Y_train.shape
        #nsamples_validY, nx_validY, ny_validY = Y_valid.shape

        if self.which_model == 'MLP':

            X_train = X_train.reshape((nsamples_train, nx_train * ny_train))
            X_valid = X_valid.reshape((nsamples_valid, nx_valid * ny_valid))
            #Y_train = Y_train.reshape((nsamples_trainY, nx_trainY * ny_trainY))
            #Y_valid = Y_valid.reshape((nsamples_validY, nx_validY * ny_validY))

        if self.which_model == 'CNN':

            X_train = X_train[:, :, :, np.newaxis]
            X_valid = X_valid[:, :, :, np.newaxis]

        if self.which_model == 'RNN':

            X_train = X_train
            X_valid = X_valid

        print('X_train size: ' + str(X_train.shape))
        print('X_valid size: ' + str(X_valid.shape))

        return X_train, X_valid, Y_train, Y_valid, Y_orig_train, Y_orig_valid

    def test_data(self):

        n = self.X_train_valid.shape[0]-self.K_fold_I[-1][-1]-1

        if n <=0:
            X_test = self.X_test
            Y_test = self.Y_test
            Y_orig_test = self.Y_test_orig

        else:
            X_test = np.concatenate([self.X_train_valid[-n:], self.X_test], axis=0)
            Y_test = np.concatenate([self.Y_train_valid[-n:], self.Y_test], axis=0)
            Y_orig_test = np.concatenate([self.Y_train_valid_orig[-n:], self.Y_test_orig], axis=0)



        nsamples_test, nx_test, ny_test = X_test.shape
        #nsamples_testY, nx_testY, ny_testY = Y_test.shape

        if self.which_model == 'MLP':

            X_test = X_test.reshape((nsamples_test, nx_test * ny_test))
            #Y_test = Y_test.reshape((nsamples_testY, nx_testY * ny_testY))

        if self.which_model == 'CNN':

            X_test = X_test[:, :, :, np.newaxis]

        if self.which_model == 'RNN':

            X_test = X_test

        print('X_test size: ' + str(X_test.shape))

        return X_test, Y_test, Y_orig_test
