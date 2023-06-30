import os
import shutil
import itertools
from tabulate import tabulate
import pandas as pd
import random
import os.path
import numpy as np

# This code is used to create the database. all the information about the Database
# are gathered in Path Classe with information such as the number of subjects used to train / validate / test
# Window_size with step or if each stride must be cut separately (only work if there is GRF)

class Path_info():

    def __init__(self, subject_ind):

        pass

subject_ind = 3
self = Path_info(subject_ind=subject_ind)

class Path_info():

    def __init__(self, subject_ind):

        self.subject_ind = subject_ind
        self.subject_name = str(subject_ind)

        self.exercises_name = ['bent_row','lat_raise', 'sh_press']

        # Sampling frequency
        self.fs_camera = 100

        self.imu_frame_rate = 100

        self.cwd = os.getcwd()
        self.path_abs = os.path.abspath(os.path.join(self.cwd, os.pardir))

        self.path_experiment_data_subject = os.path.join(self.path_abs ,'Data Experiment','Subject'+str(self.subject_ind + 1).zfill(2))

        self.exercise_list_file = os.path.join(self.path_abs, 'Data Experiment','Subject' + str(self.subject_ind+1).zfill(2), "Timing_sheet.xlsx")

        self.data_subject_path_IMU = os.path.join(self.path_abs,'Data Experiment','Subject' + str(self.subject_ind + 1).zfill(2),'IMU')
        self.data_subject_path_markers = os.path.join(self.path_abs, 'Data Experiment','Subject' + str(self.subject_ind + 1).zfill(2), 'Markers')
        self.data_subject_path_others = os.path.join(self.path_abs, 'Data Experiment','Subject', str(self.subject_ind + 1).zfill(2),'Others')

        is_exist = os.path.isfile(self.exercise_list_file)

        if is_exist:
            print("File " + self.exercise_list_file + " exist")
        #else:
            #raise Warning("File " + self.exercise_list_file + " does not exist")

        self.path_base = os.path.join(self.path_abs, 'Data')

        self.results_ML_path = os.path.normpath("Results_ML\\")
        self.results_DL_path = os.path.normpath("Results_DL\\")

        self.tensorboard_path = os.path.normpath("Output\\")

        if not os.path.exists(self.results_ML_path):
            os.makedirs(self.results_ML_path)
        if not os.path.exists(self.results_DL_path):
            os.makedirs(self.results_DL_path)
        if not os.path.exists(self.tensorboard_path):
            os.makedirs(self.tensorboard_path)

        self.path_markers = os.path.join(self.path_base, self.subject_name, 'Markers')
        self.path_markers_cut = os.path.join(self.path_base,self.subject_name, 'Markers_cut')
        self.path_markers_cut_spline = os.path.join(self.path_base, self.subject_name, 'Markers_cut_spline')

        if not os.path.exists(self.path_markers):
            os.makedirs(self.path_markers)
        if not os.path.exists(self.path_markers_cut):
            os.makedirs(self.path_markers_cut)
        if not os.path.exists(self.path_markers_cut_spline):
            os.makedirs(self.path_markers_cut_spline)

        self.MM_path = os.path.join(self.path_base, self.subject_name, 'MM')
        self.path_mass = os.path.join(self.path_base, self.subject_name, 'Mass')


        self.subject_name = str(subject_ind)

        if not os.path.exists(self.MM_path):
            os.makedirs(self.MM_path)
        if not os.path.exists(self.path_mass):
            os.makedirs(self.path_mass)

        self.path_IMU = os.path.join(self.path_base, self.subject_name, 'IMU')
        self.path_IMU_cut = os.path.join(self.path_base, self.subject_name, 'IMU_cut')
        self.path_IMU_cut_spline = os.path.join(self.path_base, self.subject_name,'IMU_cut_spline')
        self.path_IMU_forecast_spline = os.path.join(self.path_base, self.subject_name, 'IMU_forecast_spline')

        if not os.path.exists(self.path_IMU):
            os.makedirs(self.path_IMU)
        if not os.path.exists(self.path_IMU_cut):
            os.makedirs(self.path_IMU_cut)
        if not os.path.exists(self.path_IMU_cut_spline):
            os.makedirs(self.path_IMU_cut_spline)

        self.path_points = os.path.join(self.cwd, 'Points', self.subject_name)
        if not os.path.exists(self.path_points):
            os.makedirs(self.path_points)



        self.usable_exercises_ind = [0, 1, 2]
        self.set_points = {}

        self.markers = ['SHOULDER_L', 'SHOULDER_R', 'C7', 'ELBOW_L_LAT', 'ELBOW_L_MED', 'WRIST_L_LAT', 'WRIST_L_MED',
                       'ELBOW_R_LAT', 'ELBOW_R_MED', 'WRIST_R_LAT', 'WRIST_R_MED', 'ASIS_L', 'ASIS_R', 'PSIS_L', 'PSIS_R',
                       'KNEE_L_LAT', 'KNEE_L_MED', 'ANKLE_L_LAT', 'ANKLE_L_MED', 'FOOT_L_LAT', 'FOOT_L_MED',
                       'HEEL_L', 'KNEE_R_LAT', 'KNEE_R_MED', 'ANKLE_R_LAT', 'ANKLE_R_MED', 'FOOT_R_LAT', 'FOOT_R_MED',
                       'HEEL_R', 'HEAD_T', 'HEAD_L', 'HEAD_R', 'STERN_T', 'STERN_L', 'STERN_R', 'SHOUL_T', 'SHOUL_L', 'SHOUL_R',
                       'uARML_T', 'uARML_L', 'uARML_R', 'fARML_T', 'fARML_L', 'fARML_R', 'HANDL_T', 'HANDL_L', 'HANDL_R',
                       'SHOUR_T', 'SHOUR_L', 'SHOUR_R', 'uARMR_T', 'uARMR_L', 'uARMR_R', 'fARMR_T', 'fARMR_L', 'fARMR_R',
                       'HANDR_T', 'HANDR_L', 'HANDR_R', 'PELV_T', 'PELV_L', 'PELV_R', 'uLEGL_T', 'uLEGL_L', 'uLEGL_R',
                       'lLEGL_T', 'lLEGL_L', 'lLEGL_R', 'FOOTL_T', 'FOOTL_L', 'FOOTL_R', 'uLEGR_T', 'uLEGR_L', 'uLEGR_R',
                       'lLEGR_T', 'lLEGR_L', 'lLEGR_R', 'FOOTR_T', 'FOOTR_L', 'FOOTR_R']


    def get_DB_path(self, DB_N=0, delete_folder=False):

        self.path_DB = os.path.join(self.path_base,"Database")
        self.path_DB_N = os.path.join(self.path_DB,"XY_" + str(DB_N))

        if delete_folder:
            if os.path.exists(self.path_DB_N) and os.path.isdir(self.path_DB_N):
                shutil.rmtree(self.path_DB_N)

        if not os.path.exists(self.path_DB_N):
            os.makedirs(self.path_DB_N)

        # Train/Validate/Test path
        self.path_DB_base_train_valid_test = os.path.join(self.path_DB_N, "Train_valid_test")

        if not os.path.exists(self.path_DB_base_train_valid_test):
            os.makedirs(self.path_DB_base_train_valid_test)

    def get_DB_info(self, DB_N=0, display_info=True):

        parameters = {'window_size': [200],
                      'side': ['both'],
                      'which_data': [['IMU']],
                      'test_set':[0],
                      'percentage_data_to_use': [1.0],
                      'percentage_data_to_train': [0.70],
                      'percentage_data_to_valid': [0.15],
                      'percentage_data_to_test': [0.15],
                      'exercises': [['bent_row','lat_raise', 'sh_press']],
                      'what_kind_of_prediction':  ['classification']}

        keys, values = zip(*parameters.items())
        configurations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        configuration = configurations[DB_N]

        if display_info:
            print("Configuration number " + str(DB_N+1) + " among " + str(len(configurations)) + " different configurations")
            for k in sorted(configuration.keys()):
                print(str(k) + ": " + str(configuration[k]))
            df = pd.DataFrame.from_dict(configurations)
            print(tabulate(df, headers='keys', tablefmt='psql'))

        self.n_configurations = len(configurations)
        self.configuration = configuration

        self.side = self.configuration['side']
        self.window_size = self.configuration['window_size']
        self.exercises = self.configuration['exercises']

        self.which_data = self.configuration['which_data']
        self.percentage_data_to_use = self.configuration['percentage_data_to_use']
        self.what_kind_of_prediction = self.configuration['what_kind_of_prediction']
        self.percentage_data_to_train = self.configuration['percentage_data_to_train']
        self.percentage_data_to_valid = self.configuration['percentage_data_to_valid']
        self.percentage_data_to_test = self.configuration['percentage_data_to_test']

        def rotate(l, n):
            return l[-n:] + l[:-n]

        r = [i for i in range(0, 15)]
        r = [0,1,2,3,5,6,8,9,11,12,15,16,17,18,19,20,21,22,23,24,25,26,27,28]
        import random
        random.shuffle(r)
        l = rotate(r,3)
        l=rotate(l,3)
        l = rotate(l, 3)

        self.DB_info = {}

        if self.which_data == ['IMU']:

            self.n_subjects_train = 1
            self.n_subjects_valid = 1
            self.n_subjects_test = 1

            self.n_subjects_train = 30
            self.n_subjects_valid = 3
            self.n_subjects_test = 3

            self.test_set = [[19,18,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20],  # without pain
                              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20], #shoulder pain
                              [9, 8, 14, 13, 11, 3, 1, 5, 4, 12, 2, 10, 6],
                              [2, 10, 6, 9, 8, 14, 13, 11, 3, 1, 5, 4, 12]]

            self.subjects = self.test_set[self.configuration['test_set']]

            self.subjects_train_valid = self.subjects[:self.n_subjects_train + self.n_subjects_valid]
            self.subjects_test = self.subjects[-self.n_subjects_test:]

            # TODO:
            self.IMU_names = ['timestamp','acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z']
            self.dof_to_train_ind = [i for i in range(0, 6)]

            self.Y_names = self.IMU_names
            self.dof_to_train_ind = self.dof_to_train_ind


        self.figure_ML_path_DB = os.path.join('Figures_ML','DB_'+ str(DB_N))
        self.figure_DL_path_DB = os.path.join('Figures_DL','DB_' + str(DB_N))

        self.results_ML_path_DB = os.path.join(self.results_ML_path,'DB_' + str(DB_N))
        self.results_DL_path_DB = os.path.join(self.results_DL_path,'DB_' + str(DB_N))

        self.NN_path_DB = os.path.join('NN','DB_' + str(DB_N))
        self.tensorboard_path_DB = os.path.join('Output','DB_' + str(DB_N))
        self.path_save_data_app_DB = os.path.join('App','DB_' + str(DB_N), 'XY')

        if not os.path.exists(self.figure_ML_path_DB):
            os.makedirs(self.figure_ML_path_DB)
        if not os.path.exists(self.figure_DL_path_DB):
            os.makedirs(self.figure_DL_path_DB)
        if not os.path.exists(self.results_ML_path_DB):
            os.makedirs(self.results_ML_path_DB)
        if not os.path.exists(self.results_DL_path_DB):
            os.makedirs(self.results_DL_path_DB)
        if not os.path.exists(self.path_save_data_app_DB):
            os.makedirs(self.path_save_data_app_DB)
        if not os.path.exists(self.NN_path_DB):
            os.makedirs(self.NN_path_DB)
        if not os.path.exists(self.tensorboard_path_DB):
            os.makedirs(self.tensorboard_path_DB)

    def get_DB_path_2(self, DB_N=0, delete_folder=False):

        self.path_DB_2 = os.path.join(self.path_base, "Database2")
        self.path_DB_N_2 = os.path.join(self.path_DB_2, "XY_" + str(DB_N))

        if delete_folder:
            if os.path.exists(self.path_DB_N_2) and os.path.isdir(self.path_DB_N_2):
                shutil.rmtree(self.path_DB_N_2)

        if not os.path.exists(self.path_DB_N_2):
            os.makedirs(self.path_DB_N_2)

        # Train/Validate/Test path
        self.path_DB_base_train_valid_test_2 = os.path.join(self.path_DB_N_2, "Train_valid_test")

        if not os.path.exists(self.path_DB_base_train_valid_test_2):
            os.makedirs(self.path_DB_base_train_valid_test_2)

    def get_DB_info_2(self, DB_N=0, display_info=True):

        parameters = {'window_size': [200],
                      'side': ['both'],
                      'which_data': [['IMU']],
                      'test_set': [0],
                      'percentage_data_to_use': [1.0],
                      'percentage_data_to_train': [0.70],
                      'percentage_data_to_valid': [0.15],
                      'percentage_data_to_test': [0.15],
                      'exercises': [['bent_row', 'lat_raise', 'sh_press']],
                      'what_kind_of_prediction': ['classification']}

        keys, values = zip(*parameters.items())
        configurations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        configuration = configurations[DB_N]

        if display_info:
            print("Configuration number " + str(DB_N + 1) + " among " + str(
                len(configurations)) + " different configurations")
            for k in sorted(configuration.keys()):
                print(str(k) + ": " + str(configuration[k]))
            df = pd.DataFrame.from_dict(configurations)
            print(tabulate(df, headers='keys', tablefmt='psql'))

        self.n_configurations = len(configurations)
        self.configuration = configuration

        self.side = self.configuration['side']
        self.window_size = self.configuration['window_size']
        self.exercises = self.configuration['exercises']

        self.which_data = self.configuration['which_data']
        self.percentage_data_to_use = self.configuration['percentage_data_to_use']
        self.what_kind_of_prediction = self.configuration['what_kind_of_prediction']
        self.percentage_data_to_train = self.configuration['percentage_data_to_train']
        self.percentage_data_to_valid = self.configuration['percentage_data_to_valid']
        self.percentage_data_to_test = self.configuration['percentage_data_to_test']

        def rotate(l, n):
            return l[-n:] + l[:-n]

        self.r = [i for i in range(0, 15)]
        self.r = [1,2,3,4,5,6,8,9,10,11,12,13,14]
        import random
        random.shuffle(self.r)
        l = rotate(self.r,3)
        l=rotate(l,3)
        l = rotate(l, 3)


        self.DB_info = {}

        if self.which_data == ['IMU']:

            self.subjects = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]

            self.IMU_names = ['timestamp','acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z']
            self.dof_to_train_ind = [i for i in range(0, 36)]

            self.Y_names = self.IMU_names
            self.dof_to_train_ind = self.dof_to_train_ind


        self.figure_ML_path_DB = os.path.join('Figures_ML','DB2_' + str(DB_N))
        self.figure_DL_path_DB = os.path.join('Figures_DL','DB2_' + str(DB_N))

        self.results_ML_path_DB = os.path.join(self.results_ML_path,"DB2_" + str(DB_N))
        self.results_DL_path_DB = os.path.join(self.results_DL_path, "DB2_" + str(DB_N))

        self.NN_path_DB = os.path.join('NN','DB2_' + str(DB_N))
        self.tensorboard_path_DB = os.path.join('Output','DB2_'+ str(DB_N))
        self.path_save_data_app_DB = os.path.join('App','DB2_' + str(DB_N),"XY")

        if not os.path.exists(self.figure_ML_path_DB):
            os.makedirs(self.figure_ML_path_DB)
        if not os.path.exists(self.figure_DL_path_DB):
            os.makedirs(self.figure_DL_path_DB)
        if not os.path.exists(self.results_ML_path_DB):
            os.makedirs(self.results_ML_path_DB)
        if not os.path.exists(self.results_DL_path_DB):
            os.makedirs(self.results_DL_path_DB)
        if not os.path.exists(self.path_save_data_app_DB):
            os.makedirs(self.path_save_data_app_DB)
        if not os.path.exists(self.NN_path_DB):
            os.makedirs(self.NN_path_DB)
        if not os.path.exists(self.tensorboard_path_DB):
            os.makedirs(self.tensorboard_path_DB)