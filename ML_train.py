import os
import time
import warnings
import numpy as np
import scipy.stats
import seaborn as sns
import pandas as pd
import itertools
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
matplotlib.use('TKAgg')

import ML_modules as ml
from Classes.Path import Path_info
from Classes.Data_2 import Load_data, Create_DB, Up_sampling

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


DB_N = 0
subject = 0

# path_info = Path_info(subject_ind=subject)
# path_info.get_DB_path_2(DB_N=DB_N, delete_folder=False)
# path_info.get_DB_info_2(DB_N=DB_N)

hyper_DA = {'n': 0, 'random_noise_factor': 0.01, 'augmentation_types': 'rn'}

# K_fold = 6
# accuracies = np.empty([K_fold])
# pearson = np.empty([K_fold])
# y_test_pred = {}
# y_test_true = {}

model_names_classification = ['SVM','RF','MLP','KNN_Classifier','DecisionTreeClassifier']

ml_class = ml.Machine_Learning_classification(DB_N=DB_N)
df_classification,Test_acc_str,Test_pearson_str,y_pred,y_true = ml_class.perform_classification(model_names=model_names_classification, hyper_DA=hyper_DA, use_accuracy=True)