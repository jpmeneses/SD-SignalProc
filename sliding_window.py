import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mode

from Classes.Path import Path_info
from utils import Cut_data

subject_ind = 2

cut_data = Cut_data(subject_ind=subject_ind)
IMU_filt = cut_data.cut_exercise()  # save IMU segmented data
X, X2 = cut_data.split_sequence_IMU(n_steps_in=200, n_steps_out=5, lag_value=200)  # X2 is next window(no used)
Y, Y2, Y_window = cut_data.split_sequence_label(n_steps_in=200, n_steps_out=5, lag_value=200)  # Y2 is next window(no used)