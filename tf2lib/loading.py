import os
import numpy as np
import pywt
from openmovement.load import CwaData

def load_cwa(filename):
    """
    Input: Filename (+ dir) where the IMU data is located
    Output: Pandas array with sub-sampled measurements
    """
    with CwaData(filename, include_gyro=True, include_temperature=False) as cwa_data:
        samples = cwa_data.get_samples()
        samples = samples.set_index('time')
        samples = samples.astype('float32')
        samples = samples.resample('500ms').sum() # Sub-sampled data
    return samples


def window_data(X,w_l=64):
    """
    Windowing function
    Inputs
    1) X --> Data to window, with shape [num. temp. samples] x [num. features]
    2) w_l --> Size of the windows
    Output:
    1) Array of shape [num. batches] x [window width] x [num.features] x [1]
    """
    X = np.transpose(X) 
    nt, nf = X.shape
    nb = 2*nt//w_l-1
    X_w = np.zeros((nb,w_l,nf),dtype=np.float32)
    for k in range(nb):
        X_w[k,:,:] = X[k*(w_l//2):(k+1)*(w_l//2)+(w_l//2),:]
    return np.expand_dims(X_w, axis=-1) # shape: (nb,w_l,nf,1)


def load_data(dir_dataset, pat_file_names):
    """
    Load data and creates a dictionary.

    Parameters:
    dir_dataset: directory of dataset files.

    Returns:
    Dictionary containing all files.

    """
    ini_flag = False
    for p in pat_file_names:
        print('Patient No.',p)
        for cwa_f in os.listdir(os.path.join(dir_dataset, p, 'IMU')):
            print('  Analyzing file',cwa_f)
            cwt_flag = False
            id_num = cwa_f.split('_')[1]
            file_name = os.path.join(dir_dataset, p, 'IMU', cwa_f)
            df = load_cwa(file_name)
            sensor_idxs = list(df.keys())
            for sens_idx in sensor_idxs:
                print('    Data from sensor',sens_idx)
                if sens_idx.split('_')[0] == 'gyro':
                    scales = np.arange(1,641,10)
                else:
                    scales = np.arange(1,1025,16)
                # apply  PyWavelets continuous wavelet transfromation function
                coeffs, freqs = pywt.cwt(df[sens_idx], scales, wavelet = 'mexh')
                coeffs_w = window_data(coeffs)
                if not(cwt_flag):
                    X_file = coeffs_w
                    cwt_flag = True
                else:
                    X_file = np.concatenate((X_file,coeffs_w),axis=-1)
            if not(ini_flag):
                X = X_file
                y = [bytes(cwa_f.split('.')[0],'utf-8') for _ in range(X_file.shape[0])]
                ini_flag = True
            else:
                X = np.concatenate((X,X_file), axis=0)
                y += [bytes(cwa_f.split('.')[0],'utf-8') for _ in range(X_file.shape[0])]
    return X, y

