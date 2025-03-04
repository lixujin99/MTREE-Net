import numpy as np
import random
import os
import warnings
from scipy import signal
from scipy.io import loadmat

def load_data_sd(task, subjects, blocks, resolution):
    data = None
    label = None
    id_block = None
    idx = 0

    for blk in blocks:
        path_eeg = './%s/EEG/%s_%d.npz'%(task, subjects, blk)
        mat = np.load(path_eeg)
        data_eeg = mat['data']
        data_eeg = signal.resample(data_eeg, resolution, axis=-1)
        label_eeg = mat['label']

        path_eye = './%s/EM/%s_%d.mat'%(task, subjects, blk)
        mat = loadmat(path_eye)
        data_eye = mat['data']
        data_eye = np.transpose(data_eye,(0,2,1))
        data_eye = data_eye[:,:6,:]
        data_eye = signal.resample(data_eye, resolution, axis=-1)
        label_eye = mat['label']
        label_eye = np.squeeze(label_eye) 

        print((label_eeg == label_eye).all())
        data_sub = np.concatenate((data_eeg, data_eye),axis=-2)
        label_sub = label_eeg

        data = data_sub if data is None else np.concatenate((data,data_sub),axis=0)
        label = label_sub if label is None else np.concatenate((label,label_sub),axis=0)
        id_block = np.ones(data_sub.shape[0])*idx if id_block is None else np.concatenate((id_block,np.ones(data_sub.shape[0])*idx),axis=0)
        idx += 1

    return data, label, id_block
