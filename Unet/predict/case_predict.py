#!/usr/bin/python
# -*- coding: UTF-8 -*-

import cv2
import os
import glob
import nibabel as nib
import numpy as np
from keras import backend as K
from keras.models import load_model
import tensorflow as tf
import copy
from keras.models import Model
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# normalization methods
def MaxMinNorm(data, label):
    if label == 'CT':
        data[data < -1000] = -1000
        data[data > 2000] = 2000
        data_max = 2000
        data_min = -1000
    else:
        data_max = np.amax(data)
        data_min = np.amin(data)
    print("Before:", data_max, data_min)
    data_norm = copy.deepcopy(data)
    data_norm -= data_min
    data_norm /= (data_max-data_min)
    print("After:", np.amax(data_norm), np.amin(data_norm))
    return data_norm


# Description: make predictions with well-trained model for Nifti file
def predict_hand(model, test_path, tag=''):
    times = '10_1'

    path_X = glob.glob(test_path + 'Align_MR*.nii')[0]
    print(path_X)
    file_X = nib.load(path_X)
    data_X = file_X.get_fdata()

    path_Y = glob.glob(test_path + 'CT*.nii')[0]
    print(path_Y)
    file_Y = nib.load(path_Y)
    data_Y = file_Y.get_fdata()

    data_Y[data_Y < -1000] = -1000
    data_Y[data_Y > 2000] = 2000

    # MaxMin-norm
    data_input = MaxMinNorm(data_X, label='Align')

    print("data shape", data_input.shape)

    n_pixel = data_X.shape[0]
    n_slice = data_X.shape[2]

    slice_x = 1
    dir_syn = test_path
    if not os.path.exists(dir_syn):
        os.makedirs(dir_syn)

    X = np.zeros((1, n_pixel, n_pixel, slice_x))
    y_hat = np.zeros((n_pixel, n_pixel, n_slice))
    if slice_x == 1:
        for idx in range(n_slice):
            X[0, :, :, 0] = data_input[:, :, idx]
            y_hat[:, :, idx] = np.squeeze(model.predict(X))[:, :, 0]

    if slice_x == 3:
        for idx in range(n_slice):
            idx_0 = idx-1 if idx > 0 else 0
            idx_1 = idx
            idx_2 = idx+1 if idx < n_slice-1 else n_slice - 1
            X[0, :, :, 0] = data_input[:, :, idx_0]
            X[0, :, :, 1] = data_input[:, :, idx_1]
            X[0, :, :, 2] = data_input[:, :, idx_2]
            y_hat[:, :, idx] = np.squeeze(model.predict(X))

    print("Output:")
    print("Mean:", np.mean(y_hat))
    print("STD:", np.std(y_hat))
    print("Max:", np.amax(y_hat))
    print("Min:", np.amin(y_hat))

    # recover HU value to normal size
    y_hat_norm = copy.deepcopy(y_hat)

    Y_max = 2000
    Y_min = -1000

    y_hat_norm *= (Y_max-Y_min)
    y_hat_norm += Y_min
    dif = y_hat_norm - data_Y

    # save nifty file
    affine = file_Y.affine
    header = file_Y.header
    nii_file = nib.Nifti1Image(y_hat_norm, affine, header)
    nib.save(nii_file, dir_syn +'sCT_' + times + '.nii')
    print(dir_syn +'sCT_' + times + '.nii')

    # save difference
    dif_file = nib.Nifti1Image(dif, affine, header)
    nib.save(dif_file, dir_syn +'dif_' + str(times) + '.nii')


if __name__ == '__main__':
    test_path = '/code/data/per/gp_1/10-cases/test/Case11/'
    model_path = '/code/data/per/gp_1/10-cases/1st_round/gen/model/loss_model_chansey-2020-07-16-01-07.hdf5'

    model = load_model(model_path, compile=False)
    predict_hand(model, test_path)


