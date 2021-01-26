#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import copy


def MaxMinNorm(data, label):
    if label == 'CT':
        data[data < -1000] = -1000
        data[data > 2000] = 2000
    data_max = np.amax(data)
    data_min = np.amin(data)
    print("Before:", data_max, data_min)
    data_norm = copy.deepcopy(data)
    data_norm -= data_min
    data_norm /= (data_max-data_min)
    print("After:", np.amax(data_norm), np.amin(data_norm))
    return data_norm