#!/usr/bin/python3
#
# read a bunch of source light fields and write out
# training data for our autoencoder in useful chunks
#
# pre-preparation is necessary as the training data
# will be fed to the trainer in random order, and keeping
# several light fields in memory is impractical.
#
# WARNING: store data on an SSD drive, otherwise randomly
# assembing a bunch of patches for training will
# take ages.
#
# (c) Bastian Goldluecke, Uni Konstanz
# bastian.goldluecke@uni.kn
# License: Creative Commons CC BY-SA 4.0
#

from queue import Queue
import time
import code
import os
import sys
import h5py

import numpy as np
from skimage.color import rgb2hsv, hsv2rgb
from scipy.ndimage import gaussian_filter
# python tools for our lf database
import file_io
# additional light field tools
import lf_tools
import matplotlib.pyplot as plt
# OUTPUT CONFIGURATION

# patch size. patches of this size will be extracted and stored
# must remain fixed, hard-coded in NN
scale = 2

nviews = 9
channel = 3

# output file to write to
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!! careful: overwrite mode !!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
# previous training data will be erased.

test_data_dir = "H:\\testData\\super_resolution\\benchmark\\not seen\\"
test_data_filename = 'lf_test_benchmark_'
#
# data_folders = ( ( "training", "boxes" ), )
# data_folders = data_folders_base + data_folders_add
data_source = "E:\\MATLAB\\Benchmark_dataset\\data_only_benchmark\\"
# data_folders = os.listdir(data_source)
data_folders = []
# data_folders.append('dishes')
# data_folders.append('greek')
# data_folders.append('tower')
# data_folders.append('antinous')
# data_folders.append('boardgames')
# data_folders.append('boxes')
# data_folders.append('cotton')
# data_folders.append('dino')
# data_folders.append('kitchen')
# data_folders.append('medieval2')
# data_folders.append('museum')
# data_folders.append('pens')
# data_folders.append('pillows')
# data_folders.append('platonic')
# data_folders.append('rosemary')
data_folders.append('sideboard')
# data_folders.append('table')
# data_folders.append('tomb')
# data_folders.append('town')
# data_folders.append('vinyl')
# data_folders.append('herbs')
# data_folders.append('bicycle')
# data_folders.append('bedroom')
# data_folders.append('origami')

#
# loop over all datasets, write out each dataset in patches
# to feed to autoencoder in random order
#
index = 0
for lf_name in data_folders:
    file = h5py.File(test_data_dir + test_data_filename + lf_name + '.hdf5', 'w')
    data_folder = os.path.join(data_source, lf_name)
    # read diffuse color
    LF = file_io.read_lightfield(data_folder)
    LF = LF.astype(np.float32)

    LF_LR = np.zeros((LF.shape[0], LF.shape[1], int(LF.shape[2] / scale),
                      int(LF.shape[3] / scale), int(LF.shape[4])), np.float32)

    for v in range(0, nviews):
        for h in range(0, nviews):
            LF[v, h, :, :, :] = gaussian_filter(LF[v, h, :, :, :], sigma = 0.5, truncate=2)
            LF_LR[v, h, :, :, :] = LF[v, h, 0:LF.shape[2]-1:scale, 0:LF.shape[3]-1:scale, :]


    dset_LF_LR = file.create_dataset('LF_LR', data=LF_LR)
    dset_LF = file.create_dataset('LF', data=LF)

    # next dataset
    print(' done.')

