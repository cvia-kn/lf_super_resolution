import numpy as np
import tensorflow as tf
from libs.activations import lrelu
import h5py
import matplotlib.pyplot as plt

# path = 'lf_benchmark_SR.hdf5'
# data_path = '/home/aa/Python_projects/Data_train/super_resolution/'
data_path = 'H:\\trainData\\'
training_data = [
    # 'lf_benchmark_HSV.hdf5',
    data_path + 'lf_patch_synthetic_sr_1.hdf5',
    # data_path + 'lf_patch_synthetic_sr_2.hdf5',
    # data_path + 'lf_patch_synthetic_sr_3.hdf5',
    # data_path + 'lf_patch_synthetic_sr_4.hdf5',
    # data_path + 'lf_patch_synthetic_sr_5.hdf5',
    # data_path + 'lf_patch_synthetic_sr_6.hdf5',
    # data_path + 'lf_patch_synthetic_sr_7.hdf5',
]
path = training_data[0]

f = h5py.File(path, 'r')
a = f['cv'].value
b = f['stacks_v'].value
c = f['stacks_h'].value
k = 2
# b = f['stacks_h'][:, :, :, :, :]
#
#
#
# print(b.shape)
# print(b2.shape)
#
#
# for i in range(9):
#     plt.imshow(b[i,:,:,:,162])
#     plt.show()
#     plt.imshow(b2[i, :, :, :, 162])
#     plt.show()


# plt.imshow(b[4,:,:,:,99])
# plt.axis('off')
# plt.show()
# plt.imshow(b2[4,:,:,:,99])
# plt.axis('off')
# plt.show()


