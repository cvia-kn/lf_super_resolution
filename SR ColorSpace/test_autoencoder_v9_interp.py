
from queue import Queue
import time
import numpy as np
import h5py
import math
# plotting
import matplotlib.pyplot as plt
from skimage import measure
from interp import *
# timing and multithreading
import _thread

# light field GPU tools
import lf_tools
import tensorflow as tf2
# evaluator thread
from encode_decode_lightfield_v9_interp import encode_decode_lightfield
from encode_decode_lightfield_v9_interp import scale_back
from thread_evaluate_v9 import evaluator_thread

# configuration
import config_autoencoder_v9_final as hp


# Model path setup
model_id = hp.network_model
model_path = './networks/' + model_id + '/model.ckpt'
result_folder = hp.eval_res['result_folder']
data_eval_folder = hp.eval_res['test_data_folder']

# I/O queues for multithreading
inputs = Queue( 15*15 )
outputs = Queue( 15*15 )

data_folders = (

# ( "diffuse_specular", "intrinsic", "seen", "lf_test_intrinsicg3CxzfVmydmYGr" ),
# ( "diffuse_specular", "intrinsic", "seen", "lf_test_intrinsickqK4J4cafLswf2" ),
# ( "diffuse_specular",  "intrinsic", "not seen", "lf_test_intrinsicz1DefSIynpJhqi" ),
# ( "diffuse_specular",  "intrinsic", "not seen", "lf_test_intrinsiccTRQYxjW6XXw5J" ),
# ( "diffuse_specular",  "intrinsic", "not seen", "lf_test_intrinsicGumNhefYrATJLh" ),
# ( "diffuse_specular",  "intrinsic", "not seen", "lf_test_intrinsictrMltdlvXzRdOS" ),
# ( "diffuse_specular",  "intrinsic", "not seen", "lf_test_intrinsiciq16JtRgF7yzKp" ),
#
# ( "diffuse_specular",  "lytro", "seen", "lf_test_lytro_flowers_lf.mat" ),
# ( "diffuse_specular",  "lytro", "seen", "lf_test_lytro_koala1_lf.mat" ),
# ( "diffuse_specular",  "lytro", "seen", "lf_test_lytro_cat3_lf.mat" ),
# ( "diffuse_specular",  "lytro", "seen", "lf_test_lytro_hedgehog1_lf.mat" ),
# ( "diffuse_specular",  "lytro", "not seen", "lf_test_lytro_koala_lf.mat" ),
# ( "diffuse_specular",  "lytro", "not seen", "lf_test_lytro_buddha1.mat" ),
# ( "diffuse_specular",  "lytro", "not seen", "lf_test_lytro_IMG_2693_eslf.png.mat" ),
#
# ( "diffuse_specular",  "hci", "seen", "lf_test_hci_maria_lf.mat" ),
# ( "diffuse_specular",  "hci", "not seen", "lf_test_hci_cube_lf.mat" ),

( "diffuse_specular",  "benchmark", "not seen", "lf_benchmark_bicycle256" ),
( "diffuse_specular",  "benchmark", "seen", "lf_benchmark_greek256" ),
# ( "diffuse_specular",  "benchmark", "seen", "lf_benchmark_cotton" ),
( "diffuse_specular",  "benchmark", "not seen", "lf_benchmark_herbs256" ),
#
# ( "diffuse_specular", "stanford", "not seen", "lf_test_stanford_Amethyst_lf.mat" ),
# ( "diffuse_specular",  "stanford", "seen", "lf_test_stanford_LegoTruck_lf.mat" ),
#
#
# # numericat eval
# ( "diffuse_specular",  "intrinsic", "not seen", "lf_test_intrinsic0AruXjjpWdmTOz" ),
# ( "diffuse_specular",  "intrinsic", "not seen", "lf_test_intrinsic6CgoBrTon07emN" ),
# ( "diffuse_specular",  "intrinsic", "not seen", "lf_test_intrinsicBrZmxtWCIkYTFU" ),
# ( "diffuse_specular",  "intrinsic", "not seen", "lf_test_intrinsicDcO5nAshBnldAx" ),
# ( "diffuse_specular",  "intrinsic", "not seen", "lf_test_intrinsicF9oJj8EUagULX3" ),
# ( "diffuse_specular",  "intrinsic", "not seen", "lf_test_intrinsicFRMYJ3bYIKVICq" ),
)


# evaluator thread
_thread.start_new_thread( evaluator_thread,
                          ( model_path, hp, inputs,  outputs ))

# wait a bit to not skew timing results with initialization
time.sleep(20)

# loop over all datasets and collect errors
results = []


for lf_name in data_folders:
    file = h5py.File(result_folder + lf_name[3] + '.hdf5', 'w')
    # if lf_name[1] == 'intrinsic':
    # # stored directly in hdf5
    #     data_file = data_eval_folder + lf_name[0] + "/" + lf_name[1] + "/" + lf_name[2] + "/" + \
    #             lf_name[3] + ".hdf5"
    #     hdf_file = h5py.File( data_file, 'r')
    #     # hard-coded size, just for testing
    #     LF = hdf_file[ 'LF' ]
    #     cv_gt = lf_tools.cv( LF )
    #
    #     LF_diffuse_gt = hdf_file[ 'LF_diffuse' ]
    #     diffuse_gt = lf_tools.cv( LF_diffuse_gt )
    #
    #     LF_specular_gt = hdf_file[ 'LF_specular' ]
    #     specular_gt = lf_tools.cv( LF_specular_gt )
    #
    #     disp_gt = hdf_file[ 'LF_disp' ]
    #
    #     dmin = np.min(disp_gt)
    #     dmax = np.max(disp_gt)
    # elif lf_name[1] == 'benchmark':
    data_file = data_eval_folder + lf_name[0] + "/" + lf_name[1] + "/" + lf_name[2] + "/" + \
                    lf_name[3] + ".hdf5"
    hdf_file = h5py.File(data_file, 'r')
    # hard-coded size, just for testing
    LF_LR = hdf_file['LF_LR']
    LF_HR = hdf_file['LF_HR']
    cv_gt = lf_tools.cv(LF_HR)
    cv_LR = lf_tools.cv(LF_LR)

    if lf_name[2] == "seen":
        disp_gt = hdf_file['LF_disp']

        dmin = np.min( disp_gt )
        dmax = np.max( disp_gt )
    else:
        dmin = -3.5
        dmax = 3.5
        disp_gt = np.zeros((cv_gt.shape[0], cv_gt.shape[1]), dtype=np.float32)
        diffuse_gt = np.zeros((cv_gt.shape[0],cv_gt.shape[1],3), dtype = np.float32)
        specular_gt = np.zeros((cv_gt.shape[0], cv_gt.shape[1], 3), dtype=np.float32)
    # else:
    #     data_file = data_eval_folder + lf_name[0] + "/" + lf_name[1] + "/" + lf_name[2] + "/" + \
    #                 lf_name[3] + ".hdf5"
    #     hdf_file = h5py.File(data_file, 'r')
    #     # hard-coded size, just for testing
    #     LF = hdf_file['LF']
    #     cv_gt = lf_tools.cv(LF)
    #     disp_gt = np.zeros((cv_gt.shape[0],cv_gt.shape[1]), dtype = np.float32)
    #     diffuse_gt = np.zeros((cv_gt.shape[0],cv_gt.shape[1],3), dtype = np.float32)
    #     specular_gt = np.zeros((cv_gt.shape[0], cv_gt.shape[1], 3), dtype=np.float32)
    #     dmin = -3.5
    #     dmax = 3.5

    data = []
    result_cv = encode_decode_lightfield(data, LF_LR, LF_HR,
                                        inputs, outputs,
                                        decoder_path='cv',
                                        )

    cv_bicubic = np.empty((512, 512, 3), dtype=np.float32)

    for r in range(cv_bicubic.shape[0]):
        for c in range(cv_bicubic.shape[1]):
            rr = (r + 1) / cv_bicubic.shape[0] * cv_LR.shape[0] - 1
            cc = (c + 1) / cv_bicubic.shape[1] * cv_LR.shape[1] - 1

            rr_int = int(rr)
            cc_int = int(cc)

            sum_p = np.empty(cv_LR.shape[2])
            for j in range(rr_int - 1, rr_int + 3):
                for i in range(cc_int - 1, cc_int + 3):
                    w = get_w(rr - j) * get_w(cc - i)
                    p = get_item(cv_LR, j, i) * w
                    sum_p += p

            for i, entry in enumerate(sum_p):
                sum_p[i] = min(max(entry, 0), 255)

            cv_bicubic[r][c] = sum_p

    cv_out = result_cv[0]
    mask = result_cv[3]
    cv_raw = result_cv[4]

    cv_out = scale_back(cv_out, mask)
    cv_bicubic = cv_bicubic[32:-32, 32:-32, :]
    cv_out = cv_out[32:-32, 32:-32, :]
    # cv_raw = cv_raw[32:-32, 32:-32, :]
    cv_gt = cv_gt[32:-32, 32:-32, :]

    PSNR_bicubic = measure.compare_psnr(cv_gt, cv_bicubic, data_range=1, dynamic_range=None)
    SSIM_bicubic = measure.compare_ssim(cv_gt, cv_bicubic, data_range=1, multichannel=True)
    PSNR_out = measure.compare_psnr(cv_gt, cv_out, data_range=1, dynamic_range=None)
    SSIM_out = measure.compare_ssim(cv_gt, cv_out, data_range=1, multichannel=True)
    # PSNR_raw = measure.compare_psnr(cv_gt, cv_raw, data_range=1, dynamic_range=None)
    # SSIM_raw = measure.compare_ssim(cv_gt, cv_raw, data_range=1, multichannel=True)

    plt.subplot(1, 3, 1)
    plt.title("PSNR= %.2f \nSSIM= %.2f" % (PSNR_bicubic, SSIM_bicubic))
    plt.xlabel("bicubic")
    plt.imshow(np.clip(cv_bicubic, 0, 1))
    plt.subplot(1, 3, 2)
    plt.title("PSNR= %.2f \nSSIM= %.2f" % (PSNR_out, SSIM_out))
    plt.xlabel("cv_interp")
    plt.imshow(np.clip(cv_out, 0, 1))
    # plt.subplot(1, 4, 3)
    # plt.title("PSNR= %.2f \nSSIM= %.2f" % (PSNR_raw, SSIM_raw))
    # plt.xlabel("cv_raw")
    # plt.imshow(np.clip(cv_raw, 0, 1))
    plt.subplot(1, 3, 3)
    plt.imshow(np.clip(cv_gt, 0, 1))
    plt.xlabel("Ground Truth")
    plt.show()






inputs.put( () )

