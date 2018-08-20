
from queue import Queue
import time
import numpy as np
import h5py
# plotting
import matplotlib.pyplot as plt
from skimage import measure
from skimage.color import lab2rgb, yuv2rgb
from libs.convert_colorspace import rgb2YCbCr, YCbCr2rgb, rgb2YUV
from interp import *
# timing and multithreading
import _thread

# light field GPU tools
import lf_tools
# evaluator thread
from encode_decode_lightfield_v9_interp import encode_decode_lightfield
from encode_decode_lightfield_v9_interp import scale_back
from thread_evaluate_v9 import evaluator_thread
import os

# configuration
import config_autoencoder as hp


# Model path setup
model_id = hp.network_model
model_path = './networks/' + model_id + '/model.ckpt'
result_folder = hp.eval_res['result_folder']
data_eval_folder = hp.eval_res['test_data_folder']
os.makedirs(result_folder, exist_ok=True )

# I/O queues for multithreading
inputs = Queue( 15*15 )
outputs = Queue( 15*15 )

data_folders = (

# ( "super_resolution", "benchmark", "not seen", "lf_test_benchmark_antinous" ),
# ( "super_resolution", "benchmark", "not seen", "lf_test_benchmark_pillows" ),
( "super_resolution", "benchmark", "not seen", "lf_test_benchmark_herbs" ),
( "super_resolution", "stanford", "not seen", "lf_test_stanford_Bunny" ),
# ( "super_resolution", "stanford", "not seen", "lf_test_stanford_Eucalyptus" ),
( "super_resolution", "stanford", "not seen", "lf_test_stanford_JellyBeans" ),
# ( "super_resolution", "stanford", "not seen", "lf_test_stanford_LegoBulldozer" ),
# ( "super_resolution", "stanford", "not seen", "lf_test_stanford_LegoTruck" ),
# ( "super_resolution", "stanford", "not seen", "lf_test_stanford_TreasureChest" ),
( "super_resolution", "HCI", "not seen", "lf_test_HCI_buddha" ),
( "super_resolution", "HCI", "not seen", "lf_test_HCI_buddha2" ),
# ( "super_resolution", "HCI", "not seen", "lf_test_HCI_horses" ),
# ( "super_resolution", "HCI", "not seen", "lf_test_HCI_medieval" ),
( "super_resolution", "HCI", "not seen", "lf_test_HCI_monasRoom" ),
( "super_resolution", "HCI", "not seen", "lf_test_HCI_papillon" ),
# ( "super_resolution", "HCI", "not seen", "lf_test_HCI_stillLife" ),
)

# YCBCR
# evaluator thread
_thread.start_new_thread( evaluator_thread,
                          ( model_path, hp, inputs,  outputs ))

# wait a bit to not skew timing results with initialization
time.sleep(20)

# loop over all datasets and collect errors
results = []


for lf_name in data_folders:
    file = h5py.File(result_folder + lf_name[3] + '.hdf5', 'w')
    data_file = data_eval_folder + lf_name[0] + "/" + lf_name[1] + "/" + lf_name[2] + "/" + \
                    lf_name[3] + ".hdf5"
    hdf_file = h5py.File(data_file, 'r')
    # hard-coded size, just for testing
    LF_LR = hdf_file['LF_LR']
    LF_HR = hdf_file['LF']
    cv_gt = lf_tools.cv(LF_HR)
    cv_LR = lf_tools.cv(LF_LR)

    data = []

    color_space = hp.config['ColorSpace']


    if color_space == 'YUV':

        cv_gt_orig = rgb2YUV(cv_gt)
        cv_LR_orig = rgb2YUV(cv_LR)

        decoder_path = 'Y'

        result_cv = encode_decode_lightfield(data, LF_LR, LF_HR,
                                             inputs, outputs, color_space,
                                             decoder_path=decoder_path,
                                             )
        cv_out1 = result_cv[0]
        mask = result_cv[3]
        cv_out1 = scale_back(cv_out1, mask)

        decoder_path = 'UV'

        result_cv = encode_decode_lightfield(data, LF_LR, LF_HR,
                                             inputs, outputs, color_space,
                                             decoder_path=decoder_path,
                                             )
        cv_out2 = result_cv[0]
        mask = result_cv[3]
        cv_out2 = scale_back(cv_out2, mask)

        cv_out_orig = np.concatenate((cv_out1, cv_out2), -1)
        cv_out_rgb = yuv2rgb(cv_out_orig)
        cv_out_rgb = cv_out_rgb.astype(np.float32)

    elif color_space == 'YCBCR':
        cv_gt_orig = rgb2YCbCr(cv_gt)
        cv_LR_orig = rgb2YCbCr(cv_LR)

        decoder_path = 'Y'

        result_cv = encode_decode_lightfield(data, LF_LR, LF_HR,
                                             inputs, outputs, color_space,
                                             decoder_path=decoder_path,
                                             )
        cv_out1 = result_cv[0].astype(np.float64)
        mask = result_cv[3].astype(np.float64)
        cv_out1 = scale_back(cv_out1, mask).astype(np.float64)

        decoder_path = 'CBCR'

        result_cv = encode_decode_lightfield(data, LF_LR, LF_HR,
                                             inputs, outputs, color_space,
                                             decoder_path=decoder_path,
                                             )
        cv_out2 = result_cv[0]
        mask = result_cv[3]
        cv_out2 = scale_back(cv_out2, mask)

        cv_out_orig = np.concatenate((cv_out1, cv_out2), -1)
        cv_out_rgb = YCbCr2rgb(cv_out_orig)
        cv_out_rgb = cv_out_rgb.astype(np.float32)
    else:
        decoder_path = 'RGB'

        result_cv = encode_decode_lightfield(data, LF_LR, LF_HR,
                                             inputs, outputs, color_space,
                                             decoder_path=decoder_path,
                                             )
        cv_out2 = result_cv[0]
        mask = result_cv[3]
        cv_out2 = scale_back(cv_out2, mask)


    if color_space == 'YUV' or color_space == 'YCBCR':
        sh = cv_gt.shape
        sx = hp.sx
        sy = hp.sy
        cv_gt = cv_gt[sx:sh[0]+1-sx,sy:sh[1]+1-sy,:]
        cv_out_rgb = cv_out_rgb[sx:sh[0]+1-sx,sy:sh[1]+1-sy,:]
        cv_gt_orig = cv_gt_orig[sx:sh[0] + 1 - sx, sy:sh[1] + 1 - sy, :]
        cv_out_orig = cv_out_orig[sx:sh[0] + 1 - sx, sy:sh[1] + 1 - sy, :]

        PSNR_out_rgb = measure.compare_psnr(cv_gt, cv_out_rgb, data_range=1, dynamic_range=None)
        SSIM_out_rgb = measure.compare_ssim(cv_gt, cv_out_rgb, data_range=1, multichannel=True)

        plt.subplot(1, 3, 1)
        plt.title("PSNR= %.2f \nSSIM= %.2f" % (PSNR_out_rgb, SSIM_out_rgb))
        plt.xlabel("cv_interp")
        plt.imshow(np.clip(cv_out_rgb, 0, 1))
        plt.subplot(1, 3, 2)
        plt.imshow(np.clip(cv_gt, 0, 1))
        plt.xlabel("Ground Truth")
        plt.subplot(1, 3, 3)
        plt.imshow(np.clip(cv_LR, 0, 1))
        plt.xlabel("LowRes image")

        plt.show()

        PSNR_out_orig = measure.compare_psnr(cv_gt_orig, cv_out_orig, data_range=1, dynamic_range=None)
        SSIM_out_orig = measure.compare_ssim(cv_gt_orig, cv_out_orig, data_range=1, multichannel=True)

        plt.subplot(1, 3, 1)
        plt.title("PSNR= %.2f \nSSIM= %.2f" % (PSNR_out_orig, SSIM_out_orig))
        plt.xlabel("cv_interp")
        plt.imshow(np.clip(cv_out_orig, 0, 1), cmap='gray')
        plt.subplot(1, 3, 2)
        plt.imshow(np.clip(cv_gt_orig, 0, 1), cmap='gray')
        plt.xlabel("Ground Truth")
        plt.subplot(1, 3, 3)
        plt.imshow(np.clip(cv_LR_orig, 0, 1), cmap='gray')
        plt.xlabel("LowRes image")

        plt.show()
    else:
        sh = cv_gt.shape
        sx = hp.sx
        sy = hp.sy
        cv_gt = cv_gt[sx:sh[0] + 1 - sx, sy:sh[1] + 1 - sy, :]
        cv_out2 = cv_out2[sx:sh[0] + 1 - sx, sy:sh[1] + 1 - sy, :]


        PSNR_out_rgb = measure.compare_psnr(cv_gt, cv_out2, data_range=1, dynamic_range=None)
        SSIM_out_rgb = measure.compare_ssim(cv_gt, cv_out2, data_range=1, multichannel=True)

        plt.subplot(1, 3, 1)
        plt.title("PSNR= %.2f \nSSIM= %.2f" % (PSNR_out_rgb, SSIM_out_rgb))
        plt.xlabel("cv_interp")
        plt.imshow(np.clip(cv_out2, 0, 1))
        plt.subplot(1, 3, 2)
        plt.imshow(np.clip(cv_gt, 0, 1))
        plt.xlabel("Ground Truth")
        plt.subplot(1, 3, 3)
        plt.imshow(np.clip(cv_LR, 0, 1))
        plt.xlabel("LowRes image")

        plt.show()

inputs.put( () )

