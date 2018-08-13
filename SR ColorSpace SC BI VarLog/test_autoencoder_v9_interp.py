
from queue import Queue
import time
import numpy as np
import h5py
# plotting
import matplotlib.pyplot as plt
from skimage import measure
from skimage.color import lab2rgb, yuv2rgb
from libs.convert_colorspace import rgb2YCbCr, YCbCr2rgb
from interp import *
# timing and multithreading
import _thread

# light field GPU tools
import lf_tools
# evaluator thread
from encode_decode_lightfield_v9_interp import encode_decode_lightfield
from encode_decode_lightfield_v9_interp import scale_back
from thread_evaluate_v9 import evaluator_thread

# configuration
import config_autoencoder as hp


# Model path setup
model_id = hp.network_model
model_path = './networks/' + model_id + '/model.ckpt'
result_folder = hp.eval_res['result_folder']
data_eval_folder = hp.eval_res['test_data_folder']

# I/O queues for multithreading
inputs = Queue( 15*15 )
outputs = Queue( 15*15 )

data_folders = (

# ( "super_resolution", "benchmark", "not seen", "lf_test_benchmark_antinous" ),
( "super_resolution", "benchmark", "not seen", "lf_test_benchmark_pillows" ),
# ( "super_resolution", "benchmark", "not seen", "lf_test_benchmark_herbs" ),
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
    decoder_path = 'L'

    result_cv = encode_decode_lightfield(data, LF_LR, LF_HR,
                                        inputs, outputs, color_space,
                                        decoder_path=decoder_path,
                                        )
    cv_out1 = result_cv[0]
    mask = result_cv[3]
    cv_out1 = scale_back(cv_out1, mask)

    decoder_path = 'AB'

    result_cv = encode_decode_lightfield(data, LF_LR, LF_HR,
                                         inputs, outputs, color_space,
                                         decoder_path=decoder_path,
                                         )
    cv_out2 = result_cv[0]
    mask = result_cv[3]
    cv_out2 = scale_back(cv_out2, mask)

    cv_out = np.concatenate((cv_out1,cv_out2),-1)

    if color_space == 'YUV':
        cv_out = yuv2rgb(cv_out.astype(np.float64))
        cv_out = cv_out.astype(np.float32)
    elif color_space == 'YCBCR':
        cv_out = YCbCr2rgb(cv_out)
        cv_out = cv_out.astype(np.float32)
    elif color_space == 'LAB':
        cv_out = lab2rgb(cv_out.astype(np.float64))
        cv_out = cv_out.astype(np.float32)


    PSNR_out = measure.compare_psnr(cv_gt, cv_out, data_range=1, dynamic_range=None)
    SSIM_out = measure.compare_ssim(cv_gt, cv_out, data_range=1, multichannel=True)

    plt.subplot(1, 3, 1)
    plt.title("PSNR= %.2f \nSSIM= %.2f" % (PSNR_out, SSIM_out))
    plt.xlabel("cv_interp")
    plt.imshow(np.clip(cv_out, 0, 1))
    plt.subplot(1, 3, 2)
    plt.imshow(np.clip(cv_gt, 0, 1))
    plt.xlabel("Ground Truth")
    plt.subplot(1, 3, 3)
    plt.imshow(np.clip(cv_LR, 0, 1))
    plt.xlabel("LowRes image")

    plt.show()

inputs.put( () )

