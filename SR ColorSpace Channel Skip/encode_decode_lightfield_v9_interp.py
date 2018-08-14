#
# Push a light field through decoder/encoder modules of the autoencoder
#

from queue import Queue
import code
import numpy as np

import scipy
import scipy.signal

# timing and multithreading
import _thread
import time
from timeit import default_timer as timer

# light field GPU tools
import lf_tools

from skimage.color import rgb2lab
from libs.convert_colorspace import rgb2YCbCr, rgb2YUV


# data config
import config_data_format as cdf
import interpolate_mask as im
import config_autoencoder as hp

def add_result_to_cv( data, result, cv_interp, cv_raw, mask_sum, bs_x, bs_y, bxx):

  """ note: numpy arrays are passed by reference ... I think
  """
  H_mask = hp.eval_res['h_mask']
  W_mask = hp.eval_res['w_mask']
  m = hp.eval_res['m']
  print( 'x', end='', flush=True )
  by = result[1]['py']
  sv = result[0]['cv']

  mask = im.get_mask(H_mask,W_mask,m)
  mask3d = np.expand_dims(mask, axis = 2)
  num_channels = cv_interp.shape[-1]
  mask3d = np.tile(mask3d, (1, 1, num_channels))

  # cv data is in the center of the result stack
  # lazy, hardcoded the current fixed size
  p = H_mask//2 - hp.sy_HR//2
  q = H_mask//2 + hp.sy_HR//2

  H_patch = hp.H_HR
  W_patch = hp.W_HR

  for bx in range(bxx):
    px = bs_x * bx + hp.sx_HR
    py = bs_y * by + hp.sy_HR
    cv_interp[py-p:py+q , px-p:px+q, :] = np.add(cv_interp[py-p:py+q , px-p:px+q, : ],
    np.multiply(sv[bx, H_patch//2 - H_mask//2 : H_patch//2 + H_mask//2 ,H_patch//2 - H_mask//2 : H_patch//2 + H_mask//2, :], mask3d))
    cv_raw[py:py + 32, px:px + 32, :] = sv[bx, 32:64, 32:64, :]

    mask_sum[py-p:py+q , px-p:px+q] = mask_sum[py-p:py+q , px-p:px+q] + mask


def encode_decode_lightfield(data, LF_LR, LF_HR, inputs, outputs, ColorSpace, decoder_path):
  # light field size
  H = LF_LR.shape[2]
  W = LF_LR.shape[3]
  H_HR = LF_HR.shape[2]
  W_HR = LF_HR.shape[3]

  # patch step sizes
  bs_y = hp.sy
  bs_x = hp.sx
  bs_y_HR = hp.sy_HR
  bs_x_HR = hp.sx_HR
  # patch height/width
  ps_y = hp.H
  ps_x = hp.W
  ps_y_HR = hp.H_HR
  ps_x_HR = hp.W_HR
  ps_v = hp.D

  # patches per row/column
  by = np.int16((H - ps_y) / bs_y) + 1
  bx = np.int16((W - ps_x) / bs_x) + 1

  ids = []
  for i in range(0, len(hp.layer_config)):
    ids.append(hp.layer_config[i]['id'])
  pos = ids.index(decoder_path)
  num_channels = hp.layer_config[pos]['end'] -hp.layer_config[pos]['start']

  # one complete row per batch
  cv_interp = np.zeros([H_HR, W_HR, num_channels], np.float32)
  cv_raw = np.zeros([H_HR, W_HR, num_channels], np.float32)
  mask_sum = np.zeros([H_HR,W_HR], dtype = np.float32)

  print('starting LF encoding/decoding [', end='', flush=True)
  start = timer()

  results_received = 0
  for py in range(by):
    print('.', end='', flush=True)

    stacks_h = np.zeros([bx, ps_v, ps_y, ps_x, hp.C], np.float32)
    stacks_v = np.zeros([bx, ps_v, ps_y, ps_x, hp.C], np.float32)
    cv_in = np.zeros([bx, ps_y_HR, ps_x_HR, hp.C], np.float32)

    for px in range(bx):
      # get single patch
      patch = cdf.get_patch(LF_LR, py, px)
      stacks_v[px, :, :, :, :] = patch['stack_v']
      stacks_h[px, :, :, :, :] = patch['stack_h']
      cv_in[px, :, :, :] = patch['cv']
      if ColorSpace == 'YUV':
        stacks_v[px, :, :, :, :] = rgb2YUV(stacks_v[px, :, :, :, :])
        stacks_h[px, :, :, :, :] = rgb2YUV(stacks_h[px, :, :, :, :])
        cv_in[px, :, :, :] = rgb2YUV(cv_in[px, :, :, :])
      elif ColorSpace == 'YCBCR':
        stacks_v[px, :, :, :, :] = rgb2YCbCr(stacks_v[px, :, :, :, :])
        stacks_h[px, :, :, :, :] = rgb2YCbCr(stacks_h[px, :, :, :, :])
        cv_in[px, :, :, :] = rgb2YCbCr(cv_in[px, :, :, :])
      elif ColorSpace == 'LAB':
        stacks_v[px, :, :, :, :] = rgb2lab(stacks_v[px, :, :, :, :])
        stacks_h[px, :, :, :, :] = rgb2lab(stacks_h[px, :, :, :, :])
        cv_in[px, :, :, :] = rgb2lab(cv_in[px, :, :, :])


    # push complete batch to encoder/decoder pipeline
    batch = dict()
    batch['stacks_h'] = stacks_h
    batch['stacks_v'] = stacks_v
    batch['cv'] = cv_in
    batch['py'] = py
    batch['decoder_path'] = decoder_path

    inputs.put(batch)

    #
    if not outputs.empty():
      result = outputs.get()
      add_result_to_cv(data, result, cv_interp, cv_raw, mask_sum, bs_x_HR, bs_y_HR, bx)
      results_received += 1
      outputs.task_done()

  # catch remaining results
  while results_received < by:
    result = outputs.get()
    add_result_to_cv(data, result, cv_interp, cv_raw, mask_sum, bs_x_HR, bs_y_HR, bx)
    results_received += 1
    outputs.task_done()

  # elapsed time since start of dmap computation
  end = timer()
  total_time = end - start
  print('] done, total time %g seconds.' % total_time)

  # evaluate result
  mse = 0.0

  # compute stats and return result
  print('total time ', end - start)
  # print('MSE          : ', mse)

  # code.interact( local=locals() )
  return (cv_interp, total_time, mse, mask_sum, cv_raw)

def scale_back(im, mask):
  H = mask.shape[0]
  W = mask.shape[1]
  mask[mask == 0] = 1
  num_channels = im.shape[-1]

  if len(im.shape) == 3:
    mask = np.expand_dims(mask, axis=2)
    mask = np.tile(mask, (1, 1, num_channels))

  if len(im.shape) == 5:
    mask = np.expand_dims(mask, axis=2)
    mask = np.tile(mask, (1, 1, num_channels))

    mask = np.expand_dims(mask, axis = 3)
    mask = np.transpose(np.tile(mask, (1, 1, 1, 9)), [3, 0, 1, 2])
    mask1 = np.zeros((2,9,H,W,3), dtype = np.float32)
    mask1[0,:,:,:,:] = mask
    mask1[1, :, :, :, :] = mask
    del mask
    mask = mask1

  return(np.divide(im,mask))