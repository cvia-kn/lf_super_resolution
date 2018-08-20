# define the configuration (hyperparameters) for the data format
# this is hard-coded now in all the training data, should not be changed anymore


import lf_tools
import numpy as np
import config_autoencoder as hp

# general config params
# data_config = {
#     # patch size
#     'D' : 9,
#     'H' : 48,
#     'W' : 48,
#     'H_HR' : 96,
#     'W_HR' : 96,
#     # patch stepping
#     'SX' : 16,
#     'SY' : 16,
#     'SX_HR' : 32,
#     'SY_HR' : 32,
#     # depth range and number of labels
#     'dmin' : -3.5,
#     'dmax' :  3.5,
# }



# get patch at specified block coordinates
def get_patch( LF_LR, by, bx ):

  patch = dict()

  # compute actual coordinates
  y = by * hp.sy
  x = bx * hp.sx
  py = hp.H
  px = hp.W
  
  # extract data
  # (stack_h, stack_v) = lf_tools.epi_stacks( LF, y, x, py, px )
  LF_LR = LF_LR.value
  stack_h,stack_v,stack_l,stack_r = lf_tools.epi_stacks_2( LF_LR, y, x, py, px )
  # make sure the direction of the view shift is the first spatial dimension
  stack_h = np.transpose( stack_h, (0, 2, 1, 3) )
  stack_r = np.transpose(stack_r, (0, 2, 1, 3))
  patch[ 'stack_v' ] = np.concatenate([stack_v, stack_l], axis=3)
  patch[ 'stack_h' ] = np.concatenate([stack_h, stack_r], axis=3)
  patch['cv'] = np.zeros((hp.H_HR,hp.W_HR,int(hp.C/2)) )


  return patch
