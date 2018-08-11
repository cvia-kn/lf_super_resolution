# define the configuration (hyperparameters) for the data format
# this is hard-coded now in all the training data, should not be changed anymore


import lf_tools
import numpy as np

# general config params
data_config = {
    # patch size
    'D' : 9,
    'H' : 48,
    'W' : 48,
    'H_HR' : 96,
    'W_HR' : 96,
    # patch stepping
    'SX' : 16,
    'SY' : 16,
    'SX_HR' : 32,
    'SY_HR' : 32,
    # depth range and number of labels
    'dmin' : -3.5,
    'dmax' :  3.5,
}



# get patch at specified block coordinates
def get_patch( LF_LR, by, bx ):

  patch = dict()

  # compute actual coordinates
  y = by * data_config['SY']
  x = bx * data_config['SX']
  py = data_config['H']
  px = data_config['W']
  
  # extract data
  # (stack_h, stack_v) = lf_tools.epi_stacks( LF, y, x, py, px )
  (stack_v, stack_h) = lf_tools.epi_stacks( LF_LR, y, x, py, px )
  # make sure the direction of the view shift is the first spatial dimension
  stack_h = np.transpose( stack_h, (0, 2, 1, 3) )
  patch[ 'stack_v' ] = stack_v
  patch[ 'stack_h' ] = stack_h
  patch['cv'] = np.zeros((data_config['H_HR'],data_config['W_HR'],3) )


  return patch
