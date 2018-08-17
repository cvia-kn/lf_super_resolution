# define the configuration (hyperparameters) for the residual autoencoder
# for this type of network.


# NETWORK MODEL NAME
network_model = 'SR_RGB_VAE_vanilla'

data_path = '/home/aa/Python_projects/Data_train/super_resolution/'
tf_log_path = '/data/tf_logs/'
# data_path = '/home/mz/PyCharm/Data/'

# data_path = 'H:\\trainData\\'
# CURRENT TRAINING DATASET
training_data = [
    # 'lf_benchmark_HSV.hdf5',
    data_path + 'lf_patch_synthetic_rgb_sr_1.hdf5',
    data_path + 'lf_patch_synthetic_rgb_sr_2.hdf5',
    data_path + 'lf_patch_synthetic_rgb_sr_3.hdf5',
    data_path + 'lf_patch_synthetic_rgb_sr_4.hdf5',
    data_path + 'lf_patch_synthetic_rgb_sr_5.hdf5',
    data_path + 'lf_patch_benchmark_rgb_sr.hdf5',
]

# NETWORK LAYOUT HYPERPARAMETERS

# general config params
config = {
    # flag whether we want to train for RGB (might require more
    # changes in other files, can't remember right now)
    'ColorSpace'                  : 'RGB', #'YUV', 'RGB', 'LAB'
    'VisibleGPU'                  :'0,1,2,3',
    # maximum layer which will be initialized (deprecated)
    'max_layer'            : 100,
    # add interpolated input as patch for skip connection in the upscaling phase
    'interpolate'          : True,
    # this will log every tensor being allocated,
    # very spammy, but useful for debugging
    'log_device_placement' : False,
}

# encoder for 48 x 48 patch, 9 views, RGB
D = 9
H = 48
W = 48
nviews = 9
H_HR = 96
W_HR = 96
cv_pos = int((nviews-1)/2)


# patch stepping
sx =  16
sy = 16
sx_HR = 32
sy_HR = 32

C = 3
C_value = 1
C_color = 2

# Number of features in the layers
L = 8
L0 = 16
L1 = 24
L2 = 32
L3 = 64
L4 = 96
L5 = 128
L6 = 160
L7 = 192


# fraction between encoded patch and decoded patch. e.g.
# the feature maps of decoded patch are 3 times as many
# as the encoded patch, then patch_weight = 3
patch_weight  = 3

# Encoder stack for downwards convolution

layer_config = [
    {
      'id': 'RGB',# 'YUV', 'RGB', 'YCBCR', 'LAB' and any combinations
      'channels' : C,
      'start' : 0,
      'end': 3,
      'layout': [
                # for h and s channels
                { 'conv'   : [ 3, 3, 3, C, L1 ],
                  'stride' : [ 1,1,1, 1, 1 ]
                }],
        # transposed convolution
      'upscale': [
                {'conv': [3, 3, 3, L0, L1],
                 'stride': [1, 1, 1, 1, 1]
                 }],
      'final':  [
                { 'conv'   : [ 1, 1, C*patch_weight, C],
                  'stride' : [ 1, 1, 1, 1 ]
                },]

    },
]

# chain of dense layers to form small bottleneck (can be empty)
layers = dict()

# layers['encoder_3D'] = [
#                 { 'conv'   : [ 3,3,3, L1, L2 ],
#                   'stride' : [ 1,1, 2,2, 1 ]
#                 },
#                  # resolution now 9 x 24 x 24
#                 { 'conv'   : [ 3,3,3, L2, L2 ],
#                   'stride' : [ 1,1, 1,1, 1 ]
#                 },
#                 { 'conv'   : [ 3,3,3, L2, L3 ],
#                   'stride' : [ 1,2, 1,1, 1 ]
#                 },
#                 # resolution now 5 x 24 x 24
#                 { 'conv'   : [ 3,3,3, L3, L3 ],
#                   'stride' : [ 1,1, 1,1, 1 ]
#                 },
#                 { 'conv'   : [ 3,3,3, L3, L4 ],
#                   'stride' : [ 1,1, 2,2, 1 ]
#                 },
#                 # resolution now 5 x 12 x 12
#                 { 'conv'   : [ 3,3,3, L4, L4 ],
#                   'stride' : [ 1,1, 1,1, 1 ]
#                 },
#                 { 'conv'   : [ 3,3,3, L4, L5 ],
#                   'stride' : [ 1,1, 2,2, 1 ]
#                 },
#                 # # resolution now 5 x 6 x 6
#                 { 'conv'   : [ 3,3,3, L5, L5 ],
#                   'stride' : [ 1,1, 1,1, 1 ]
#                 },
#                 { 'conv'   : [ 3,3,3, L5, L6 ],
#                   'stride' : [ 1,2, 1,1, 1 ]
#                 },
#                 # resolution now 3 x 6 x 6
#                 { 'conv'   : [ 3,3,3, L6, L6 ],
#                   'stride' : [ 1,1, 1,1, 1 ]
#                 },
#                 {'conv': [3, 3, 3, L6, L7],
#                  'stride': [1, 1, 2, 2, 1]
#                  },
#                 # resolution now 3 x 3 x 3
#                 {'conv': [3, 3, 3, L7, L7],
#                  'stride': [1, 1, 1, 1, 1]
#                  },
#
#                 ]


layers['encoder_3D'] = [
                { 'conv'   : [ 3,3,3, L1, L2 ],
                  'stride' : [ 1,1, 2,2, 1 ]
                },
                 # resolution now 9 x 24 x 24
                { 'conv'   : [ 3,3,3, L2, L2 ],
                  'stride' : [ 1,1, 2,2, 1 ]
                },
                { 'conv'   : [ 3,3,3, L2, L3 ],
                  'stride' : [ 1,2, 1,1, 1 ]
                },
                # resolution now 5 x 24 x 24
                { 'conv'   : [ 3,3,3, L3, L3 ],
                  'stride' : [ 1,1, 2,2, 1 ]
                },
                { 'conv'   : [ 3,3,3, L3, L4 ],
                  'stride' : [ 1,2, 1,1, 1 ]
                },
                # resolution now 5 x 12 x 12
                { 'conv'   : [ 3,3,3, L4, L4 ],
                  'stride' : [ 1,1, 2,2, 1 ]
                },

                ]

# transposed convolution
layers[ 'upscale'] = [
                {'conv': [3, 3, L, L0],
                 'stride': [1, 2, 2, 1],
                 'bicubic_connection' : False
                 },
                {'conv': [3, 3, L, L],
                 'stride': [1, 1, 1, 1],
                 'bicubic_connection': True
                 },
                {'conv': [3, 3, C, L],
                 'stride': [1, 1, 1, 1],
                 'bicubic_connection' : False
                 },
                # {'conv': [3, 3, C, C],
                #  'stride': [1, 1, 1, 1],
                #  'bicubic_connection': False
                #  },
]

layers[ 'autoencoder_nodes' ] = []
layers[ 'preferred_gpu' ] = 0
layers[ 'merge_encoders' ] = True
layers[ '2D_encoder_nodes' ] = []#[2*3*3*3*L7]
# only if merge encoders is True
layers[ 'vae' ] = True
sigma_vae = 1 #0.05

# 3D ENCODERS
encoders_3D = [
    {
      'id': 'RGB', # 'YUV', 'RGB', 'YCBCR', 'LAB' and any combinations
      'channels': C,
      'preferred_gpu' : 1,
    },
]
#
# 2D DECODERS
#
# Each one generates a 2D upsampling pathway next to the
# two normal autoencoder pipes.
#
# Careful, takes memory. Remove some from training if limited.
#
decoders_2D = [
    {
      'id': 'RGB', # 'YUV', 'RGB', 'YCBCR', 'LAB' and any combinations
      'channels': C,
      'preferred_gpu' : 2,
      'loss_fn':  'L2',
      'train':    True,
      'weight':   0.5,
      'no_relu': False,
      'skip_connection': True,
    },
]

# MINIMIZERS
minimizers = [

  # center view super resolution
  {
    'id': 'RGB_min', # 'YCBCR_min', 'LAB_min', 'RGB_min',
    'losses_2D' : [ 'RGB','KL_divergence' ],  # 'YUV', 'RGB', 'YCBCR', 'LAB' and any combinations
    'optimizer' : 'Adam',
    'preferred_gpu' : 3,
    'step_size' : 1e-4,
  },
 ]


# TRAINING HYPERPARAMETERS
training = dict()

# subsets to split training data into
# by default, only 'training' will be used for training, but the results
# on mini-batches on 'validation' will also be logged to check model performance.
# note, split will be performed based upon a random shuffle with filename hash
# as seed, thus, should be always the same for the same file.
#
training[ 'subsets' ] = {
  'validation'   : 0.05,
  'training'     : 0.95,
}


# number of samples per mini-batch
# reduce if memory is on the low side,
# but if it's too small, training becomes ridicuously slow
training[ 'samples_per_batch' ] = 30

# log interval (every # mini-batches per dataset)
training[ 'log_interval' ] = 5

# save interval (every # iterations over all datasets)
training[ 'save_interval' ] = 100

# noise to be added on each input patch
# (NOT on the decoding result)
training[ 'noise_sigma' ] = 0.0

# decay parameter for batch normalization
# should be larger for larger datasets
training[ 'batch_norm_decay' ]  = 0.9
# flag whether BN center param should be used
training[ 'batch_norm_center' ] = False
# flag whether BN scale param should be used
training[ 'batch_norm_scale' ]  = False
# flag whether BN should be zero debiased (extra param)
training[ 'batch_norm_zero_debias' ]  = False

eval_res = {
    'h_mask': 90,
    'w_mask': 90,
    'm': 10,
    'min_mask': 0.1,
    'result_folder': "./results/",
    'test_data_folder': "H:\\testData\\"
}