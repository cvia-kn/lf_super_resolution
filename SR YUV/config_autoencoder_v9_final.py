# define the configuration (hyperparameters) for the residual autoencoder
# for this type of network.


# NETWORK MODEL NAME
network_model = 'SR_YUV_v2'

data_path = '/home/aa/Python_projects/Data_train/super_resolution/'
# data_path = 'H:\\trainData\\'
# CURRENT TRAINING DATASET
training_data = [
    # 'lf_benchmark_HSV.hdf5',
    data_path + 'lf_patch_synthetic_YUV_sr_1.hdf5',
    # data_path + 'lf_patch_synthetic_sr_2.hdf5',
    # data_path + 'lf_patch_synthetic_sr_3.hdf5',
    # data_path + 'lf_patch_synthetic_sr_4.hdf5',
    # data_path + 'lf_patch_synthetic_sr_5.hdf5',
    # data_path + 'lf_patch_synthetic_sr_6.hdf5',
    # data_path + 'lf_patch_synthetic_sr_7.hdf5',
]

# NETWORK LAYOUT HYPERPARAMETERS

# general config params
config = {
    # flag whether we want to train for RGB (might require more
    # changes in other files, can't remember right now)
    'rgb'                  : True,

    # maximum layer which will be initialized (deprecated)
    'max_layer'            : 100,

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

if config['rgb']:
    C = 3
else:
    C = 1

# Number of features in the layers
L = 16
L0 = 24
L1 = 32 #32
L2 = 64 #64
L3 = 96 #96
L4 = 128 #128
L5 = 160 #160
L6 = 192 #192


C_hs = 2
C_v = 1

L_hs = 2*D*C_hs
L_v = 2*D*C_v


# fraction between encoded patch and decoded patch. e.g.
# the feature maps of decoded patch are 3 times as many
# as the encoded patch, then patch_weight = 3
patch_weight  = 3

# Encoder stack for downwards convolution

layer_config = [
    { 'id': 'YUV',
      'channels' : C,
      'start' : 0,
      'end': 3,
      'preferred_gpu' : 0,
      'layout': [
                # for h and s channels
                { 'conv'   : [ 3,3,3, C, L0 ],
                  'stride' : [ 1,1,1, 1, 1 ]
                }],
      'upscale':[
                { 'conv'   : [ 3, 3, 3, L, L0  ], #L_hs*patch_weight+L_hs
                  'stride' : [ 1, 1, 1, 1, 1 ]
                }],
      'final':  [
                { 'conv'   : [ 1, 1, C*3, C ],
                  'stride' : [ 1, 1, 1, 1 ]
                },]

    },
]

# chain of dense layers to form small bottleneck (can be empty)
layers = dict()
# layers['encoder_3D'] = [
#                 { 'conv'   : [ 3,3,3, L0, L0 ],
#                   'stride' : [ 1,1,2, 2, 1 ]
#                 },
#                 { 'conv'   : [ 3,3,3, L0, L1 ],
#                   'stride' : [ 1,1, 2,2, 1 ]
#                 },
#                  # resolution now 9 x 24 x 24
#                 { 'conv'   : [ 3,3,3, L1, L1 ],
#                   'stride' : [ 1,2, 2,2, 1 ]
#                 },
#                 { 'conv'   : [ 3,3,3, L1, L1 ],
#                   'stride' : [ 1,2, 1,1, 1 ]
#                 },
#                 { 'conv'   : [ 3,3,3, L1, L2 ],
#                   'stride' : [ 1,1, 2,2, 1 ]
#                 },
#                ]
layers['encoder_3D'] = [
                { 'conv'   : [ 3,3,3, L0, L1 ],
                  'stride' : [ 1,1, 2,2, 1 ]
                },
                 # resolution now 9 x 24 x 24
                { 'conv'   : [ 3,3,3, L1, L1 ],
                  'stride' : [ 1,1, 1,1, 1 ]
                },
                { 'conv'   : [ 3,3,3, L1, L2 ],
                  'stride' : [ 1,2, 1,1, 1 ]
                },
                # resolution now 5 x 24 x 24
                { 'conv'   : [ 3,3,3, L2, L2 ],
                  'stride' : [ 1,1, 1,1, 1 ]
                },
                { 'conv'   : [ 3,3,3, L2, L3 ],
                  'stride' : [ 1,1, 2,2, 1 ]
                },
                # resolution now 5 x 12 x 12
                { 'conv'   : [ 3,3,3, L3, L3 ],
                  'stride' : [ 1,1, 1,1, 1 ]
                },
                { 'conv'   : [ 3,3,3, L3, L4 ],
                  'stride' : [ 1,1, 2,2, 1 ]
                },
                # resolution now 5 x 6 x 6
                { 'conv'   : [ 3,3,3, L4, L4 ],
                  'stride' : [ 1,1, 1,1, 1 ]
                },
                { 'conv'   : [ 3,3,3, L4, L5 ],
                  'stride' : [ 1,2, 1,1, 1 ]
                },
                # resolution now 3 x 6 x 6
                { 'conv'   : [ 3,3,3, L5, L5 ],
                  'stride' : [ 1,1, 2,2, 1 ]
                },
                # resolution now 3 x 3 x 3
                ]

layers[ 'upscale'] = [
                { 'conv'   : [ 3, 3, C*3, L*3 + L],
                  'stride' : [ 1, 2, 2, 1 ]
                },
]

layers[ 'autoencoder_nodes' ] = []
layers[ '2D_decoder_nodes' ] = []
layers[ 'preferred_gpu' ] = 1
# if skip-connections are used
pinhole_connections = True

# 3D ENCODERS
encoders_3D = [
    { 'id': 'YUV',
      'channels': C,
      'preferred_gpu' : 0,
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
    { 'id': 'YUV',
      'channels': C,
      'preferred_gpu' : 1,
      'loss_fn':  'L2',
      'train':    True,
      'weight':   1.0,
    },
]

# MINIMIZERS
minimizers = [

  # center view super resolution
  { 'id'        : 'YUV_min',
    'losses_2D' : [ 'YUV' ],
    'optimizer' : 'Adam',
    'preferred_gpu' : 1,
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
training[ 'samples_per_batch' ] = 20

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
    'test_data_folder': "./test_data/"
}