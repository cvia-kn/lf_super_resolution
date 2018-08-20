# Class definition for the combined CRY network
# drops: deep regression on angular patch stacks
#
# in this version, we take great care to have nice
# variable scope names.
#
# start session
import code
import tensorflow as tf
import numpy as np
import math
import libs.layers as layers
from tensorflow.image import yuv_to_rgb, resize_bicubic

from resnet_v1 import resnet_v1, resnet_arg_scope, resnet_v1_50
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import tensorflow.contrib.slim as slim

loss_min_coord_3D = 0
loss_max_coord_3D = 48
loss_max_coord_3D_1 = 96
loss_min_coord_2D = 0
loss_max_coord_2D = 48
loss_max_coord_2D_1 = 96

# main class defined in module
class create_cnn:

  def __init__( self, config ):

    # config (hyperparameters)
    self.config = config
    self.max_layer = config.config[ 'max_layer' ]
    self.interpolate = config.config['interpolate']
    # we get two input paths for autoencoding:
    # 1. vertical epi stack in stack_v
    # 2. horizontal epi stack in stack_h

    # both stacks have 9 views, patch size 16x16 + 16 overlap on all sides,
    # for a total of 48x48.
    self.C = config.C
    self.cv_pos = config.cv_pos
    self.D = config.D
    self.H = config.H
    self.W = config.W
    self.H_HR = config.H_HR
    self.W_HR = config.W_HR
    self.reuse_resnet = False

    # with tf.device('/device:GPU:%i' % (self.config.layers['preferred_gpu'])):
    #   self.X = tf.placeholder(tf.float32, shape=[None, self.H_HR, self.W_HR, 3], name='input_resnet')

    # regularization weights
    self.beta = 0.0001

    # input layers
    with tf.device( '/device:GPU:%i' % ( self.config.layers['preferred_gpu'] ) ):
      with tf.variable_scope( 'input' ):

        self.stack_v = tf.placeholder(tf.float32, shape=[None, self.D, self.H, self.W, self.C] )
        self.stack_h = tf.placeholder(tf.float32, shape=[None, self.D, self.H, self.W, self.C] )

        self.stack_shape = self.stack_v.shape.as_list()
        self.stack_shape[ 0 ] = -1

        self.phase = tf.placeholder(tf.bool, name='phase')
        self.keep_prob = tf.placeholder(tf.float32)
        self.noise_sigma = tf.placeholder(tf.float32)

    # FEATURE LAYERS
    self.batch_size = tf.shape(self.stack_v)[0]

    self.encoders_3D = dict()
    self.decoders_2D = dict()
    self.minimizers = dict()

    self.create_3D_encoders()
    self.merge_encoders()
    self.create_dense()
    self.create_vae_features()
    self.create_2D_decoders()
    self.setup_losses()

#
   # CREATE DECODER LAYERS FOR ADDITIONAL DECODERS CONFIGURED IN THE CONFIG FILE
   #
  def create_3D_encoders(self):
      for encoder_config in self.config.encoders_3D:
          with tf.device('/device:GPU:%i' % (encoder_config['preferred_gpu'])):
              self.create_3D_encoder(encoder_config)

  def create_3D_encoder(self, encoder_config):
      encoder = dict()
      encoder_id = encoder_config['id']
      ids = []
      for i in range(0,len(self.config.layer_config)):
          ids.append(self.config.layer_config[i]['id'])
      pos = ids.index(encoder_id)
      layout = []
      layout.insert(0,self.config.layer_config[pos]['layout'][0])
      for i in range(0,len(self.config.layers['encoder_3D'])):
          layout.append(self.config.layers['encoder_3D'][i])
      print('creating encoder pipeline for ' + encoder_id)
      with tf.variable_scope(encoder_id):
          encoder['id'] = encoder_id
          encoder['channels'] = encoder_config['channels']
          encoder['preferred_gpu'] = encoder_config['preferred_gpu']
          encoder['variables'] = []
          encoder['features_v'] = None
          encoder['features_h'] = None
          encoder['conv_layers_v'] = []
          encoder['conv_layers_h'] = []
          ####################################################################################################
          # create encoder variables
          last_layer = min(len(layout), self.max_layer)
          for i in range(0, last_layer):
              layer_id = "encoder_%i" % i
              print('    creating 3D encoder variables ' + layer_id)
              encoder['variables'].append(layers.encoder_variables(layer_id, layout[i]))
          ####################################################################################################
          # create 3D encoder layers for stacks
          start = self.config.layer_config[pos]['start']
          end = self.config.layer_config[pos]['end']
          shape = [self.stack_shape[0],self.stack_shape[1],self.stack_shape[2],
                   self.stack_shape[3],encoder['channels']]
          if encoder['features_v'] == None:
              encoder['features_v'] = self.stack_v[:,:,:,:,start:end]
              encoder['features_v'] = tf.reshape(encoder['features_v'], shape) # why we need to reshape ?
          if encoder['features_h'] == None:
              encoder['features_h'] = self.stack_h[:,:,:,:,start:end]
              encoder['features_h'] = tf.reshape(encoder['features_h'], shape)
          print('    CREATING 3D encoder layers for %s ' % encoder_id)
          for i in range(0, last_layer):
              layer_id_v = "v_%s_%i" % (encoder_id,i)
              layer_id_h = "h_%s_%i" % (encoder_id,i)
              print('    generating downconvolution layer structure for %s %i' % (encoder_id,i))
              encoder['conv_layers_v'].append(layers.layer_conv3d(layer_id_v,
                                                                  encoder['variables'][i],
                                                                  encoder['features_v'],
                                                                  self.phase,
                                                                  self.config.training))
              encoder['conv_layers_h'].append(layers.layer_conv3d(layer_id_h,
                                                                  encoder['variables'][i],
                                                                  encoder['features_h'],
                                                                  self.phase,
                                                                  self.config.training))
              # update layer shapes
              encoder['variables'][i].input_shape = encoder['conv_layers_v'][i].input_shape
              encoder['variables'][i].output_shape = encoder['conv_layers_v'][i].output_shape
              # final encoder layer: vertical/horizontal features
              encoder['features_v'] = encoder['conv_layers_v'][-1].out
              encoder['features_h'] = encoder['conv_layers_h'][-1].out
          ####################################################################################################
          # create dense layers
          print('    creating dense layers for %s' %encoder_id)
          encoder['feature_shape'] = encoder['features_v'].shape.as_list()
          sh = encoder['feature_shape']
          encoder['encoder_input_size'] = sh[1] * sh[2] * sh[3] * sh[4]
          # setup shared feature space between horizontal/vertical encoder
          # encoder['features'] = tf.concat([tf.reshape(encoder['features_h'], [-1, encoder['encoder_input_size']]),
          #                           tf.reshape(encoder['features_v'], [-1, encoder['encoder_input_size']])], 1)
          encoder['features_transposed'] = tf.concat(
              [tf.reshape(tf.transpose(encoder['features_h'], [0, 1, 3, 2, 4]), [-1, encoder['encoder_input_size']]),
               tf.reshape(encoder['features_v'], [-1, encoder['encoder_input_size']])], 1)
          encoder['encoder_nodes'] = encoder['features_transposed'].shape.as_list()[1]

          self.encoders_3D[encoder_id] = encoder

  def merge_encoders(self):
      self.merged = dict()
      # self.merged['features'] = None
      self.merged['features_transposed'] = None
      self.merged['encoder_nodes'] = None
      if self.config.layers['merge_encoders']:
          print('Merge the encoders in the bottelneck')
          with tf.device('/device:GPU:%i' % (self.config.layers['preferred_gpu'])):
            for encoder_config in self.config.encoders_3D:
                encoder_id = encoder_config['id']
                # if self.merged['features'] is None:
                #     self.merged['features'] = self.encoders_3D[encoder_id]['features']
                # else:
                #     self.merged['features']= tf.concat([self.merged['features'], self.encoders_3D[encoder_id]['features']],1)

                if self.merged['features_transposed'] is None:
                    self.merged['features_transposed'] = self.encoders_3D[encoder_id]['features_transposed']
                else:
                    self.merged['features_transposed'] = tf.concat([self.merged['features_transposed'],
                                                                   self.encoders_3D[encoder_id]['features_transposed']],1)
            self.merged['encoder_nodes'] = self.merged['features_transposed'].shape.as_list()[1]

  def create_dense(self):
      if len(self.config.layers['2D_encoder_nodes']) > 0:
          num_nodes = self.config.layers['2D_encoder_nodes'][0]
          if self.config.layers['merge_encoders']:
              with tf.device('/device:GPU:%i' % (self.config.layers['preferred_gpu'])):
                  self.merged['features_transposed'] = layers.bn_dense(self.merged['features_transposed'],
                                                     self.merged['encoder_nodes'], num_nodes, self.phase, self.config.training,
                                                     'bn_encoder_merged_in')
                  self.merged['encoder_nodes'] = self.merged['features_transposed'].shape.as_list()[1]
          else:
              for encoder_config in self.config.encoders_3D:
                  with tf.device('/device:GPU:%i' % (encoder_config['preferred_gpu'])):
                    encoder_id = encoder_config['id']
                    self.encoders_3D[encoder_id]['features_transposed'] = layers.bn_dense(self.encoders_3D[encoder_id]['features_transposed'],
                                             self.encoders_3D[encoder_id]['encoder_nodes'], num_nodes, self.phase, self.config.training,
                                             'bn_encoder_' + encoder_id + '_in')
                    self.encoders_3D[encoder_id]['encoder_nodes'] = self.encoders_3D[encoder_id]['features_transposed'].shape.as_list()[1]

  def create_vae_features(self):
      if self.config.layers['vae']:
          with tf.device('/device:GPU:%i' % (self.config.layers['preferred_gpu'])):
              with tf.variable_scope('vae_encoder'):
                  print('  creating vae encoder nodes.')
                  self.merged['mn_features_transposed'] = None
                  self.merged['sd_features_transposed'] = None
                  n = self.merged['encoder_nodes']
                  self.merged['mn_features_transposed'] = layers.bn_dense(
                      self.merged['features_transposed'], n, n, self.phase,
                      self.config.training,
                      'mn_feature_transposed_encoder_%i' % n)

                  self.merged['sd_features_transposed'] = layers.bn_dense(
                      self.merged['features_transposed'], n, n, self.phase,
                      self.config.training,
                      'sd_feature_transposed_encoder_%i' % n)
                  tf.summary.histogram('hist_mn', self.merged['mn_features_transposed'])
                  tf.summary.histogram('hist_sd', self.merged['sd_features_transposed'])

                  self.dim_mn_features = self.merged['mn_features_transposed'].shape.as_list()[1]
                  # Draw samples from the distribution P(z|X),
                  # to generate samples for the decoder input from the encoder distribution
                  # For this, we need a normal distribution for "reparameterization trick"
                  eps = tf.random_normal(shape=(self.batch_size, self.dim_mn_features), mean=0.0,
                                         stddev=1.0)
                  # we get z_mean, z_log_sigma_sq from encoder, and draw z from N(z_mean,z_sigma^2)
                  # self.features = tf.add(self.mn_features, tf.multiply(tf.multiply(self.config.sigma_vae,tf.sqrt(tf.exp(self.sd_features))), eps))  # using reparameterization trick
                  # self.merged['features_transposed'] = tf.add(self.merged['mn_features_transposed'],
                  #                        tf.multiply(tf.multiply(self.config.sigma_vae,tf.sqrt(tf.exp(self.merged['sd_features_transposed']))), eps))  # using reparameterization trick
                  self.merged['features_transposed'] = tf.add(self.merged['mn_features_transposed'],
                                                              tf.multiply(tf.sqrt(tf.exp(self.merged['sd_features_transposed'])),eps))
                  self.merged['encoder_nodes'] = self.merged['features_transposed'].shape.as_list()[1]
  #
  # CREATE DECODER LAYERS FOR ADDITIONAL DECODERS CONFIGURED IN THE CONFIG FILE
  #
  def create_2D_decoders( self ):
    for decoder_config in self.config.decoders_2D:
      with tf.device( '/device:GPU:%i' % ( decoder_config[ 'preferred_gpu' ] )):
        self.create_2D_decoder( decoder_config)


  def create_2D_decoder( self, decoder_config):

    decoder = dict()
    decoder_id = decoder_config[ 'id' ]
    ids = []
    for i in range(0, len(self.config.layer_config)):
        ids.append(self.config.layer_config[i]['id'])
    pos_layout = ids.index(decoder_id)
    print( 'creating decoder pipeline ' + decoder_id )

    # create a decoder pathway (center view only)
    with tf.variable_scope( decoder_id ):

      decoder[ 'id' ] = decoder_id
      decoder[ 'channels' ] = decoder_config[ 'channels' ]
      decoder[ 'loss_fn' ] = decoder_config[ 'loss_fn' ]
      decoder[ 'weight' ] = decoder_config[ 'weight' ]
      decoder[ 'train' ] = decoder_config[ 'train' ]
      decoder[ 'preferred_gpu' ] = decoder_config[ 'preferred_gpu' ]
      decoder['skip_connection'] = decoder_config['skip_connection']
      decoder['no_relu'] = decoder_config['no_relu']
      decoder[ 'start'] = self.config.layer_config[pos_layout]['start']
      decoder[ 'end'] = self.config.layer_config[pos_layout]['end']
      decoder['percep_loss'] = decoder_config['percep_loss']

      decoder[ '2D_variables' ] = []
      decoder[ 'concat_variables'] = []
      decoder[ 'upscale_variables'] = []

      sh = self.encoders_3D[decoder_id]['feature_shape']
      decoder['nodes'] = sh[2] * sh[3] * sh[4] * self.config.patch_weight
      if self.config.layers['merge_encoders']:
          if decoder['nodes'] == self.merged['encoder_nodes']:
              print('no dense layer for the decoder')
              decoder['upconv_in'] = self.merged['features_transposed']
          else:
              decoder['upconv_in'] = layers.bn_dense(self.merged['features_transposed'],
                                                     self.merged['encoder_nodes'],
                                                     decoder['nodes'], self.phase, self.config.training,
                                                     'bn_decoder_' + decoder_id + '_in')
      else:
          if decoder['nodes'] == self.encoders_3D[decoder_id]['encoder_nodes']:
              print('no dense layer for the decoder')
              decoder['upconv_in'] = self.encoders_3D[decoder_id]['features_transposed']
          else:
              decoder['upconv_in'] = layers.bn_dense( self.encoders_3D[decoder_id]['features_transposed'],
                                                  self.encoders_3D[decoder_id]['encoder_nodes'],
                                                  decoder['nodes'], self.phase, self.config.training,
                                                  'bn_decoder_' + decoder_id + '_in' )

      sh = self.encoders_3D[decoder_id]['feature_shape']
      decoder['upconv'] = tf.reshape(decoder['upconv_in'], [-1, sh[2], sh[3], sh[4] * self.config.patch_weight])
      decoder['layers'] = []

      ########################################################################################################
      # decoder variables
      layout = []
      layout.insert(0, self.config.layer_config[pos_layout]['layout'][0])
      for i in range(0, len(self.config.layers['encoder_3D'])):
          layout.append(self.config.layers['encoder_3D'][i])

      last_layer = min(len(layout), self.max_layer)
      if decoder['skip_connection']:
          patches = self.config.nviews
          for i in range(0, last_layer):
              layer_id_cat = "skip_patch_%s_%i" % (decoder_id, i)
              print('    generating decoder ' + layer_id_cat)
              patches = math.ceil(patches/layout[i]['stride'][1])
              decoder['concat_variables'].append(layers.concat_variables(layer_id_cat, layout[i],patches,
                                                 input_features = [], output_features = []))

          layer_id_cat = "skip_patch_%s_%i" % (decoder_id, i+1)
          decoder['concat_variables'].insert(0,layers.concat_variables(layer_id_cat, layout[i], self.config.nviews,
                                       input_features = self.config.layer_config[pos_layout]['layout'][0]['conv'][-2],
                                       output_features = self.config.layer_config[pos_layout]['upscale'][0]['conv'][-2]))

      layout[0] = self.config.layer_config[pos_layout]['upscale'][0]
      for i in range(0, last_layer):
          layer_id = "decoder_%s_%i" % (decoder_id, i)
          print('    generating upconvolution variables ' + layer_id)
          decoder['2D_variables'].append(layers.decoder_variables_2D(layer_id,layout[i],
                                         i, last_layer, self.config.patch_weight, decoder['skip_connection'] ))


      for i in range(0, last_layer):
          layer_id = "decoder_%s_layer%i" % (decoder_id, last_layer - i - 1)
          layer_id_cat = "skip_patch_%s_layer%i" % (decoder_id, last_layer - i - 1)
          print('    generating upconvolution layer structure ' + layer_id)
          out_channels = -1
          # evil hack
          no_relu = False
          decoder['layers'].insert(-1-i,
                                   layers.layer_upconv2d_v2(layer_id,
                                                            decoder['2D_variables'][-1 - i],
                                                            self.batch_size,
                                                            decoder['upconv'],
                                                            self.phase,
                                                            self.config.training,
                                                            out_channels=out_channels,
                                                            no_relu=no_relu))
          if decoder['skip_connection']:
              if i != last_layer - 1:
                  encoder_skip = None
                  for skip_id in decoder_config['skip_id']:
                      if encoder_skip is None:
                          encoder_skip = layers.layer_concat(self.encoders_3D[skip_id]['conv_layers_v'][-2 - i].out,
                                                         self.encoders_3D[skip_id]['conv_layers_h'][-2 - i].out).out
                      else:
                          encoder_skip = tf.concat([encoder_skip, layers.layer_concat(self.encoders_3D[skip_id]['conv_layers_v'][-2 - i].out,
                                                         self.encoders_3D[skip_id]['conv_layers_h'][-2 - i].out).out], -1)

                  concat_cv = layers.layer_conv_one(layer_id_cat,
                                                    decoder['concat_variables'][-2 - i],encoder_skip).out
              else:
                  # start = self.config.layer_config[pos_layout]['start']
                  # end = self.config.layer_config[pos_layout]['end']
                  # concat_cv = layers.layer_conv_one(layer_id_cat,decoder['concat_variables'][-2 - i],
                  #                                   layers.layer_concat(self.stack_v[:, :, :, :, start:end],
                  #                                   self.stack_h[:, :, :, :, start:end]).out).out
                  concat_cv = layers.layer_conv_one(layer_id_cat,decoder['concat_variables'][-2 - i],
                                                    layers.layer_concat(self.stack_v, self.stack_h).out).out
              decoder['upconv'] = tf.concat([decoder['layers'][-1 - i].out, concat_cv], axis=3)
          else:
              decoder['upconv'] = decoder['layers'][-1 - i].out

      # decoder['layers'] = tf.reverse(decoder['layers'],0)
      decoder['upconv_reduce'] = decoder['upconv'][:, loss_min_coord_2D:loss_max_coord_2D,
                                 loss_min_coord_2D:loss_max_coord_2D, :]

      decoder['input'] = tf.placeholder(tf.float32, [None, self.H_HR, self.W_HR, decoder['channels']])
      decoder['input_reduce'] = decoder['input'][:, loss_min_coord_2D:loss_max_coord_2D_1,
                                loss_min_coord_2D:loss_max_coord_2D_1, :]


      if self.interpolate:
          decoder['bicubic'] = resize_bicubic(self.stack_v[:, self.cv_pos, :, :, decoder['start']:decoder['end']],
                                              [self.H_HR, self.W_HR])
      layout = []
      for i in range(0, len(self.config.layers['upscale'])):
          layout.append(self.config.layers['upscale'][i])

      last_layer_up = min(len(layout), self.max_layer)
      # change later
      skip = decoder['skip_connection']
      for i in range(0,last_layer_up):
          layer_id_upscale = "upscale_%s_%i" %(decoder_id,i)
          print('    creating variables for ' + layer_id_upscale)
          if layout[i]['bicubic_connection']:
              bicubic_features = decoder['bicubic'].shape.as_list()[-1]
          else:
              bicubic_features = 0

          decoder['upscale_variables'].append(layers.upscale_variables(layer_id_upscale, layout[i],self.config.patch_weight,
                                                                       skip, bicubic_features))
          skip = False


      layout_final = self.config.layer_config[pos_layout]['final'][0]

      no_relu = False
      for upscale in range(0, last_layer_up):
          out_channels = -1
          layer_id = "upscale_%s_%i" % (decoder_id, upscale)
          print('    creating layers for ' + layer_id)

          if self.interpolate and layout[upscale]['bicubic_connection']:
              decoder['upconv'] = tf.concat([decoder['upconv'],decoder['bicubic']],axis=3)


          decoder['upconv'] = layers.layer_upconv2d_v2(layer_id,
                                                       decoder['upscale_variables'][upscale],
                                                       self.batch_size,
                                                       decoder['upconv'],
                                                       self.phase,
                                                       self.config.training,
                                                       out_channels=out_channels,
                                                       no_relu=no_relu).out

      no_relu = decoder['no_relu']
      decoder['SR'] = layers.layer_pure_conv2D("upscale_final",
                                               layout_final,
                                               decoder['upconv'],
                                               self.phase,
                                               self.config.training, no_relu = no_relu).out
      self.decoders_2D[decoder_id] = decoder

    # with slim.arg_scope(resnet_arg_scope()):
    #     with tf.variable_scope('', reuse=tf.AUTO_REUSE):
    #        logits_sr, end_points_sr = resnet_v1_50(decoder['SR'], num_classes=1000, is_training=False)
    #        # logits_input, end_points_input = resnet_v1_50(decoder['input_reduce'], num_classes=1000, is_training=False)
    #
    # with tf.variable_scope(decoder_id):
    #     decoder['logits_sr'] = tf.identity(logits_sr)
    #     decoder['end_point_sr'] = tf.identity(end_points_sr['resnet_v1_50/block4'])
    #     # decoder['logits_input'] = tf.identity(logits_input)
    #     # decoder['end_point_input'] = tf.identity(end_points_input['resnet_v1_50/block4'])



  def add_training_ops( self ):

    print( 'creating training ops' )

    # what needs to be updated before training
    self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    # L2-loss on feature layers

    for cfg in self.config.minimizers:

      minimizer = dict()
      minimizer[ 'id' ] = cfg[ 'id' ]
      print( '  minimizer ' + cfg[ 'id' ] )

      with tf.device( '/device:GPU:%i' % ( cfg[ 'preferred_gpu' ] )):
        minimizer[ 'loss' ] = 0
        minimizer[ 'requires' ] = []


        if 'losses_2D' in cfg:
          for id in cfg[ 'losses_2D' ]:
            if 'computational' in self.decoders_2D[id]:
                  minimizer['loss'] += self.decoders_2D[id]['weight'] * self.decoders_2D[id]['loss']
            elif self.decoders_2D[id][ 'train' ]:
              minimizer[ 'loss' ] += self.decoders_2D[id]['weight'] * (self.decoders_2D[id]['loss'] + self.decoders_2D[id]['loss_p'])
              # minimizer[ 'requires' ].append( self.decoders_2D[id]['id'] )

        with tf.control_dependencies( self.update_ops ):
          # Ensures that we execute the update_ops before performing the train_step
          minimizer[ 'optimizer' ] = tf.train.AdamOptimizer( cfg[ 'step_size' ] )
          # gradient clipping
          gradients, variables = zip(
            *minimizer['optimizer'].compute_gradients(minimizer['loss'], colocate_gradients_with_ops=True))
          gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
          minimizer['train_step'] = minimizer['optimizer'].apply_gradients(zip(gradients, variables))

      self.minimizers[ cfg[ 'id' ] ] = minimizer

  def resnet_forward(self, x, layer, scope):
    x = 255.0 * (0.5 * (x + 1.0))
    # subtract means
    mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3],
                         name='img_mean')  # RGB means from VGG paper
    x = x - mean
    # send through resnet
    with slim.arg_scope(resnet_arg_scope()):
        _, layers = resnet_v1_50(x, num_classes=1000, is_training=False, reuse=self.reuse_resnet)
    self.reuse_resnet = True
    return layers['resnet_v1_50/' + layer]

  # add training ops for additional decoder pathway (L2 loss)
  def setup_losses( self ):
   for id in self.decoders_2D:
      # encoder decoder have same id for correspondence
      for k in range(0, len(self.encoders_3D[id]['variables'])):
           tf.summary.histogram(self.encoders_3D[id]['variables'][k].encoder_W.name,
                                self.encoders_3D[id]['variables'][k].encoder_W)
           tf.summary.histogram(self.encoders_3D[id]['variables'][k].encoder_b.name,
                                self.encoders_3D[id]['variables'][k].encoder_b)
      for k in range(0, len(self.decoders_2D[id]['2D_variables'])):
           tf.summary.histogram(self.decoders_2D[id]['2D_variables'][k].decoder_W.name,
                                self.decoders_2D[id]['2D_variables'][k].decoder_W)
           tf.summary.histogram(self.decoders_2D[id]['2D_variables'][k].decoder_b.name,
                                self.decoders_2D[id]['2D_variables'][k].decoder_b)
      for k in range(0, len(self.decoders_2D[id]['upscale_variables'])):
           tf.summary.histogram(self.decoders_2D[id]['upscale_variables'][k].decoder_W.name,
                                self.decoders_2D[id]['upscale_variables'][k].decoder_W)
           tf.summary.histogram(self.decoders_2D[id]['upscale_variables'][k].decoder_b.name,
                                self.decoders_2D[id]['upscale_variables'][k].decoder_b)
      # loss function for auto-encoder
      with tf.device( '/device:GPU:%i' % ( self.decoders_2D[id][ 'preferred_gpu' ] )):
        with tf.variable_scope( 'training_2D_' + id ):
            if self.decoders_2D[id][ 'loss_fn' ] == 'L2':
              print( '  creating L2-loss for decoder pipeline ' + id )
              self.decoders_2D[id]['loss'] = tf.losses.mean_squared_error( self.decoders_2D[id]['input_reduce'],
                                                                           self.decoders_2D[id]['SR'] )/ (2.0 * self.config.sigma_vae**2)

              tf.summary.scalar('loss' + id, self.decoders_2D[id]['loss'])
        self.decoders_2D[id]['loss_p'] = 0
        if len(self.decoders_2D[id][ 'percep_loss' ]) > 0:
            for p_layer in self.decoders_2D[id][ 'percep_loss' ]:
                with tf.name_scope('resnet_v1_50') as scope:
                    resnet_y = self.resnet_forward(self.decoders_2D[id]['input_reduce'], p_layer, scope)
                with tf.name_scope('resnet_v1_50') as scope:
                    resnet_y_pred = self.resnet_forward(self.decoders_2D[id]['SR'], p_layer, scope)
                with tf.variable_scope('training_2D_' + id):
                    self.decoders_2D[id]['loss_p'] = self.decoders_2D[id]['loss_p'] + tf.losses.mean_squared_error( resnet_y, resnet_y_pred )
            tf.summary.scalar('loss_p' + id, self.decoders_2D[id]['loss_p'])

   if self.config.layers['vae']:
       KL_loss = dict()
       KL_loss['id'] = 'KL_divergence'
       KL_loss['loss'] = tf.reduce_mean(- 0.5 * tf.reduce_sum(1 + self.merged['sd_features_transposed'] -\
                        tf.square(self.merged['mn_features_transposed']) - tf.exp(self.merged['sd_features_transposed']), 1))
       KL_loss['computational'] = True
       KL_loss['weight'] = 0.5
       self.decoders_2D['KL_divergence'] = KL_loss
       tf.summary.scalar('loss_KL', self.decoders_2D['KL_divergence']['loss'])
   if self.config.config['ColorSpace']=='YUV':
        tf.summary.image(id + '_input_hres', tf.expand_dims(self.decoders_2D['Y']['input_reduce'][:, :, :, 0], -1), max_outputs=3)
        tf.summary.image(id + '_input_stacks', tf.expand_dims(self.stack_v[:, 4, :, :, 0], -1), max_outputs=3)
        tf.summary.image(id + '_res', tf.expand_dims(self.decoders_2D['Y']['SR'][:, :, :, 0], -1), max_outputs=3)
   elif self.config.config['ColorSpace'] == 'YCBCR':
        tf.summary.image(id + '_input_hres', tf.expand_dims(self.decoders_2D['Y']['input_reduce'][:,:,:,0],-1), max_outputs=3)
        tf.summary.image(id + '_input_stacks', tf.expand_dims(self.stack_v[:, 4, :, :, 0],-1), max_outputs=3)
        tf.summary.image(id + '_res', tf.expand_dims(self.decoders_2D['Y']['SR'][:,:,:,0],-1), max_outputs=3)
   elif self.config.config['ColorSpace'] == 'LAB':
        tf.summary.image(id + '_input_hres', tf.expand_dims(self.decoders_2D['L']['input_reduce'][:, :, :, 0], -1), max_outputs=3)
        tf.summary.image(id + '_input_stacks', tf.expand_dims(self.stack_v[:, 4, :, :, 0], -1), max_outputs=3)
        tf.summary.image(id + '_res', tf.expand_dims(self.decoders_2D['L']['SR'][:, :, :, 0], -1), max_outputs=3)
   elif self.config.config['ColorSpace'] == 'RGB':
        tf.summary.image(id + '_input_hres', self.decoders_2D['RGB']['input_reduce'], max_outputs=3)
        tf.summary.image(id + '_input_stacks', self.stack_v[:, 4, :, :, 0:3], max_outputs=3)
        tf.summary.image(id + '_res', self.decoders_2D['RGB']['SR'], max_outputs=3)


   self.merged = tf.summary.merge_all()

  # initialize new variables
  def initialize_uninitialized( self, sess ):
    global_vars          = tf.global_variables()
    is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    
    for i in not_initialized_vars:
      print( str(i.name) )

    if len(not_initialized_vars):
      sess.run(tf.variables_initializer(not_initialized_vars))

  # prepare input
  def prepare_net_input( self, batch ):

      # default params for network input
      nsamples   = batch[ 'stacks_v' ].shape[0]

      # default params for network input
      # net_in = {  self.input_features:np.zeros([nsamples, self.encoder_nodes], np.float32),
      #             self.keep_prob:       1.0,
      #             self.phase:           False,
      #             self.noise_sigma:     self.config.training[ 'noise_sigma' ] }
      net_in = {  self.keep_prob:       1.0,
                  self.phase:           False,
                  self.noise_sigma:     self.config.training[ 'noise_sigma' ] }

      # bind 2D decoder inputs to batch stream
      for id in self.decoders_2D:

        decoder = self.decoders_2D[id]
        if 'input' in decoder:
            start = decoder['start']
            end = decoder['end']
            net_in[decoder['input']] = batch['cv'][:, :, :, start:end]

      # bind global input to stream

      net_in[ self.stack_v ] = batch[ 'stacks_v' ]
      net_in[ self.stack_h ] = batch[ 'stacks_h' ]


      return net_in