#!/usr/bin/python3

# start session
import os.path
# timing and multithreading
import _thread
from queue import Queue
import time


# datasets
import autoencoder_data_streams as data

# global configuration
import config_autoencoder as hp


# final_layer_to_load = end_points['resnet_v1_50/block4']

# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#   saver.restore(sess, ckpt_path)
#   frozen_graph_def = convert_variables_to_constants(
#     sess, sess.graph_def,
#     output_node_names=[final_layer_to_load.name.split(':')[0]])
#
# frozen_graph = tf.Graph()
# with frozen_graph.as_default():
#   tf.import_graph_def(frozen_graph_def, name='')

# resnet

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = hp.config['VisibleGPU']

# import data
feeds = []
for file in hp.training_data:
  feeds.append( data.dataset( file, subsets=hp.training[ 'subsets' ] ))

# I/O queues for multithreading
inputs = Queue( 100 )

# Start evaluation thread
model_id = hp.network_model
model_path = './networks/' + model_id + '/'
os.makedirs( model_path, exist_ok=True )

# evaluator thread
from thread_train_v9 import trainer_thread
_thread.start_new_thread( trainer_thread,
                          ( model_path, hp, inputs ))


# wait a bit to not skew timing results with initialization
#time.sleep(10)
#print( 'DATA FETCH THREAD: starting train feed in 80 seconds' )
time.sleep(10)
print( 'DATA FETCH THREAD: starting train feed in 70 seconds' )
time.sleep(10)
print( 'DATA FETCH THREAD: starting train feed in 60 seconds' )
time.sleep(10)
print( 'DATA FETCH THREAD: starting train feed in 50 seconds' )
time.sleep(10)
print( 'DATA FETCH THREAD: starting train feed in 40 seconds' )
time.sleep(10)
print( 'DATA FETCH THREAD: starting train feed in 30 seconds' )
time.sleep(10)
print( 'DATA FETCH THREAD: starting train feed in 20 seconds' )
time.sleep(10)
print( 'DATA FETCH THREAD: starting train feed in 10 seconds' )
time.sleep(10)
print( 'DATA FETCH THREAD: starting train feed' )

# run training session
niter = 0
index = 0
while True:

  nfeed = 0
  for feed in feeds:
    batch = feed.next_batch( hp.config['ColorSpace'], hp.training[ 'samples_per_batch' ], 'training' )

    batch[ 'index' ] = index
    batch[ 'niter' ] = niter
    batch[ 'feed_id' ] = feed._file_id
    batch[ 'nfeed' ] = nfeed
    batch[ 'training'] = True

    if niter % hp.training[ 'log_interval' ] == 0:

      batch[ 'logging' ] = True
      batch[ 'logfile' ] = 'training_history.csv'
      inputs.put( batch )

      # pull the respective batch from the validation dataset and evaluate it
      batch = feed.next_batch( hp.config['ColorSpace'], hp.training[ 'samples_per_batch' ], 'validation' )
      batch[ 'index' ] = index
      batch[ 'niter' ] = niter
      batch[ 'feed_id' ] = feed._file_id
      batch[ 'nfeed' ] = nfeed
      batch[ 'logging' ] = True
      batch[ 'training'] = False
      batch[ 'logfile' ] = 'validation_history.csv'
      inputs.put( batch )

    else:
      # no logging, just train
      batch[ 'logging' ] = False
      inputs.put( batch )

    index += 1
    nfeed += 1

  niter += 1
