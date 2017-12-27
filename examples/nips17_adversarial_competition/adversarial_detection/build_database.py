"""Implementation of adversarial detection.

This code builds the search database extracing features from the network to
defend.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import lmdb
import numpy as np
from glob import iglob
from tqdm import tqdm
from scipy.misc import imread, imresize

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception

slim = tf.contrib.slim


tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path', 'inception_v3.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'feature_layer', 'PreLogits', 'Name of the layer from which extract features.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'feature_pool', 'avg_pool', 'How to aggregate spatial features (avg_pool | max_pool, default: avg_pool).')

tf.flags.DEFINE_string(
    'output_database', '', 'Output file to save the features.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 128, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS


def load_images(input_dir, batch_shape):
  """Read png images from input directory in batches.

  Args:
    input_dir: input directory
    batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

  Yields:
    filenames: list file names without path of each image
      Lenght of this list could be less than batch_size, in this case only
      first few images of the result are elements of the minibatch.
    images: array with all images from this batch
  """
  images = np.zeros(batch_shape)
  filenames = []
  idx = 0
  batch_size = batch_shape[0]
  for filepath in iglob(os.path.join(input_dir, 'n*/*.JPEG')):
    with tf.gfile.Open(filepath) as f:
      image = imread(f, mode='RGB')
      old_shape = np.array(image.shape[:2])
      ratio = 299.0 / old_shape.min()
      new_shape = (ratio * old_shape).astype(int)
      image = imresize(image, new_shape) # resize shortest to 299
      start = (new_shape - 299) // 2
      image = image[start[0]:start[0] + 299, start[1]:start[1] + 299,:] # central crop
      image = image.astype(np.float) / 255.0
    # Images for inception classifier are normalized to be in [-1, 1] interval.
    images[idx, :, :, :] = image * 2.0 - 1.0
    filenames.append(os.path.basename(filepath))
    idx += 1
    if idx == batch_size:
      yield filenames, images
      filenames = []
      images = np.zeros(batch_shape)
      idx = 0
  if idx > 0:
    yield filenames, images


def main(_):
  batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
  num_classes = 1001

  tf.logging.set_verbosity(tf.logging.INFO)
  
  layer_names = FLAGS.feature_layer.split(',')
  output_databases = FLAGS.output_database.split(',')
  poolings = FLAGS.feature_pool.split(',')

  with tf.Graph().as_default():
    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=batch_shape)

    with slim.arg_scope(inception.inception_v3_arg_scope()):
      _, end_points = inception.inception_v3(
          x_input, num_classes=num_classes, is_training=False)

    # feature_layer = tf.squeeze(end_points['PreLogits'])
    # feature_layer = slim.avg_pool2d(end_points[FLAGS.feature_layer], (8, 8))
    
    # Global Pool
    feature_layers = []
    if 'avg_pool' in poolings:
      feature_layers += [tf.reduce_mean(end_points[l], axis=(1,2)) for l in layer_names]
      
    if 'max_pool' in poolings:
      feature_layers += [tf.reduce_max(end_points[l], axis=(1,2)) for l in layer_names]
    
    # Run computation
    saver = tf.train.Saver(slim.get_model_variables())
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=tf.train.Scaffold(saver=saver),
        checkpoint_filename_with_path=FLAGS.checkpoint_path,
        master=FLAGS.master)

    envs = [lmdb.open(out_db, map_size=1e12) for out_db in output_databases]
    with tf.train.MonitoredSession(session_creator=session_creator) as sess:
      for filenames, images in tqdm(load_images(FLAGS.input_dir, batch_shape)):
        list_of_features = sess.run(feature_layers, feed_dict={x_input: images})
        for env, features in zip(envs, list_of_features):
          with env.begin(write=True) as txn:
            for filename, feature in zip(filenames, features):
              txn.put(filename, feature)
          

if __name__ == '__main__':
  tf.app.run()
