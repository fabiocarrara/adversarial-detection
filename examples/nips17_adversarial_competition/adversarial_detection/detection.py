"""Implementation of adversarial detection.


"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import faiss

import numpy as np
from tqdm import tqdm
from scipy.misc import imread

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception

slim = tf.contrib.slim


tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path', 'inception_v3.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'feature_layer', '', 'Layer to extract as features.')

tf.flags.DEFINE_string(
    'features_database', '', 'Database to search features in.')

tf.flags.DEFINE_string(
    'labels_database', '', 'Labels of the elements in the database.')

tf.flags.DEFINE_string(
    'output_file', '', 'Output file to save scores.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 16, 'How many images process at one time.')

tf.flags.DEFINE_integer(
    'k_neighbors', 1000, 'How many neighbors to search in the database.')

tf.flags.DEFINE_string(
    'knn_scoring', 'knn', 'kNN scoring scheme, one of: (knn, wknn, dwknn)')

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
  for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
    with tf.gfile.Open(filepath) as f:
      image = imread(f, mode='RGB').astype(np.float) / 255.0
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
  attack_name = os.path.basename(FLAGS.input_dir)
  
  features_db = faiss.read_index(FLAGS.features_database)
  labels_db = np.load(FLAGS.labels_database)

  tf.logging.set_verbosity(tf.logging.INFO)

  with tf.Graph().as_default():
    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=batch_shape)

    with slim.arg_scope(inception.inception_v3_arg_scope()):
      _, end_points = inception.inception_v3(
          x_input, num_classes=num_classes, is_training=False)

    # extacted_features = tf.squeeze(end_points['PreLogits'])
    # extacted_features = slim.avg_pool2d(end_points[FLAGS.feature_layer], (8, 8))
    # extacted_features = tf.squeeze(extacted_features)
    # Global Average Pool
    extacted_features = tf.reduce_mean(end_points[FLAGS.feature_layer], axis=(1,2))
    
    predicted_labels = tf.argmax(end_points['Predictions'], 1)

    # Run computation
    saver = tf.train.Saver(slim.get_model_variables())
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=tf.train.Scaffold(saver=saver),
        checkpoint_filename_with_path=FLAGS.checkpoint_path,
        master=FLAGS.master)

    with tf.train.MonitoredSession(session_creator=session_creator) as sess:
      with tf.gfile.Open(FLAGS.output_file, 'a+') as out_file:
        for filenames, images in tqdm(load_images(FLAGS.input_dir, batch_shape)):
          # predict labels and extract features
          labels, features = sess.run([predicted_labels, extacted_features], feed_dict={x_input: images})
          # search the database
          distances, idx = features_db.search(features, FLAGS.k_neighbors)
          # get the labels of retrieved elements
          retrieved_labels = labels_db[idx] + 1 # shift labels, 0 is now 'background' class
          # indicator variable I{c_i == c}
          indicator = (labels.reshape(-1,1) == retrieved_labels).astype(float)
          if FLAGS.knn_scoring == 'knn':
            scores = indicator.sum(axis=1) / FLAGS.k_neighbors
          elif FLAGS.knn_scoring == 'wknn':
            weights = 1. / np.arange(1, FLAGS.k_neighbors + 1)
            scores = (indicator * weights).sum(axis=1) / weights.sum()
          elif FLAGS.knn_scoring == 'dwknn':
            weights = 1. / distances
            scores = (indicator * weights).sum(axis=1) / weights.sum(axis=1)
          elif FLAGS.knn_scoring == 'lwknn':
            weights = 1. / np.log(1 + np.arange(1, FLAGS.k_neighbors + 1))
            scores = (indicator * weights).sum(axis=1) / weights.sum()
          elif FLAGS.knn_scoring == 'ldwknn':
            weights = 1. / np.log(1 + distances)
            scores = (indicator * weights).sum(axis=1) / weights.sum(axis=1)
          
          for filename, label, score in zip(filenames, labels, scores):
            out_file.write('{3},{0},{1},{2}\n'.format(filename, label, score, attack_name))


if __name__ == '__main__':
  tf.app.run()
