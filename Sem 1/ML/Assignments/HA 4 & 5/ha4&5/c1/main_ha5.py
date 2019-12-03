#Nandini Vij
#822048806

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import scipy
import matplotlib.pyplot as plt
from keras.datasets import mnist
from util import func_confusion_matrix

import tensorflow as tf
import time
from datetime import datetime
import os.path
import data_helpers
import func_two_layer_fc
from util import func_confusion_matrix
from util import get_confusion_matrix_and_test

# load (downloaded if needed) the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# transform each image from 28 by28 to a 784 pixel vector
pixel_count = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], pixel_count).astype('float32')
x_test = x_test.reshape(x_test.shape[0], pixel_count).astype('float32')

# normalize inputs from gray scale of 0-255 to values between 0-1
x_train = x_train / 255
x_test = x_test / 255
x = x_train
y = y_train

x_train, x_val = x[:50000,:], x[50000:,:]
y_train, y_val = y[:50000,], y[50000:,]



# Please write your own codes in responses to the homework assignment 5
####################################################################
############## step-0: setting parameters       ####################
####################################################################

# Model parameters as external flags
flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate for the training.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 150, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('batch_size', 400,
  'Batch size. Must divide dataset sizes without remainder.')
flags.DEFINE_string('train_dir', 'tf_logs',
  'Directory to put the training data.')
flags.DEFINE_float('reg_constant', 0.1, 'Regularization constant.')

FLAGS._parse_flags()
print('\nParameters:')
for attr, value in sorted(FLAGS.__flags.items()):
  print('{} = {}'.format(attr, value))
print()

IMAGE_PIXELS = 784
CLASSES = 10
# Put logs for each run in separate directory
logdir = FLAGS.train_dir + '/' + datetime.now().strftime('%Y%m%d-%H%M%S') + '/'

beginTime = time.time()



####################################################################
############## step-2: Prepare the Tensorflow graph ################
####################################################################

# -----------------------------------------------------------------------------
# Prepare the Tensorflow graph
# (We're only defining the graph here, no actual calculations taking place)
# -----------------------------------------------------------------------------

# Define input placeholders
images_placeholder = tf.placeholder(tf.float32, shape=[None, IMAGE_PIXELS],
  name='images')
labels_placeholder = tf.placeholder(tf.int64, shape=[None], name='image-labels')

# Operation for the classifier's result
logits = func_two_layer_fc.inference(images_placeholder, IMAGE_PIXELS,
  FLAGS.hidden1, CLASSES, reg_constant=FLAGS.reg_constant)

# Operation for the loss function
loss = func_two_layer_fc.loss(logits, labels_placeholder)

# Operation for the training step
train_step = func_two_layer_fc.training(loss, FLAGS.learning_rate)

# Operation calculating the accuracy of our predictions
accuracy = func_two_layer_fc.evaluation(logits, labels_placeholder)

# Operation merging summary data for TensorBoard
summary = tf.summary.merge_all()

# Define saver to save model state at checkpoints
saver = tf.train.Saver()

# -----------------------------------------------------------------------------
# Run the TensorFlow graph
# -----------------------------------------------------------------------------

with tf.Session() as sess:
  # Initialize variables and create summary-writer
  sess.run(tf.global_variables_initializer())
  summary_writer = tf.summary.FileWriter(logdir, sess.graph)

  # Generate input data batches
  zipped_data = zip(x_train, y_train)
  batches = data_helpers.gen_batch(list(zipped_data), FLAGS.batch_size,
    FLAGS.max_steps)

  for i in range(FLAGS.max_steps):

    # Get next input data batch
    batch = next(batches)
    images_batch, labels_batch = zip(*batch)
    feed_dict = {
      images_placeholder: images_batch,
      labels_placeholder: labels_batch
    }

    # Periodically print out the model's current accuracy
    if i % 100 == 0:
      train_accuracy = sess.run(accuracy, feed_dict=feed_dict)
      print('Step {:d}, training accuracy {:g}'.format(i, train_accuracy))
      summary_str = sess.run(summary, feed_dict=feed_dict)
      summary_writer.add_summary(summary_str, i)

    # Perform a single training step
    sess.run([train_step, loss], feed_dict=feed_dict)

    # Periodically save checkpoint
    if (i + 1) % 1000 == 0:
      checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint')
      saver.save(sess, checkpoint_file, global_step=i)
      print('Saved checkpoint')

  # After finishing the training, evaluate on the test set
  test_accuracy = sess.run(accuracy, feed_dict={
    images_placeholder: x_val,
    labels_placeholder: y_val})
  print('Validation Test accuracy {:g}'.format(test_accuracy))
  pred = tf.arg_max(logits, 1)
  predicted_val = pred.eval(feed_dict = {images_placeholder: x_test})

  confusionMatrix, accuracy, recallArray, precisionArray = get_confusion_matrix_and_test(y_test, predicted_val)

  print('confusion matrix: \n {} \n'.format(confusionMatrix))
  print('Accuracy: \n{}\n'.format(accuracy))
  print('recallArray: \n{}\n'.format(recallArray))
  print('precisionArray: \n{}\n'.format(precisionArray))


  #w = 10
  #h = 10
  #fig = plt.figure(figsize=(8, 8))
  #columns = 4
  #rows = 5
  #for i in range(1, columns * rows + 1):
   #  fig.add_subplot(rows, columns, i)
   #  plt.imshow(x_train[i, :].reshape((28, 28)))
  #plt.show()

  ## Wrong prediction for first 100 errors
  b = 0
  row = 3
  col = 5
  fig1 = plt.figure(figsize=(5, 5))
  for x in range(0, 100):
      if predicted_val[x,] != y_test[x, ]:
          b = b+1
          fig1.add_subplot(row, col, b)
          plt.imshow(x_test[b, :].reshape((28,28)))
  plt.show()




endTime = time.time()
print('Total time: {:5.2f}s'.format(endTime - beginTime))

