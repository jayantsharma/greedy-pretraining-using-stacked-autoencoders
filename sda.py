"""A deep MNIST classifier using stacked autoencoders."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

def stackedAutoencoder(x, num_layers=3, layer_units=[784, 400, 400, 400], pretraining=True):
    # ae_vars = {'h_enc0': x}
    reprs = {0: x}
    for i in range(1, num_layers+1):
        # reuse = False if pretraining and i == num_layers else True
        with tf.variable_scope('ae{}'.format(i), reuse=tf.AUTO_REUSE):
            x_hat = reprs[i-1]
            if i == num_layers and pretraining:
                input = tf.multiply(
                          x_hat, 
                          tf.contrib.distributions.Bernoulli(
                              probs=0.8, dtype=tf.float32
                              ).sample([layer_units[i-1]])
                        ) 
            else:
                input = x_hat

            with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
                W_enc = tf.get_variable('W', initializer=tf.truncated_normal([layer_units[i-1], layer_units[i]], stddev=0.1))
                b_enc = tf.get_variable('b', initializer=tf.constant(0.1, shape=[layer_units[i]]))
                reprs[i] = tf.nn.sigmoid(tf.matmul(input, W_enc) + b_enc)

    if pretraining:
        with tf.variable_scope('ae{}'.format(i)):
            with tf.variable_scope('decoder'):
                W_dec = tf.get_variable('W', initializer=tf.truncated_normal([layer_units[i], layer_units[i-1]], stddev=0.1))
                b_dec = tf.get_variable('b', initializer=tf.constant(0.1, shape=[layer_units[i-1]]))
                recon = tf.matmul(reprs[i], W_dec) + b_dec
                # recon = tf.nn.sigmoid(tf.matmul(h_enc, W_dec) + b_dec)
                return x_hat, recon

    return reprs[i]

def deepnn(x, num_ae_layers=3):
    h = stackedAutoencoder(x, num_ae_layers, pretraining=False)
    with tf.variable_scope('fc'):
        W = tf.get_variable('W', initializer=tf.truncated_normal([400,10], stddev=0.1))
        b = tf.get_variable('b', initializer=tf.constant(0.1, shape=[10]))
        logits = tf.matmul(h, W) + b
        return logits

def pretrain(layer, mnist, sess, iters, pretrain_flag):
  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])

  # Build the graph for the deep net
  x_hat, x_recon = stackedAutoencoder(x, layer)

  with tf.name_scope('greedy_pretraining_{}_loss'.format(layer)):
    # print(x.get_shape())
    # print(x_recon.get_shape())
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=x_hat, logits=x_recon)
  cross_entropy = tf.reduce_mean(cross_entropy)

  # Will minimize and initialize these vars
  model_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ae{}'.format(layer))

  with tf.variable_scope('adam_optimizer_layer_{}'.format(layer), reuse=tf.AUTO_REUSE):
    optimizer = tf.train.AdamOptimizer(5e-3)
    train_step = optimizer.minimize(cross_entropy, var_list=model_variables)
  # model_variables = my_model.get_variables_list()
  optimizer_slots = [
      optimizer.get_slot(var, name)
      for name in optimizer.get_slot_names()
      for var in model_variables
  ]
  if isinstance(optimizer, tf.train.AdamOptimizer):
      optimizer_slots.extend([
          optimizer._beta1_power, optimizer._beta2_power
      ])
  all_variables = [
      *model_variables,
      *optimizer_slots,
  ]
  init_op = tf.variables_initializer(all_variables)

#   with tf.name_scope('accuracy'):
#     correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
#     correct_prediction = tf.cast(correct_prediction, tf.float32)
#   accuracy = tf.reduce_mean(correct_prediction)

#   graph_location = tempfile.mkdtemp()
#   print('Saving graph to: %s' % graph_location)
#   train_writer = tf.summary.FileWriter(graph_location)
#   train_writer.add_graph(tf.get_default_graph())

  with sess.as_default():
    init_op.run()
    # if layer == 1:
        # print(model_variables[0][500,:50].eval(session=sess))
    for i in range(iters * bool(pretrain_flag)):
      batch = mnist.train.next_batch(50)
      if i % 100 == 0:
        train_xentropy = cross_entropy.eval(feed_dict={x: batch[0]})
        # simil = tf.reduce_mean(tf.abs((input - x_hat)))
        print('step %d, train xentropy %g' % (i, train_xentropy))
      train_step.run(feed_dict={x: batch[0]})

    # if layer == 1:
        # print(model_variables[0][500,:50].eval(session=sess))
    print('test xentropy %g' % cross_entropy.eval(feed_dict={x: mnist.test.images}))

def finetune(layers, mnist, sess, iters):
  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # Build the graph for the deep net
  y_conv = deepnn(x, layers)

  with tf.variable_scope('supervised_loss', reuse=tf.AUTO_REUSE):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)

  # Will minimize all feed-forward vars
  var_list = []
  for i in range(1,layers+1):
      with tf.variable_scope('ae{}'.format(i), reuse=tf.AUTO_REUSE):
          with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
              W_enc = tf.get_variable('W')
              b_enc = tf.get_variable('b')
              var_list.extend([W_enc, b_enc])

  fc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fc')
  var_list.extend(fc_vars)

  with tf.name_scope('adam_optimizer_supervised'):
    optimizer = tf.train.AdamOptimizer(1e-4)
    train_step = optimizer.minimize(cross_entropy, var_list=var_list)
    # train_step = optimizer.minimize(cross_entropy, var_list)

  optimizer_slots = [
      optimizer.get_slot(var, name)
      for name in optimizer.get_slot_names()
      for var in var_list
  ]
  if isinstance(optimizer, tf.train.AdamOptimizer):
      optimizer_slots.extend([
          optimizer._beta1_power, optimizer._beta2_power
      ])
  all_variables = [
      *fc_vars,
      *optimizer_slots,
  ]
  init_op = tf.variables_initializer(all_variables)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  with sess.as_default():
    init_op.run()
    for i in range(iters):
      batch = mnist.train.next_batch(50)
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
        train_xentropy = cross_entropy.eval(feed_dict={x: batch[0], y_: batch[1]})
        print('step %d, training accuracy %g' % (i, train_accuracy), '\t|||\t', 'train xentropy %g' % (train_xentropy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    # print(var_list[0][500,:50].eval(session=sess))

    print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


def weight_variable(shape, collection=None):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  W = tf.Variable(initial)
  return W


def bias_variable(shape, layer=None):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  b = tf.Variable(initial)
  return b


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  for layer in range(1, FLAGS.num_layers+1):
      pretrain(layer, mnist, sess, FLAGS.unsup_iters, FLAGS.pretrain)
  finetune(FLAGS.num_layers, mnist, sess, FLAGS.sup_iters)

  sess.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  parser.add_argument('--num_layers', type=int,
                      default=3,
                      help='Number of pre-trained layers')
  parser.add_argument('--unsup_iters', type=int,
                      default=500,
                      help='Unsupervised training iters')
  parser.add_argument('--sup_iters', type=int,
                      default=500,
                      help='Supervised training iters')
  parser.add_argument('--pretrain', type=int,
                      default=1,
                      help='Pretrain - 1=True/0=False (default: 1)')
  FLAGS, unparsed = parser.parse_known_args()
  # import pdb; pdb.set_trace()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
