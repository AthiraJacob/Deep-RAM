from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

distributions = tf.contrib.distributions


# def weight_variable(shape):
#   initial = tf.truncated_normal(shape, stddev=0.01)
#   return tf.Variable(initial)

def weight_variable(shape):
  #With xavier initialization
  if len(shape) == 4:
    stddev = np.sqrt(2/(shape[0]*shape[1]*shape[2]))
  elif len(shape) == 2:
    stddev = np.sqrt(2/shape[0])
  initial = tf.truncated_normal(shape, stddev=stddev)
  return tf.Variable(initial)

def conv2d(inputs, filt):
  return tf.nn.conv2d(inputs, filt, strides = [1,1,1,1], padding = 'VALID' )

def bias_variable(shape):
  initial = tf.constant(0.01, shape=shape)
  return tf.Variable(initial)


def loglikelihood(mean_arr, sampled_arr, sigma):
  mu = tf.stack(mean_arr)  # mu = [timesteps, batch_sz, loc_dim]
  sampled = tf.stack(sampled_arr)  # same shape as mu
  gaussian = distributions.Normal(mu, sigma)
  logll = gaussian.log_pdf(sampled)  # [timesteps, batch_sz, loc_dim]
  logll = tf.reduce_sum(logll, 2)
  logll = tf.transpose(logll)  # [batch_sz, timesteps]
  return logll