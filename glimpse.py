from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pdb

from utils import weight_variable, bias_variable, conv2d


class GlimpseNet(object):
  """Glimpse network.
  Take glimpse location input and output features for RNN.
  """

  def __init__(self, config, images_ph):
    self.original_size = config.original_size
    self.num_channels = config.num_channels
    self.sensor_size = config.sensor_size
    self.win_size = config.win_size
    self.minRadius = config.minRadius
    self.depth = config.depth

    self.hg_size = config.hg_size
    self.hl_size = config.hl_size
    self.g_size = config.g_size
    self.loc_dim = config.loc_dim

    self.images_ph = images_ph

    self.init_weights()

  def init_weights(self):
    """ Initialize all the trainable weights."""
    #G_image
    conv_filt = (64,64,128)
    self.w_c0 = weight_variable((5, 5, self.depth, conv_filt[0]))
    self.b_c0 = bias_variable((conv_filt[0],))
    self.w_c1 = weight_variable((3, 3, conv_filt[0], conv_filt[1]))
    self.b_c1 = bias_variable((conv_filt[1],))
    self.w_c2 = weight_variable((3, 3, conv_filt[1], conv_filt[2]))
    self.b_c2 = bias_variable((conv_filt[2],))
    self.flat_dim = (self.win_size - 8)**2 * conv_filt[2]
    self.w_x0 = weight_variable((self.flat_dim, self.hg_size))
    self.b_x0 = bias_variable((self.hg_size,))
    #G_loc
    self.w_l0 = weight_variable((self.loc_dim, self.hl_size))
    self.b_l0 = bias_variable((self.hl_size,))


  def get_glimpse(self, loc):
    """Take glimpse on the original images."""
    imgs = tf.reshape(self.images_ph, [
        tf.shape(self.images_ph)[0], self.original_size, self.original_size,self.num_channels])
    # pdb.set_trace()
    glimpse_imgs_1 = tf.image.extract_glimpse(imgs,[self.win_size, self.win_size], loc)
    glimpse_imgs_2 = tf.image.extract_glimpse(imgs,[self.win_size*2, self.win_size*2], loc)
    glimpse_imgs_2 = tf.image.resize_images(glimpse_imgs_2, [self.win_size, self.win_size])

    glimpse_imgs = tf.concat([glimpse_imgs_1, glimpse_imgs_2], 3)
    # print('Shape of glimpse:')
    # print(glimpse_imgs.get_shape())

    return glimpse_imgs

  def __call__(self, loc):
    glimpse_input = self.get_glimpse(loc)

    conv1 = conv2d(glimpse_input, self.w_c0) + self.b_c0
    conv2 = conv2d(conv1, self.w_c1) + self.b_c1
    conv3 = conv2d(conv2, self.w_c2) + self.b_c2
    flat = tf.reshape(conv3, [-1,self.flat_dim])

    x = tf.nn.relu(tf.nn.xw_plus_b(flat, self.w_x0, self.b_x0))

    l = tf.nn.relu(tf.nn.xw_plus_b(loc, self.w_l0, self.b_l0))

    g = tf.multiply(x , l)
    # print('Shape of g:')
    # print(g.get_shape())

    return g


class EmissionNet(object):
  """Location network.
  Take output from other network and produce and sample the next location.
  """

  def __init__(self, config):
    self.loc_dim = config.loc_dim
    self.input_dim = config.cell_output_size
    self.loc_std = config.loc_std
    self._sampling = True

    self.init_weights()

  def init_weights(self):
    self.w = weight_variable((self.input_dim, self.loc_dim))
    self.b = bias_variable((self.loc_dim,))

  def __call__(self, input):
    # pdb.set_trace()
    mean = tf.clip_by_value(tf.nn.xw_plus_b(input, self.w, self.b), -1., 1.)
    mean = tf.stop_gradient(mean)
    if self._sampling:
      loc = mean + tf.random_normal(
          (tf.shape(input)[0], self.loc_dim), stddev=self.loc_std)
      loc = tf.clip_by_value(loc, -1., 1.)
    else:
      loc = mean
    loc = tf.stop_gradient(loc)
    return loc, mean

  @property
  def sampling(self):
    return self._sampling

  @sampling.setter
  def sampling(self, sampling):
    self._sampling = sampling


class ContextNet(object):
  """Context network.
  Take image input, downsample it and output initial state for RNN.
  """

  def __init__(self, config, images_ph):
    self.original_size = config.original_size
    self.coarse_size = config.coarse_size
    self.num_channels = config.num_channels
    self.win_size = config.win_size
    self.minRadius = config.minRadius
    self.depth = config.depth
    self.output_dim = config.cell_output_size

    self.images_ph = images_ph
    self.images_coarse = tf.image.resize_images(images_ph,[self.coarse_size, self.coarse_size] )

    self.init_weights()

  def init_weights(self):
    """ Initialize all the trainable weights."""
    #G_image
    conv_filt = (64,64,128)
    self.w_c0 = weight_variable((5, 5, self.num_channels, conv_filt[0]))
    self.b_c0 = bias_variable((conv_filt[0],))
    self.w_c1 = weight_variable((3, 3, conv_filt[0], conv_filt[1]))
    self.b_c1 = bias_variable((conv_filt[1],))
    self.w_c2 = weight_variable((3, 3, conv_filt[1], conv_filt[2]))
    self.b_c2 = bias_variable((conv_filt[2],))
    self.flat_dim = (self.coarse_size - 8)**2 * conv_filt[2]
    self.w_x0 = weight_variable((self.flat_dim, self.output_dim))
    self.b_x0 = bias_variable((self.output_dim,))

  def __call__(self):

    conv1 = conv2d(self.images_coarse, self.w_c0) + self.b_c0
    conv2 = conv2d(conv1, self.w_c1) + self.b_c1
    conv3 = conv2d(conv2, self.w_c2) + self.b_c2
    # pdb.set_trace()
    flat = tf.reshape(conv3, [-1,self.flat_dim])
    c = tf.nn.relu(tf.nn.xw_plus_b(flat, self.w_x0, self.b_x0))

    # print('Shape of c:')
    # print(tf.shape(c))

    return c