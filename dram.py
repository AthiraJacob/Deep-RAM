from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import pdb

from glimpse import GlimpseNet, EmissionNet, ContextNet
from utils import weight_variable, bias_variable, loglikelihood
from config import Config
from dataLoader import DataSet

# from tensorflow.examples.tutorials.mnist import input_data

CUDA_VISIBLE_DEVICES="0"

logging.getLogger().setLevel(logging.INFO)

rnn_cell = tf.contrib.rnn
seq2seq = tf.contrib.legacy_seq2seq

# mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
dataFold = '/home/athira/AngiogramProject/'
augment = True
field = 'area'

if field is 'area':
  cutoff_percentage = 60
  output_fold = '/home/athira/Codes/simpleCNN/OutputArea_class/'
elif field == 'dia':
  cutoff_percentage = 40
  output_fold = '/home/athira/Codes/simpleCNN/OutputDia_class/'
field_onehot = field + '_onehot'

data = DataSet(dataFold, preprocess = True, wavelet = False, histEq = False, tophat = False, 
    norm = '0to1', resize = False, smooth = False, context = False)

train, val, test = data.splitData()
nTrain = data.nTrain; nVal = data.nVal; nTest = data.nTest
train = data.createOnehot(train, percentage = cutoff_percentage)
val = data.createOnehot(val, percentage = cutoff_percentage)
test = data.createOnehot(test, percentage = cutoff_percentage)
nFeatures = data.nFeatures


config = Config()
n_steps = config.step

loc_mean_arr = []
sampled_loc_arr = []


def get_next_input(output,i):
  loc, loc_mean = em_net(output)
  gl_next = gl(loc)
  loc_mean_arr.append(loc_mean)
  sampled_loc_arr.append(loc)
  return gl_next

# placeholders
images_ph = tf.placeholder(tf.float32,
                           [None, config.original_size ,  config.original_size , 
                            config.num_channels])
labels_ph = tf.placeholder(tf.int64, [None])

# Build the aux nets.
with tf.variable_scope('glimpse_net'):
  gl = GlimpseNet(config, images_ph)
with tf.variable_scope('loc_net'):
  em_net = EmissionNet(config)
with tf.variable_scope('context_net'):
  cont_net = ContextNet(config, images_ph)

# number of examples
N = tf.shape(images_ph)[0]

# Recurrent  network.
# with tf.variable_scope('lstm_1'):
#   lstm_cell_1 = rnn_cell.LSTMCell(config.cell_size, state_is_tuple=False)
# with tf.variable_scope('lstm_2'):
#   lstm_cell_2 = rnn_cell.LSTMCell(config.cell_size, state_is_tuple=False)

def lstm_cell():
  return rnn_cell.LSTMCell(config.cell_size, state_is_tuple=True)
lstm_net = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(2)])

# c_init = tf.zeros([N, config.cell_size])
c_init = tf.truncated_normal(shape =  [N, config.cell_size], stddev=0.01)
r1_init = tf.zeros([N, config.cell_output_size])
r2_init = cont_net()
# r1 = r1_init; r2 = r2_init; c = c_init
init_state = tuple([rnn_cell.LSTMStateTuple(c_init,r1_init), rnn_cell.LSTMStateTuple(c_init,r2_init)])
# pdb.set_trace()
init_glimpse = get_next_input(r2_init, 0)
# pdb.set_trace()
inputs = [init_glimpse]
inputs.extend([0] * (config.num_glimpses))

loc_mean_arr = []
sampled_loc_arr = []

# pdb.set_trace()
outputs, state = seq2seq.rnn_decoder(inputs, init_state, lstm_net, loop_function=get_next_input)
cell1 = state[0]
r1 = cell1[1]

# Time independent baselines
with tf.variable_scope('baseline'):
  w_baseline = weight_variable((config.cell_output_size, 1))
  b_baseline = bias_variable((1,))
baselines = []
for t, output in enumerate(outputs[1:]):
  baseline_t = tf.nn.xw_plus_b(output, w_baseline, b_baseline)  #Shape: [batchsize x 1]
  baseline_t = tf.squeeze(baseline_t) #[batchsize]
  baselines.append(baseline_t) #List of timestep elements: [[batchsize], ...]
# pdb.set_trace()
baselines = tf.stack(baselines)  # [timesteps, batch_sz]  #tf.stack???
baselines = tf.transpose(baselines)  # [batch_sz, timesteps]

# Take the last step only.
output = r1  #[batchsize x lstm_output]
# Build classification network.
with tf.variable_scope('cls'):
  w_logit = weight_variable((config.cell_output_size, config.num_classes))
  b_logit = bias_variable((config.num_classes,))
logits = tf.nn.xw_plus_b(output, w_logit, b_logit)
softmax = tf.nn.softmax(logits)

# cross-entropy.
xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = labels_ph)
xent = tf.reduce_mean(xent)
pred_labels = tf.argmax(logits, 1)
# 0/1 reward.
reward = tf.cast(tf.equal(pred_labels, labels_ph), tf.float32)
rewards = tf.expand_dims(reward, 1)  # [batch_sz, 1]
rewards = tf.tile(rewards, (1, config.num_glimpses))  # [batch_sz, timesteps]
logll = loglikelihood(loc_mean_arr, sampled_loc_arr, config.loc_std)
advs = rewards - tf.stop_gradient(baselines)
logllratio = tf.reduce_mean(logll * advs)
reward = tf.reduce_mean(reward)

baselines_mse = tf.reduce_mean(tf.square((rewards - baselines)))
var_list = tf.trainable_variables()
# hybrid loss
loss = -logllratio + xent + baselines_mse  # `-` for minimize
grads = tf.gradients(loss, var_list)
grads, _ = tf.clip_by_global_norm(grads, config.max_grad_norm)

# learning rate
global_step = tf.get_variable(
    'global_step', [], initializer=tf.constant_initializer(0), trainable=False)
training_steps_per_epoch = nTrain // config.batch_size
starter_learning_rate = config.lr_start
# decay per training epoch
learning_rate = tf.train.exponential_decay(
    starter_learning_rate,
    global_step,
    training_steps_per_epoch,
    0.97,
    staircase=True)
learning_rate = tf.maximum(learning_rate, config.lr_min)
opt = tf.train.MomentumOptimizer(learning_rate, momentum = config.momentum, use_nesterov=True)
train_op = opt.apply_gradients(zip(grads, var_list), global_step=global_step)

datagen = ImageDataGenerator(
      samplewise_center=False, samplewise_std_normalization=False,
      featurewise_center=False, featurewise_std_normalization=False,
      rotation_range=40,
      width_shift_range=0.3,
      height_shift_range=0.3,
      horizontal_flip=True,
      vertical_flip=True,
      zoom_range=0,
      fill_mode = 'nearest',
      cval = 0)
print('---Augmenting images!---')

s = 128 
X = train['imgs'].reshape([-1,s,s,nFeatures])
valX = val['imgs'].reshape([-1,s,s,nFeatures])
testX = test['imgs'].reshape([-1,s,s,nFeatures])
saver = tf.train.Saver()

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())
  i = 0
  for images, labels in datagen.flow(X, train[field_onehot], batch_size=config.batch_size):
    # images, labels = mnist.train.next_batch(config.batch_size)
    # duplicate M times, see Eqn (2)
    # pdb.set_trace() 
    images = np.tile(images, [config.M, 1, 1, 1])
    labels = np.tile(labels, [config.M])
    em_net.sampling = True
    adv_val, baselines_mse_val, xent_val, logllratio_val, \
        reward_val, loss_val, lr_val, _ = sess.run(
            [advs, baselines_mse, xent, logllratio,
             reward, loss, learning_rate, train_op],
            feed_dict={
                images_ph: images,
                labels_ph: labels
            })
    # pdb.set_trace()
    if i and i % 5 == 0:
      logging.info('step {}: lr = {:3.6f}'.format(i, lr_val))
      logging.info(
          'step {}: reward = {:3.4f}\tloss = {:3.4f}\txent = {:3.4f}'.format(
              i, reward_val, loss_val, xent_val))
      logging.info('llratio = {:3.4f}\tbaselines_mse = {:3.4f}'.format(
          logllratio_val, baselines_mse_val))

    if i and i % training_steps_per_epoch == 0:
      # Evaluation
      for dataset in [val, test]:
        N = dataset['imgs'].shape[0]
        steps_per_epoch =  N// config.eval_batch_size
        correct_cnt = 0
        # num_samples = steps_per_epoch * config.batch_size
        em_net.sampling = True
        t = 0
        for test_step in range(steps_per_epoch):
          # images, labels = dataset.next_batch(config.batch_size)
          t = (test_step+1)*config.eval_batch_size 
          if t>N:
            t = N 
          images = dataset['imgs'][test_step*config.eval_batch_size : t]
          images = images.reshape([-1,s,s,nFeatures])
          labels = dataset[field_onehot][test_step*config.eval_batch_size : t]
          labels_bak = labels
          # Duplicate M times
          images = np.tile(images, [config.M,1, 1, 1])
          labels = np.tile(labels, [config.M])
          softmax_val = sess.run(softmax,
                                 feed_dict={images_ph: images,labels_ph: labels})
          # pdb.set_trace()
          softmax_val = np.reshape(softmax_val,[config.M, -1, config.num_classes])
          softmax_val = np.mean(softmax_val, 0)
          pred_labels_val = np.argmax(softmax_val, 1)
          pred_labels_val = pred_labels_val.flatten()
          correct_cnt += np.sum(pred_labels_val == labels_bak)
        acc = correct_cnt / N
        if dataset == val:
          logging.info('valid accuracy = {}'.format(acc))
        else:
          logging.info('test accuracy = {}'.format(acc))
        print('----------------')
    i = i+1
    saver.save(sess, 'Output/model', global_step=500)
    if i>=n_steps:
      break

