class Config(object):
  win_size = 32
  bandwidth = win_size**2
  batch_size = 64
  eval_batch_size = 50
  loc_std = 0.22
  original_size = 128
  coarse_size = int(original_size/4)
  num_channels = 1
  depth = 2 
  sensor_size = win_size**2 * depth
  minRadius = 8 #????
  hg_size = hl_size = 1024
  g_size = 1024
  cell_output_size = 512  #???
  loc_dim = 2
  cell_size = 512
  cell_out_size = cell_size
  num_glimpses = 6
  num_classes = 2
  max_grad_norm = 5.


  step = 100000
  lr_start = 1e-2
  lr_min = 1e-4
  momentum = 0.9

  # Monte Carlo sampling
  M = 10