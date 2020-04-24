# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Model configuration for pascal dataset"""

import numpy as np

from .config import base_model_config

def kitti_squeezeDetPlus_config():
  """Specify the parameters to tune below."""
  mc                       = base_model_config('KITTI')

  mc.IMAGE_WIDTH           = 816#608#1216
  mc.IMAGE_HEIGHT          = 1216#400#816
  mc.BATCH_SIZE            = 2

  mc.WEIGHT_DECAY          = 0.0001
  mc.LEARNING_RATE         = 0.005
  mc.DECAY_STEPS           = 10000
  mc.MAX_GRAD_NORM         = 1.0
  mc.MOMENTUM              = 0.9
  mc.LR_DECAY_FACTOR       = 0.5

  mc.LOSS_COEF_BBOX        = 5.0
  mc.LOSS_COEF_CONF_POS    = 75.0
  mc.LOSS_COEF_CONF_NEG    = 100.0
  mc.LOSS_COEF_CLASS       = 1.0



  mc.PLOT_PROB_THRESH      = 0.2
  mc.NMS_THRESH            = 0.2 # bboxes are considered as overlapped if their iou is larger than this value
  mc.PROB_THRESH           = 0.05 # 
  mc.TOP_N_DETECTION       = 20

  mc.DATA_AUGMENTATION     = False
  mc.DRIFT_X               = 150#75#150
  mc.DRIFT_Y               = 100#50#100
  mc.EXCLUDE_HARD_EXAMPLES = False

  mc.ANCHOR_BOX            = set_anchors(mc)
  mc.ANCHORS               = len(mc.ANCHOR_BOX)
  mc.ANCHOR_PER_GRID       = 2

  return mc

def set_anchors(mc):
  H, W, B = 76, 51, 2#25, 38, 2#51, 76, 2
  anchor_shapes = np.reshape(
      [np.array(
          #[[  300.,  60.],[  300.,  100.]])] * H * W,
          #[[  120.,  30.],[  120.,  50.]])] * H * W,
          [[  80.,  40.],[  80.,  60.]])] * H * W,
      (H, W, B, 2)
  ) # 51,76,1,2
  center_x = np.reshape(
      np.transpose(
          np.reshape(
              np.array([np.arange(1, W+1)*float(mc.IMAGE_WIDTH)/(W+1)]*H*B), 
              (B, H, W)
          ),
          (1, 2, 0)
      ),
      (H, W, B, 1)
  )
  center_y = np.reshape(
      np.transpose(
          np.reshape(
              np.array([np.arange(1, H+1)*float(mc.IMAGE_HEIGHT)/(H+1)]*W*B),
              (B, W, H)
          ),
          (2, 1, 0)
      ),
      (H, W, B, 1)
  )
  anchors = np.reshape(
      np.concatenate((center_x, center_y, anchor_shapes), axis=3),
      (-1, 4)
  )
  return anchors
