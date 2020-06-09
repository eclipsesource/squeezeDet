# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Model configuration for pascal dataset"""

import numpy as np
from pathlib import Path

from .config import base_model_config
import cv2 as cv

def kitti_squeezeDetPlus_config():
  """Specify the parameters to tune below."""
  mc                       = base_model_config('KITTI')

  mc.IMAGE_WIDTH           = 1040#1632#608#1216
  mc.IMAGE_HEIGHT          = 1632#2432#400#816
  mc.BATCH_SIZE            = 3

  mc.WEIGHT_DECAY          = 0.0001
  mc.LEARNING_RATE         = 0.001
  mc.DECAY_STEPS           = 10000
  mc.MAX_GRAD_NORM         = 1.0
  mc.MOMENTUM              = 0.9
  mc.LR_DECAY_FACTOR       = 0.5

  mc.LOSS_COEF_BBOX        = 5.0
  mc.LOSS_COEF_CONF_POS    = 75.0
  mc.LOSS_COEF_CONF_NEG    = 100.0
  mc.LOSS_COEF_CLASS       = 1.0



  mc.PLOT_PROB_THRESH      = 0.8
  mc.NMS_THRESH            = .3 # bboxes are considered as overlapped if their iou is larger than this value
  mc.PROB_THRESH           = 0.8 # This will influece at the start but plays trival roles after some epochs, only visulization
  mc.TOP_N_DETECTION       = 100 # only influence visulization

  mc.DATA_AUGMENTATION     = False
  mc.DRIFT_X               = 10 #10#75#150  # The range to randomly shift the image widht and height
  mc.DRIFT_Y               = 5 #15#50#100
  mc.EXCLUDE_HARD_EXAMPLES = False

  mc.ANCHOR_BOX            = set_anchors(mc)
  mc.ANCHORS               = len(mc.ANCHOR_BOX)
  mc.ANCHOR_PER_GRID       = 2

  mc.CHANNEL_NUM = 3
  mc.IMAGE_COLOR = _get_image_color(mc.CHANNEL_NUM)
  mc.IMG_MEANS = _get_mean_image(mc.IMAGE_COLOR)

  return mc



def set_anchors(mc):
  H, W, B = 204, 130, 2#152, 102, 2#25, 38, 2#51, 76, 2
  anchor_shapes = np.reshape(
      [np.array(
          #[[  300.,  60.],[  300.,  100.]])] * H * W,_viz_prediction_result
          #[[  120.,  30.],[  120.,  50.]])] * H * W,
          #[[  140.,  30.],[140., 50.], [140., 80.]])] * H * W,
          [[  70.,  10.], [70., 40.]])] * H * W,
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


def _get_mean_image(image_color):
    image_dir = 'data/KITTI/training/image'
    image_path_list = list(Path(image_dir).glob('*G' or '*g'))
    assert(image_path_list), 'Cannot find images ends with *G under forlder {}'.format(image_dir)
    sum_image = np.zeros_like(cv.imread(str(image_path_list[0]), image_color)).astype(np.float32)
    for image_path in image_path_list:
        sum_image += cv.imread(str(image_path), image_color)
    # if len(sum_image) < 3:
    #    sum_image = np.expand_dims(sum_image, -1)
    return sum_image / float(len(image_path_list))


def _get_image_color(channel_num):
    assert channel_num == 1 or 3, 'Unknown channel number {}'.channel_num
    if channel_num == 1:
        return cv.IMREAD_GRAYSCALE
    return cv.IMREAD_COLOR