from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
from datetime import datetime
import os.path
import sys
import time
from utils.util import *
from typing import List
from tensorflow import lite


import numpy as np
from six.moves import xrange
import tensorflow as tf

from config import *
from dataset import pascal_voc, kitti
from utils.util import bbox_transform, Timer
from nets import *
from tensorflow.python.platform import gfile
from pathlib import Path



FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('checkpoint_path', 'tmp/logs/train/model.ckpt-220', """Path to the selected checkpoint.""")
tf.app.flags.DEFINE_string('frozen_graph_name', 'freeze.pb', """Path to the frozen graph.""")
tf.app.flags.DEFINE_string('tflite_name', 'mobile_model.tflite', """Path to the tflite model.""")
tf.app.flags.DEFINE_string('data_path', 'data/KITTI', """Root directory of data""")
tf.app.flags.DEFINE_string('gpu', '0', """gpu id.""")

def eval_once(
    saver, ckpt_path, imdb,
    model, output_node_names):
  gpu_config = tf.ConfigProto(allow_soft_placement=True)
  gpu_config.gpu_options.allow_growth = True

  with tf.Session(config=gpu_config) as sess:
    saver.restore(sess, ckpt_path)
    images, scales = imdb.read_image_batch(shuffle=True)
    det_boxes, det_probs, det_class = sess.run(
        [model.det_boxes, model.det_probs, model.det_class],
        feed_dict={model.ph_image_input:images})
    output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, output_node_names
                )  # convert to constants?
    frozen_graph_path = str(Path(ckpt_path).parents[2].joinpath(FLAGS.frozen_graph_name))
    with gfile.GFile(frozen_graph_path, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
    _convert_pb_to_tflite(frozen_graph_path=frozen_graph_path, input_shape= model.ph_image_input.shape)

def evaluate():
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
  with tf.Graph().as_default() as g:
    mc = kitti_squeezeDetPlus_config()
    mc.BATCH_SIZE = 1 # TODO(bichen): allow batch size > 1
    mc.LOAD_PRETRAINED_MODEL = False
    model = SqueezeDetPlus(mc, state='val')
    output_node_names = ["out_det_probs", "out_det_boxes", "out_det_class"]
    det_probs = tf.identity(model.det_probs, name=output_node_names[0])
    det_boxes = tf.identity(model.det_boxes, name=output_node_names[1])
    det_class = tf.identity(model.det_class, name=output_node_names[2])
    imdb = kitti('val', FLAGS.data_path, mc)

    saver = tf.train.Saver(model.model_params)
    eval_once(
        saver, FLAGS.checkpoint_path, imdb, model, output_node_names)

def _convert_pb_to_tflite(
    frozen_graph_path: str,
    input_shape: List[int],
    input_node_names: List[str] = ['image_input'],
    output_node_names: List[str] = ["out_det_probs", "out_det_boxes", "out_det_class"]
        ):
    converter = lite.TFLiteConverter.from_frozen_graph(
        frozen_graph_path, input_node_names, output_node_names, input_shapes={'image_input':input_shape}
            )
    tflite_model = converter.convert()
    open(str(Path(frozen_graph_path).parent.joinpath('tflite_model.tflite')), "wb").write(tflite_model)
    #converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    converter.post_training_quantize=True
    tflite_quant_model = converter.convert()
    open(str(Path(frozen_graph_path).parent.joinpath('tflite_quant_model.tflite')), "wb").write(tflite_quant_model)

def main(argv=None):  # pylint: disable=unused-argument
  evaluate()


if __name__ == '__main__':
  tf.app.run()
