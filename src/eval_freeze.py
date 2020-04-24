from __future__ import absolute_import, division, print_function

import os.path
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List
from dataset import pascal_voc, kitti

import cv2
import numpy as np
import tensorflow as tf
from six.moves import xrange
from tensorflow import lite
from tensorflow.python.platform import gfile

from config import *
from nets import *
from utils.util import Timer, bbox_transform

tf.app.flags.DEFINE_string('checkpoint_path', 'tmp/logs/train/model.ckpt-20',
                            """Path to the selected checkpoint.""")
tf.app.flags.DEFINE_string('frozen_graph_name', 'freeze.pb',
                            """Path to the frozen graph.""")
tf.app.flags.DEFINE_string('tflite_name', 'mobile_model.tflite',
                            """Path to the tflite model.""")
tf.app.flags.DEFINE_string('data_path', 'data/KITTI', """Root directory of data""")

FLAGS = tf.app.flags.FLAGS

def main(argv=None):  # pylint: disable=unused-argument
    freezing_graph = tf.Graph() # initialize tf computation graph
    with freezing_graph.as_default(): # set this graph as default graph
        mc = kitti_squeezeDetPlus_config() # get model and config
        mc.LOAD_PRETRAINED_MODEL = False
        #mc.BATCH_SIZE=4
        model = SqueezeDetPlus(mc, state=val)
        #with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            # Restores from checkpoint
            #print(sess.run([v for v in tf.trainable_variables() if v.name == "fire2/squeeze1x1/biases:0"][0]))
            #print('\n\n')
            #saver.restore(sess, FLAGS.checkpoint_path)  

        #images = tf.placeholder(dtype=tf.float32, shape = tf.TensorShape([None, None, None, 1]), name = 'in_images')
        #model.image_input = tf.identity(model.image_input, name='in_images')# images       
        output_node_names = ["out_det_probs", "out_det_boxes", "out_det_class"]
        det_probs = tf.identity(model.det_probs, name = output_node_names[0])
        det_boxes = tf.identity(model.det_boxes, name = output_node_names[1])
        det_class = tf.identity(model.det_class, name = output_node_names[2])
        
    _freeze_graph(freezing_graph, FLAGS.checkpoint_path, FLAGS.frozen_graph_name, output_node_names, mc, model)
    _convert_pb_to_tflite(FLAGS.checkpoint_path, FLAGS.frozen_graph_name, FLAGS.tflite_name, output_node_names, model.image_input)

def _freeze_graph(graph, input_checkpoint, output_file_name: str, output_node_names: str, mc, model):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    imdb = kitti('val', FLAGS.data_path, mc)

    images, scales = imdb.read_image_batch(shuffle=False)
    image = images[0][np.newaxis, ::] # numpy array
    image = np.tile(image, [4, 1,1,1])
    #image = tf.convert_to_tensor(image)
    with tf.Session(graph=graph, config=config) as sess:

        sess.run(tf.global_variables_initializer())
        restorer = tf.train.Saver(tf.global_variables())
        restorer.restore(sess, input_checkpoint)
        det_boxes, det_probs, det_class = sess.run(
          [model.det_boxes, model.det_probs, model.det_class],
          feed_dict={model.ph_image_input:image})
        print(det_probs)

        # tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
        # for (i, tensor_name) in enumerate(tensor_name_list):
        #     if i < 50:
        #         print(tensor_name, '\n')

        #print(tf.global_variables())
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names)
        with gfile.GFile(str(Path(input_checkpoint).parents[2].joinpath(output_file_name)), 'wb') as f:
            f.write(output_graph_def.SerializeToString())

def _convert_pb_to_tflite(input_checkpoint:str, frozen_graph_name: str, tflite_file_name: str, output_node_names: List[str], image):
    converter = lite.TFLiteConverter.from_frozen_graph(str(Path(input_checkpoint).parents[2].joinpath(frozen_graph_name)),
                                                       ['image_input'],
                                                       output_node_names,
                                                       input_shapes={'image_input':image.shape})
    tflite_model = converter.convert()
    open(str(Path(input_checkpoint).parents[2].joinpath(tflite_file_name)), "wb").write(tflite_model)




if __name__ == '__main__':
  tf.app.run()
