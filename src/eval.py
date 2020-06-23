# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Evaluation"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
from datetime import datetime
import os.path
import sys
import time
from utils.util import *

import numpy as np
from six.moves import xrange
import tensorflow as tf

from config import *
from dataset import pascal_voc, kitti
from utils.util import bbox_transform, Timer
from nets import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset', 'KITTI',
                           """Currently support PASCAL_VOC or KITTI dataset.""")
tf.app.flags.DEFINE_string('data_path', 'data/KITTI', """Root directory of data""")
tf.app.flags.DEFINE_string('image_set', 'val',
                           """Only used for VOC data."""
                           """Can be train, trainval, val, or test""")
tf.app.flags.DEFINE_string('eval_dir', 'tmp/logs/eval',
                            """Directory where to write event logs """)
tf.app.flags.DEFINE_string('checkpoint_path', 'tmp/logs/train',
                            """Path to the training checkpoint.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 10 * 1,
                             """How often to check if new cpt is saved.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                             """Whether to run eval only once.""")
tf.app.flags.DEFINE_string('net', 'squeezeDet+',
                           """Neural net architecture.""")
tf.app.flags.DEFINE_string('gpu', '0', """gpu id.""")
tf.app.flags.DEFINE_string('viz_eval', 'error', """viz_eval or not.""")

def _draw_box(im, box_list, label_list, color=(0,255,0), cdict=None, form='center'):
  assert form == 'center' or form == 'diagonal', \
      'bounding box format not accepted: {}.'.format(form)

  for bbox, label in zip(box_list, label_list):
    if form == 'center':
      bbox = bbox_transform(bbox)

    xmin, ymin, xmax, ymax = [int(b) for b in bbox]

    l = label.split(':')[0] # text before "CLASS: (PROB)"
    if cdict and l in cdict:
      c = cdict[l]
    else:
      c = color

    # draw box
    cv2.rectangle(im, (xmin, ymin), (xmax, ymax), c, 1)
    # draw label
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(im, label, (xmin, ymax), font, 0.5, c, 1)
  #cv2.imwrite('image.jpg', im)
  return im

def _analyse_det(gt_bboxes, det_bboxes):


  detected = [False]*len(gt_bboxes)
  num_objs = len(gt_bboxes)
  num_dets = 0
  num_repeated_error = 0
  num_loc_error = 0
  num_cls_error = 0
  num_bg_error = 0
  num_detected_obj = 0
  num_missed_error = 0
  num_correct = 0
  for i, det in enumerate(det_bboxes):
    if i < len(gt_bboxes):
      num_dets += 1
    ious = batch_iou(gt_bboxes[:, :4], det[0][:4]) # compute iou with each gt
    max_iou = np.max(ious) # find the gt which is closest
    gt_idx = np.argmax(ious)
    if max_iou > 0.1: 
      if gt_bboxes[gt_idx, 4] == det[1]: # if the class is the same with gt one 
        if max_iou >= 0.6:
          if i < len(gt_bboxes):
            if not detected[gt_idx]:
              num_correct += 1
              detected[gt_idx] = True
            else:
              num_repeated_error += 1
        else:
          if i < len(gt_bboxes):
            num_loc_error += 1
      else:
        if i < len(gt_bboxes): # if det class is wrong
          num_cls_error += 1
    else:
      if i < len(gt_bboxes):
        num_bg_error += 1 # detected bg as a checkbox
  for i, gt in enumerate(gt_bboxes):
    if not detected[i]:
      num_missed_error += 1
  num_detected_obj += sum(detected)
  return num_objs, num_dets, num_repeated_error, num_loc_error, num_cls_error, num_bg_error, num_detected_obj, num_missed_error, num_correct



def eval_once(
    saver, ckpt_path, summary_writer, imdb, model):
  gpu_config = tf.ConfigProto(allow_soft_placement=True)
  gpu_config.gpu_options.allow_growth = True
  with tf.Session(config=gpu_config) as sess:
    # Restores from checkpoint
    saver.restore(sess, ckpt_path)
    # Assuming model_checkpoint_path looks something like:
    #   /ckpt_dir/model.ckpt-0,
    # extract global_step from it.
    print(ckpt_path + '!!!\n')
    global_step = ckpt_path.split('/')[-1].split('-')[-1]
    num_images = len(imdb.image_idx)
    all_boxes = [[[] for _ in xrange(num_images)]for _ in xrange(imdb.num_classes)] # this is an empty list of list

    _t = {'im_detect': Timer(), 'im_read': Timer(), 'misc': Timer()}
    num_detection = 0.0
    gt_bboxes = imdb._rois
    perm_idx = imdb._perm_idx
    num_objs = 0
    num_dets = 0
    num_repeated_error = 0
    num_loc_error = 0
    num_cls_error = 0
    num_bg_error = 0
    num_detected_obj = 0
    num_missed_error = 0
    num_correct = 0

    for i in xrange(int(num_images/imdb.mc.BATCH_SIZE)):
      #_t['im_read'].tic()
      images, scales = imdb.read_image_batch(shuffle=True)
      #_t['im_read'].toc()
      #_t['im_detect'].tic()
      det_boxes, det_probs, det_class = sess.run([model.det_boxes, model.det_probs, model.det_class], feed_dict={model.image_input:images})
      #_t['im_detect'].toc()
      #_t['misc'].tic()
      for j in range(len(det_boxes)):  # batch
        det_bbox, score, det_cls = model.filter_prediction(det_boxes[j], det_probs[j], det_class[j])
        images[j] = _draw_box(images[j] + imdb.mc.IMG_MEANS , det_bbox, [model.mc.CLASS_NAMES[idx]+': (%.2f)'% prob \
            for idx, prob in zip(det_cls, score)], (0, 0, 255))

        num_detection += len(det_bbox)
        #for c, b, s in zip(det_cls, det_bbox, score):
        #  all_boxes[c][i].append(bbox_transform(b) + [s])
        gt_bbox = np.array(gt_bboxes[perm_idx[i*imdb.mc.BATCH_SIZE+j]])
        #gt_bboxes = np.array(gt_bboxes)
        gt_bbox[:, 0:4:2] *= scales[j][0]
        gt_bbox[:, 1::2] *= scales[j][1]
        if len(gt_bbox) >= 1:
          per_img_num_objs,per_img_num_dets, per_img_num_repeated_error,\
            per_img_num_loc_error, per_img_num_cls_error,\
              per_img_num_bg_error, per_img_num_detected_obj,\
                per_img_num_missed_error, per_img_num_correct = _analyse_det(gt_bbox, zip(det_bbox, det_cls))
          num_objs += per_img_num_objs
          num_dets += per_img_num_dets
          num_repeated_error += per_img_num_repeated_error
          num_loc_error += per_img_num_loc_error
          num_cls_error += per_img_num_cls_error
          num_bg_error += per_img_num_bg_error
          num_detected_obj += per_img_num_detected_obj
          num_missed_error += per_img_num_missed_error
          num_correct += per_img_num_correct
    viz_image_per_batch = bgr_to_rgb(images)
    viz_summary = sess.run(model.viz_op, feed_dict={model.image_to_show: viz_image_per_batch})
    summary_writer.add_summary(viz_summary, global_step)
    summary_writer.flush()
    print ('Detection Analysis:')
    print ('    Number of detections: {}'.format(num_dets))
    print ('    Number of objects: {}'.format(num_objs))
    print ('    Percentage of correct detections: {}'.format(
      num_correct/(num_dets+sys.float_info.epsilon)))
    print ('    Percentage of localization error: {}'.format(
      num_loc_error/(num_dets+sys.float_info.epsilon)))
    print ('    Percentage of classification error: {}'.format(
      num_cls_error/(num_dets+sys.float_info.epsilon)))
    print ('    Percentage of background error: {}'.format(
      num_bg_error/(num_dets+sys.float_info.epsilon)))
    print ('    Percentage of repeated detections: {}'.format(
      num_repeated_error/(num_dets+sys.float_info.epsilon)))
    print ('    Recall: {}'.format(
      num_detected_obj/num_objs))



def evaluate():
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
  with tf.Graph().as_default() as g:
    mc = kitti_squeezeDetPlus_config()
    #mc.BATCH_SIZE = 1 # TODO(bichen): allow batch size > 1
    mc.LOAD_PRETRAINED_MODEL = False
    model = SqueezeDetPlus(mc, state='val')
    imdb = kitti(FLAGS.image_set, FLAGS.data_path, mc)
    # add summary ops and placeholders
    saver = tf.train.Saver(model.model_params)
    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)
    ckpts = set() 
    while True:
      if FLAGS.run_once:
        # When run_once is true, checkpoint_path should point to the exact
        # checkpoint file.
        eval_once(
            saver, FLAGS.checkpoint_path, summary_writer, imdb, model)
        return
      else:
        # When run_once is false, checkpoint_path should point to the directory
        # that stores checkpoint files.
        from os.path import basename
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
        #base_name, max_idx = ckpt.model_checkpoint_path.split('-')
        #for idx in range(0, int(max_idx), )
        if ckpt and ckpt.model_checkpoint_path:
          if ckpt.model_checkpoint_path in ckpts:
            # Do not evaluate on the same checkpoint
            print ('Wait {:d}s for new checkpoints to be saved ... '.format(FLAGS.eval_interval_secs))
            time.sleep(FLAGS.eval_interval_secs)
          else:
            ckpts.add(ckpt.model_checkpoint_path)
            print ('Evaluating {}...'.format(ckpt.model_checkpoint_path))
            eval_once(saver, ckpt.model_checkpoint_path, summary_writer, imdb, model)
            
        else:
          print('No checkpoint file found')
          if not FLAGS.run_once:
            print ('Wait {:d}s for new checkpoints to be saved ... '
                      .format(FLAGS.eval_interval_secs))
            time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()
