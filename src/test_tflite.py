from itertools import chain
from os import makedirs
from pathlib import Path
from typing import Callable, Dict, List

import cv2 as cv
import numpy as np
import tensorflow as tf
from cached_property import cached_property

from config import *
from nets import *
from utils.util import *

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model_path', 'tmp/tflite_model.tflite', """Path to the selected checkpoint.""")
tf.app.flags.DEFINE_string('image_dir', 'data/KITTI/testing', """Path to test data.""")
tf.app.flags.DEFINE_string('save_dir', 'test_result', """Path to detected_results.""")

# todo: use gray images for feature matching and plot on color images
class Tester():
    save_result: bool

    def __init__(self, model_path: str, image_dir: str = None, save_dir: str = None):
        self._tflite_interpreter = tf.lite.Interpreter(model_path=model_path)
        self._tflite_interpreter.allocate_tensors()
        self._image_dir = image_dir
        self._save_dir = save_dir
        self._input_details = self._tflite_interpreter.get_input_details()
        self._output_details = self._tflite_interpreter.get_output_details()
        self.image_size = self._input_details[0]['shape'][1:3]
        map(self._print_details, [self._input_details, self._output_details])
        self.mc = kitti_squeezeDetPlus_config()
        self.mc.LOAD_PRETRAINED_MODEL = False

    def test_images(self, save_result: bool = False) -> Dict:
        save_result = save_result
        image_path_list = list(Path(self._image_dir).glob('**/*g' or '**/*G'))
        assert(len(image_path_list) > 1), 'Cannot find input images from given folder: {}'.format(image_dir)
        det_results = {}
        for image_path in image_path_list:
            image = cv.resize(cv.imread(str(image_path)), tuple(self.image_size[::-1]))
            det_results[image_path.name] = self.test_one_image(image, save_result, Path(image_path).name)
        return det_results

    def test_one_image(self, image: str, save_result: bool = False, save_name = None) -> Dict:
        assert type(image) is np.ndarray, 'Image has to be a numpy array.'
        self._tflite_interpreter.set_tensor(self._input_details[0]['index'], np.expand_dims(image.astype(np.float32), 0))
        self._tflite_interpreter.invoke()
        probs, boxes, classes = [self._tflite_interpreter.get_tensor(output_detail['index']) for output_detail in self._output_details]
        filtered_results = _filter_prediction(boxes[0], probs[0], classes[0], self.mc)
        if save_result:
            self._save_result(image, *filtered_results, save_name)
        return dict(zip(['boxes', 'probs', 'classes'], filtered_results))

    @cached_property
    def _save_result(self) -> Callable:
        assert self._save_dir is not None, 'Please define saving dir.'
        makedirs(self._save_dir, exist_ok=True)

        def save_result(image, det_boxes, det_probs, det_classes, save_name, color=(255, 0, 255)):
            for bbox, prob, idx in zip(det_boxes, det_probs, det_classes):
                label = self.mc.CLASS_NAMES[idx]+': (%.2f)' % prob
                xmin, ymin, xmax, ymax = [int(b) for b in bbox_transform(bbox)]
                cv.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)
                cv.putText(image, label, (xmin, ymax), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv.imwrite(str(Path(self._save_dir).joinpath(save_name)), image)
        return save_result

    def _print_details(self, details):
        print("\n== {} details ==".format(details.name))
        for detail in details:
            print("shape:", detail['shape'])
            print("type:", detail['dtype'])


def _filter_prediction(boxes, probs, cls_idx, mc):
    """Filter bounding box predictions with probability threshold and
    non-maximum supression.

    Args:
      boxes: array of [cx, cy, w, h].
      probs: array of probabilities
      cls_idx: array of class indices
    Returns:
      final_boxes: array of filtered bounding boxes.
      final_probs: array of filtered probabilities
      final_cls_idx: array of filtered class indices
    """

    #print(len(probs))
    if mc.TOP_N_DETECTION < len(probs) and mc.TOP_N_DETECTION > 0:
      order = probs.argsort()[:-mc.TOP_N_DETECTION-1:-1]
      probs = probs[order]
      boxes = boxes[order]
      cls_idx = cls_idx[order]
    filtered_idx = np.nonzero(probs>mc.PROB_THRESH)[0]
    probs = probs[filtered_idx]
    boxes = boxes[filtered_idx]
    cls_idx = cls_idx[filtered_idx]

    final_boxes = []
    final_probs = []
    final_cls_idx = []
    '''
    keep = nms(boxes, probs, mc.NMS_THRESH)
    for i in range(len(keep)):
    if keep[i]:
        final_boxes.append(boxes[i])
        final_probs.append(probs[i)
        final_cls_idx.append(cls_idx[i])
    return final_boxes, final_probs, final_cls_idx
    
    '''
    for c in range(mc.CLASSES):
      idx_per_class = [i for i in range(len(probs)) if cls_idx[i] == c]
      keep = nms(boxes[idx_per_class], probs[idx_per_class], mc.NMS_THRESH)
      for i in range(len(keep)):
        if keep[i]:
          final_boxes.append(boxes[idx_per_class[i]])
          final_probs.append(probs[idx_per_class[i]])
          final_cls_idx.append(c)
    return final_boxes, final_probs, final_cls_idx
    


if __name__ == '__main__':
    Tester(FLAGS.model_path, FLAGS.image_dir, FLAGS.save_dir).test_images(True)
