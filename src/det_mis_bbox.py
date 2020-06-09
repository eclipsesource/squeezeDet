import itertools
from pathlib import Path

import cv2 as cv
import numpy as np
import tensorflow as tf
from cached_property import cached_property

from test_tflite import Tester
from utils.util import *

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('template_dir', 'data', """Path to template image, template image has to be name as template.""")
tf.app.flags.DEFINE_string('image_path', 'data/KITTI/testing/image_2/IMG_7178.jpg', """Path to template image, template image has to be name as template.""")
tf.app.flags.DEFINE_string('test_model_path', 'tmp/tflite_model.tflite', """Path to the selected checkpoint.""")
tf.app.flags.DEFINE_bool ('save_result', False, """Path to the selected checkpoint.""")



class MisDetector():
    def __init__(self, template_dir: str, model_path: str):
        self._template_path_list = list(Path(template_dir).glob('**/template*'))
        self._tester = Tester(model_path=model_path, image_dir=FLAGS.image_dir, save_dir=FLAGS.save_dir)

    def detect_missed(self, image_path, save_result):
        template = cv.resize(cv.imread(str(self._template_path_list[0])), tuple(self._tester.image_size[::-1]))
        image = cv.resize(cv.imread(str(image_path)), tuple(self._tester.image_size[::-1]))
        feature_map_result, H = map_feature(image, template, .75)
        det_results = self._tester.test_one_image(image, save_result)
        proj_boxes = [bbox_transform_inv(project_to_template(H, *det_box)) for det_box in det_results['boxes']]
        proj_boxes.sort(key=lambda x: x[1])
        plot_to_see(H, feature_map_result, *proj_boxes[1], (0,0,255))

        template_det_results = self._tester.test_one_image(template, save_result)
        temp_boxes = [list(temp_box) for temp_box in template_det_results['boxes']]
        temp_boxes.sort(key=lambda x: x[1])
        #eu_distance_list = [list(map(lambda x: np.linalg.norm(np.array(x)-np.array(proj_box)), temp_boxes)) for proj_box in proj_boxes]
        distance_list = np.array([fake_iou(np.array(temp_boxes), np.array(proj_box)) for proj_box in proj_boxes])
        #match_idx = [np.argmin(eu_distance) for eu_distance in eu_distance_list]
        match_idx = list(np.argmax(distance_list, 1))

        #cv.imwrite('./image.jpg', feature_map_result)
        #plot_to_check(H, det_result['boxes'], feature_map_result)
        return set(range(len(temp_boxes))) - set(match_idx)

    def _compare(self, det_boxes, tmp_boxes):
        pass


    @cached_property
    def _template(self):
        #return cv.imread(str(self._template_path_list[0]))
        pass


def map_feature(image, template, ratio_thresh):
    akaze = cv.AKAZE_create()
    key_points, descriptors = akaze.detectAndCompute(template, None)
    key_points2, descriptors2 = akaze.detectAndCompute(image, None)
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE_HAMMING)
    knn_matches = matcher.knnMatch(descriptors, descriptors2, 2)
    good_matches = [m for m, n in knn_matches if m.distance < ratio_thresh * n.distance]
    img_matches = np.empty((max(template.shape[0], image.shape[0]), template.shape[1]+image.shape[1], 3), dtype=np.uint8)
    cv.drawMatches(template, key_points, image, key_points2, good_matches, img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    obj = np.empty((len(good_matches), 2), dtype=np.float32)
    scene = np.empty((len(good_matches), 2), dtype=np.float32)
    for i in range(len(good_matches)):
        obj[i, 0] = key_points[good_matches[i].queryIdx].pt[0]
        obj[i, 1] = key_points[good_matches[i].queryIdx].pt[1]
        scene[i, 0] = key_points2[good_matches[i].trainIdx].pt[0]
        scene[i, 1] = key_points2[good_matches[i].trainIdx].pt[1]
    H, _ = cv.findHomography(scene, obj, cv.RANSAC)
    return cv.warpPerspective(src=image, M=H, dsize=template.shape[1::-1]), H


def project_to_template(H, x, y, w, h):
    x_min, y_min, x_max, y_max = [coord for coord in [x-w/2, y-h/2, x+w/2, y+h/2]]
    proj_x_y_min = H.dot(np.array([x_min,y_min,1]))
    proj_x_y_min = proj_x_y_min[:2]/proj_x_y_min[2]
    proj_x_y_max = H.dot(np.array([x_max,y_max,1]))
    proj_x_y_max = proj_x_y_max[:2]/proj_x_y_max[2]
    #proj_coord = list(np.concatenate((H.dot(np.array([x_min,y_min,1]))[:2], H.dot(np.array([x_max,y_max,1]))[:2])))
    #p1, p2 = [list(map(int, H.dot(np.array(p)))) for p in [[x_min, y_min, 1], [x_max, y_max, 1]]]
    # cv.rectangle(feature_map_result, tuple(p1[:2]), tuple(p2[:2]), (255, 0, 0), cv.FILLED)
    # cv.imwrite('./image.jpg', feature_map_result)
    return list(np.concatenate((proj_x_y_min, proj_x_y_max)))

def plot_to_see(H, feature_map_result, x_min, y_min, x_max, y_max, color):
    #x_min, y_min, x_max, y_max = [(int(coord)) for coord in [x-w/2, y-h/2, x+w/2, y+h/2]]

    p1 = list(map(int, map(round, [x_min, y_min])))
    p2 = list(map(int, map(round, [x_max, y_max])))
    #p1, p2 = [list(map(int, H.dot(np.array(p)))) for p in [[x_min, y_min, 1], [x_max, y_max, 1]]]
    cv.rectangle(feature_map_result, tuple(p1[:2]), tuple(p2[:2]), color, 2)
    cv.imwrite('./image.jpg', feature_map_result)
    #return H.dot(np.array([x,y,1]))[1:2]


def fake_iou(boxes, box):
    """Compute the Intersection-Over-Union of a batch of boxes with another
    box.

    Args:
        box1: 2D array of [cx, cy, width, height].
        box2: a single array of [cx, cy, width, height]
    Returns:
        ious: array of a float number in range [0, 1].
    """
    tb = np.maximum(
        np.minimum(boxes[:,1]+0.5*boxes[:,3], box[1]+0.5*box[3]) - \
        np.maximum(boxes[:,1]-0.5*boxes[:,3], box[1]-0.5*box[3]),
        0
    )
    inter = tb
    temp = boxes[:,3]
    return inter/temp


if __name__ == "__main__":
    print(MisDetector(FLAGS.template_dir, FLAGS.test_model_path).detect_missed(FLAGS.image_path, FLAGS.save_result))
