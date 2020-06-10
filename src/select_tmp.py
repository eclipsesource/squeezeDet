import tensorflow as tf
from nptyping import NDArray
import cv2 as cv
from pathlib import Path
import numpy as np

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('template_dir', 'data', """Path to template image, template image has to be name as template.""")
tf.app.flags.DEFINE_string('image_path', 'data/KITTI/testing/image_2/IMG_7178.jpg', """Path to template image, template image has to be name as template.""")


def select_tmp(input_image: NDArray, tmp_dir: str):
    resize_input_image = cv.resize(input_image, (500, 600))
    template_path_list = list(Path(tmp_dir).glob('**/template*'))
    similarity_list = [compute_similarity(resize_input_image, template_path) for template_path in template_path_list]
    return template_path_list[similarity_list.index(min(similarity_list))]


def compute_similarity(image: NDArray, template_path: str):
    template_img = cv.resize(cv.imread(str(template_path)), (500, 600))
    feature_map_result = map_feature(image, template_img, .75)
    cv.imwrite('./unwarpped'+template_path.name,  feature_map_result)

    template_img = cv.medianBlur(template_img, 11)
    feature_map_result = cv.medianBlur(feature_map_result, 11)
    blured_img = cv.GaussianBlur(template_img, (201, 201), 0)
    return np.mean((image - feature_map_result)**2)


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
    return cv.warpPerspective(src=image, M=H, dsize=template.shape[1::-1])


if __name__ == "__main__":
    input_image = cv.imread(FLAGS.image_path)
    print(select_tmp(input_image=input_image, tmp_dir=FLAGS.template_dir))