import numpy as np
import tensorflow as tf

from config import kitti_squeezeDetPlus_config
from dataset import kitti

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_path', 'data/KITTI', """Root directory of data""")
tf.app.flags.DEFINE_string('graph_file', 'tmp/freeze.pb', """Root directory of data""")
tf.app.flags.DEFINE_string('image_set', 'val', """Can be train, trainval, val, or test""")


def eval_frozen_graph(image):
    image = image[np.newaxis, ::] # numpy array
    image = tf.convert_to_tensor(image)
    import_graph("tmp/freeze.pb")

    with tf.Graph().as_default():
        # # To print the list of operations in the graph for debugging purposes
        for op in tf.get_default_graph().get_operations():
            print(op.name)
        config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            [p1] = sess.run(["pool1/MaxPool"])
           #[out_det_boxes, out_det_probs, out_det_class] = sess.run(["out_det_boxes", "out_det_probs", "out_det_class"])
           #import pdb; pdb.set_trace()







if __name__ == '__main__':
    with tf.gfile.GFile("tmp/freeze.pb", 'rb') as f:
        mc = kitti_squeezeDetPlus_config()
        mc.BATCH_SIZE = 1 # TODO(bichen): allow batch size > 1
        imdb = kitti(FLAGS.image_set, FLAGS.data_path, mc)
        images, scales = imdb.read_image_batch(shuffle=False)
        #image = tf.convert_to_tensor(image)
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")
        #print(tf.global_variables())
         # # To print the list of operations in the graph for debugging purposes
        # for op in tf.get_default_graph().get_operations():
        #     print(op.name)
        config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True
        tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
        for tensor_name in tensor_name_list:
            print(tensor_name, '\n')
        with tf.Session(config=config) as sess:
            #[p1] = sess.run(["pool1/MaxPool"])
            #images, scales = imdb.read_image_batch(shuffle=False)
            #print(tf.global_variables())
            # out = sess.run(
            # ["out_det_class:0"],
            # feed_dict={'image_input:0':image})
            # print(out[0].shape)


            det_probs, det_boxes, det_class = sess.run(
            ["out_det_probs:0", "out_det_boxes:0", "out_det_class:0"],
            feed_dict={'image_input:0':images})
            print(det_probs.shape)
            print(det_boxes.shape)
            print(det_class.shape)
            #[out_det_boxes, out_det_probs, out_det_class] = sess.run(["out_det_boxes", "out_det_probs", "out_det_class"])
        #import pdb; pdb.set_trace()

        

        #eval_frozen_graph(*images)
