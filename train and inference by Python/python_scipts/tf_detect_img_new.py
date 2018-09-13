import numpy as np
import os, sys
import tensorflow as tf
import cv2

MODEL_ROOT = "/home/raiden/Projects/tensorflow_objectdetection/"
sys.path.append(MODEL_ROOT)

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

#MODEL_PATH = MODEL_ROOT + "faster_rcnn_resnet101_kitti_2018_01_28"
PATH_TO_CKPT = MODEL_ROOT + 'output/frozen_inference_graph.pb'  # frozen model path
PATH_TO_LABELS = os.path.join(MODEL_ROOT, 'output', 'label_map.pbtxt')
NUM_CLASSES = 8

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

gpu_memory_fraction = 0.4
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False, allow_soft_placement=True)
config.gpu_options.allow_growth = False

def detect(image_path):
    with detection_graph.as_default():
      with tf.Session(graph=detection_graph, config=config) as sess:
          image = cv2.imread(image_path)
          image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          image_np_expanded = np.expand_dims(image_np, axis=0)
          image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
          boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
          scores = detection_graph.get_tensor_by_name('detection_scores:0')
          classes = detection_graph.get_tensor_by_name('detection_classes:0')
          num_detections = detection_graph.get_tensor_by_name('num_detections:0')
          (boxes, scores, classes, num_detections) = sess.run(
              [boxes, scores, classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
          vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=4)
          new_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
          cv2.imshow("test", new_img)
          cv2.waitKey(0)

if __name__ == '__main__':
    detect("/home/raiden/Projects/tensorflow_objectdetection/python_tools/L1_R_49VX90F_XX_515_7284.jpg")
