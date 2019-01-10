# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
from Lip_Reading_Using_CNN_and_LSTM.object_detection.utils import label_map_util
from Lip_Reading_Using_CNN_and_LSTM.object_detection.utils import visualization_utils as vis_util

def unnormalize_coordinates(xmin, ymin, xmax, ymax, img):
    num_rows, num_cols = img.shape[:2]
    xmax = (xmax * (num_cols - 1.))-6
    xmin = (xmin*(num_cols - 1.))+7
    ymax = (ymax * (num_rows - 1.))-3
    ymin = (ymin*(num_rows - 1.))+6
    return xmin,ymin,xmax,ymax

def normalizedMouth(mouth):

    image_min = mouth[mouth > 0].min()
    image_max = mouth[mouth > 0].max()

    mouth = (mouth - image_min) / (float(image_max - image_min))

    return mouth


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'C:/Users/Jaden/PycharmProjects/Thesis/Lip_Reading_Using_CNN_and_LSTM/inference_graph'
VIDEO_NAME = 'D:/Datasets/s1/bbaf2n.mpg'

counter=0
# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','label.pbtxt')

# Path to video
PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 6

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Open video file
video = cv2.VideoCapture(PATH_TO_VIDEO)

while(video.isOpened()):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()

    tryput = frame

    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    # Draw the results of the detection (aka 'visulaize the results')
    image, area=vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.80)

    box =unnormalize_coordinates(area[0], area[1], area[2], area[3],image)

    print(np.shape(image))

    cropped=tryput[int(box[1]):int(box[3]), int(box[0]):int(box[2])]

    normMouth = normalizedMouth(cropped)

    print(len(cropped))

    cv2.imshow("mouth", cropped)
    cv2.waitKey(1)
    # All the results have been drawn on the frame, so it's time to display it.
    counter+=1

# Clean up
video.release()
cv2.destroyAllWindows()