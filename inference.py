import matplotlib
import matplotlib.pyplot as plt
import os
import io
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
import glob
import cv2
import sys

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import tensorflow as tf
from models.research.object_detection.utils import config_util
from models.research.object_detection.builders import model_builder
import pathlib
import random
from scipy.spatial import distance
import time

color = {'Red':(255,0,0), 'Blue':(0,0,255), 'Green':(0,255,0),'Yellow':(255,255,0)}

#Color detector
def colors(image):
    means = {}
    img=image.reshape((image.shape[1]*image.shape[0],3)) #flatten images
    kmeans=KMeans(n_clusters=3) #Call kmeans
    kmeans.fit(img) #Centroid formation process
    centroid=kmeans.cluster_centers_ #Get centroids
    centroid_new=centroid.mean(axis=0) #Calculate mean
    for i in color.keys(): # Compare with colors present in pixel space
        dst = distance.euclidean(centroid_new, color[i])
        means[i] = dst
    return color[min(means, key=means.get)] #Select pixel with minimum distance to centroid


def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: the file path to the image

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


if len(sys.argv) > 3:
    print('You have specified too many arguments')
if len(sys.argv) < 3:
    print('You need to specify the path to be listed')

#recover our saved model
pipeline_config = sys.argv[2]+"/pipeline.config"
#generally you want to put the last ckpt from training in here
model_dir = str(sys.argv[2]+"/checkpoint/ckpt-0.index").replace('.index','')
configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']
detection_model = model_builder.build(
      model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(
      model=detection_model)
ckpt.restore(os.path.join(sys.argv[2]+"/checkpoint/ckpt-0.index").replace('.index',''))


def get_model_detection_function(model):
  """Get a tf.function for detection."""

  @tf.function
  def detect_fn(image):
    """Detect objects in image."""

    image, shapes = model.preprocess(image)
    prediction_dict = model.predict(image, shapes)
    detections = model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])

  return detect_fn



detect_fn = get_model_detection_function(detection_model)
#map labels for inference decoding

TEST_IMAGE_PATHS = glob.glob(sys.argv[1])
image_path = random.choice(TEST_IMAGE_PATHS)
image_np = load_image_into_numpy_array(image_path)

input_tensor = tf.convert_to_tensor(
    np.expand_dims(image_np, 0), dtype=tf.float32)
detections, predictions_dict, shapes = detect_fn(input_tensor)

dets = detections['detection_boxes'].numpy()
classes = detections['detection_classes'][0].numpy()
scores = detections['detection_scores'][0].numpy()

im_height = image_np.shape[0]
im_width = image_np.shape[1]

labels = ['Helmet', 'Head'] 

for i in range(len(dets[0])):
    if scores[i] > 0.2:
       y1, x1, y2, x2 = dets[0][i,:]
       left, right, top, bottom = x1 * im_width, x2 * im_width, y1 * im_height, y2 * im_height
       if classes[i] == 0.0:
           k = colors(image_np[int(top):int(bottom),int(left):int(right),:])
           cv2.rectangle(image_np, (int(left),int(top)), (int(right),int(bottom)), k, 2) #Call color detector if helmet is present in detection
           cv2.putText(image_np, str(labels[int(classes[i])])+":"+str(int(scores[i]*100)), (int(left), int(top-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
       else:
           cv2.rectangle(image_np, (int(left),int(top)), (int(right),int(bottom)), (36,255,12), 2)
           cv2.putText(image_np, str(labels[int(classes[i])])+":"+str(int(scores[i]*100)), (int(left), int(top-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
im = Image.fromarray(image_np)
im.save("image1.png") ##Saves image in directory
