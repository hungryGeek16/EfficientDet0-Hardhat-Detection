import numpy as np
import argparse
import tensorflow as tf
import cv2
import time

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util

import sys

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile


if len(sys.argv) > 3:
    print('You have specified too many arguments')
if len(sys.argv) < 3:
    print('You need to specify the path to be listed')


def load_model(model_path):
    model = tf.saved_model.load(model_path)
    return model


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]
    
    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                                    output_dict['detection_masks'], output_dict['detection_boxes'],
                                    image.shape[0], image.shape[1])      
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
    return output_dict


def run_inference(model, category_index, cap):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))
    prev_frame_time = 0
    new_frame_time = 0
    while cap.isOpened():
        ret, image_np = cap.read()
        if not ret:
            break
        new_frame_time = time.time()
        # Actual detection.
        detections = run_inference_for_single_image(model, image_np)
        # Visualization of the results of a detection.
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = str(int(fps))
        dets = detections['detection_boxes']
        classes = detections['detection_classes']
        scores = detections['detection_scores']

        im_height = image_np.shape[0]
        im_width = image_np.shape[1]

        labels = ['Helmet', 'Head']

        for i in range(len(dets[0])):
            if (scores[i] > 0.2) and (len(classes) != 0):
                y1, x1, y2, x2 = dets[i,:]
                left, right, top, bottom = x1 * im_width, x2 * im_width, y1 * im_height, y2 * im_height
                cv2.rectangle(image_np, (int(left),int(top)), (int(right),int(bottom)), (36,255,12), 2)
                cv2.putText(image_np, str(labels[int(classes[i])-1])+":"+str(int(scores[i]*100)), (int(left), int(top-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                
        cv2.putText(image_np,"FPS:"+fps, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        img = cv2.resize(image_np,(640,480))
        out.write(img)
        cv2.imshow('object_detection',image_np)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            out.release()
            break


if __name__ == '__main__':

    detection_model = load_model(sys.argv[2]+"/saved_model")
    category_index = label_map_util.create_category_index_from_labelmap("label_map.pbtxt", use_display_name=True)

    cap = cv2.VideoCapture(sys.argv[1])
    run_inference(detection_model, category_index, cap)
