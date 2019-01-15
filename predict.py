#%% [markdown]
# ## Rboxnet - Test prediction
#
#%%

import os
import json
import random
import time
import cv2
import datetime
import numpy as np
import tensorflow as tf
import rboxnet
import rboxnet.dataset
import imgaug as ia
from imgaug import augmenters as iaa
from rboxnet import inference, config, model, drawing
from rboxnet.base import unmold_detections, top_line_to_vertices, vertices_fliplr
from rboxnet.eval import calc_ious, plot_average_recall, compute_match
from evalcoco import evalcoco

import rboxtrain

#%%
# define global parameters
#################################################

# Root directory of the project
ROOT_DIR = os.getcwd()
# ROOT_DIR = "/home/gustavoneves/sources/rboxnet/"

# Trained model directory
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to Shapes trained weights
# RBOXNET_MODEL_PATH = "assets/weights/rboxnet_gemini_resnet50_deltas_0160.h5"
RBOXNET_MODEL_PATH = "assets/weights/rboxnet_gemini_resnet50_deltas_last.h5"
# RBOXNET_MODEL_PATH = "assets/weights/rboxnet_gemini_resnet50_rotdim_last.h5"
# RBOXNET_MODEL_PATH = "assets/weights/rboxnet_gemini_resnet101_deltas_0160.h5"
# RBOXNET_MODEL_PATH = "assets/weights/rboxnet_gemini_resnet101_deltas_last.h5"
# RBOXNET_MODEL_PATH = "assets/weights/rboxnet_gemini_resnet101_rotdim_last.h5"
# RBOXNET_MODEL_PATH = "assets/weights/faster_rcnn_gemini_resnet50_last.h5"
# RBOXNET_MODEL_PATH = "assets/weights/faster_rcnn_gemini_resnet101_last.h5"
# RBOXNET_MODEL_PATH = "assets/weights/20181221/rboxnet_gemini_resnet101_deltas_0160.h5"
# RBOXNET_MODEL_PATH = "assets/weights/20181221/rboxnet_gemini_resnet101_rotdim_0155.h5"
# RBOXNET_MODEL_PATH = "assets/weights/rboxnet_gemini_resnet50_verts_last.h5"
# RBOXNET_MODEL_PATH = "assets/weights/rboxnet_gemini_resnet101_verts_last.h5"
# RBOXNET_MODEL_PATH = "assets/weights/20181221/rboxnet_gemini_resnet50_deltas_0160.h5"
# RBOXNET_MODEL_PATH = "assets/weights/20181221/rboxnet_gemini_resnet50_rotdim_0160.h5"

RBOXNET_MODEL_PATH = os.path.join(ROOT_DIR, RBOXNET_MODEL_PATH)

# Path to configuration file
# CONFIG_PATH = os.path.join(ROOT_DIR, "cfg/gemini.json")
CONFIG_PATH = os.path.join(ROOT_DIR, "cfg/gemini_deltas.json")
# CONFIG_PATH = os.path.join(ROOT_DIR, "cfg/gemini_rotdim.json")
# CONFIG_PATH = os.path.join(ROOT_DIR, "cfg/gemini_verts.json")


# ## Create inference configuration.
class InferenceConfig(rboxtrain.TrainingConfig):
  GPU_COUNT = 1
  IMAGES_PER_GPU = 1
  DETECTION_MIN_CONFIDENCE = 0.7
  BACKBONE = "resnet50"
  # BACKBONE = "resnet101"
  IMAGE_MAX_DIM = 448



config = InferenceConfig()

# Filter labels
# FILTER_LABELS = ["ssiv_bahia", "jequitaia", "balsa"]
FILTER_LABELS = ["ssiv_bahia"]
# FILTER_LABELS = ["jequitaia"]
# FILTER_LABELS = ["balsa"]

# Dataset shuffle
SHUFFLE = False

# do not draw rotate bounding-boxes
DISABLE_ROTATED_BOXES = False

# do not draw rotate boxes
DISABLE_BOXES = False

# number of classes
NB_CLASS = 3

# class labels
labels = ["ssiv_bahia", "jequitaia", "balsa"]

# total images to be process
MAX_IMAGES = -1

# show detection
SHOW_DETECTION = True

# enable output
VERBOSE = True

# image augmented
IMAGE_AUG = False

# save results
SAVE_RESULTS = False

# save image
SAVE_IMAGE = True

# noise shift
NOISE_SHIFT = 0.03

#%%
# define functions
#################################################
# extract annotations
def extract_annotations(annotations_info):
  annotations = []
  for ann in annotations_info:
    annotations.append({
        "id": ann['id'],
        "bbox": ann['bbox'],
        "rbox": ann['segmentation']
    })
  return annotations


# extract detections
def extract_detections(class_ids, scores, boxes, rotated_boxes):
  detections = []
  for i, cls_id in enumerate(class_ids):
    detections.append({
        "id": cls_id,
        "score": scores[i].tolist(),
        "bbox": boxes[i].tolist(),
        "rbox": rotated_boxes[i].tolist() if not rotated_boxes is None else []
    })
  return detections


## Load dataset#
with open(CONFIG_PATH) as f:
  cfg = json.load(f)
  anns_path = os.path.join(ROOT_DIR, cfg['annotations']['test'])
  dataset = rboxnet.dataset.gemini_dataset(
      anns_path, shuffle=SHUFFLE, labels=FILTER_LABELS)
  if not DISABLE_ROTATED_BOXES:
    config.regressor = cfg['regressor']

DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0
with tf.device(DEVICE):
  net = inference.Inference(config)

if config.regressor:
    config.NAME = "{0}_{1}_{2}".format(config.NAME, config.BACKBONE,
                                       config.regressor)

def load_weights(model_path=RBOXNET_MODEL_PATH):
  print("Model: ", model_path)
  # Load trained weights
  net.load_weights(model_path, by_name=True)
  net.keras_model.summary()

def crop_roi(img, roi):
  x1, y1, x2, y2 = roi
  return img[y1:y2, x1:x2]

def predict(noise=None):

  config.display()

  # if not noise is None:
  #   aug_pipe = iaa.Sequential([iaa.AdditiveGaussianNoise(scale=(noise+NOISE_SHIFT) * 255)])
  # elif IMAGE_AUG:
  #   aug_pipe = iaa.Sequential([
  #     iaa.SomeOf((0, 5), [
  #         iaa.Multiply((0.5, 1.5)), \
  #         iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))
  #       ], random_order=True)
  #   ], random_order=True)


  print("Configuration Name: ", config.NAME)
  print("Images: {0}\nClasses: {1}".format(
      len(dataset.image_ids), dataset.class_names))

  print("Image Augmentation: ", IMAGE_AUG)
  print("Save result: ", SAVE_RESULTS)
  print("Show detection: ", SHOW_DETECTION)
  print("Labels: ", FILTER_LABELS)
  print("Dataset size: ", len(dataset.image_ids))
  print("Total images: ", len(dataset.image_ids[:MAX_IMAGES]))
  print("Start predictions")

  all_ious = []
  results = []

  total_images = len(dataset.image_ids[:MAX_IMAGES])
  count = 0
  for image_id in dataset.image_ids[0:total_images]:
  # ssiv bahia
  # for image_id in dataset.image_ids[2055:2056]:
  # for image_id in dataset.image_ids[2500:2501]:
  # for image_id in dataset.image_ids[2786:2787]:
  # for image_id in dataset.image_ids[2907:2908]:
  # jequitaia
  # for image_id in dataset.image_ids[546:547]:
  # for image_id in dataset.image_ids[931:932]:
  # for image_id in dataset.image_ids[2030:2031]:
  # for image_id in dataset.image_ids[2210:2211]:
  # balsa
  # for image_id in dataset.image_ids[36:37]:
  # for image_id in dataset.image_ids[290:291]:
  # for image_id in dataset.image_ids[951:952]:
  # for image_id in dataset.image_ids[2003:2004]:
    image = dataset.load_image(image_id)

    # if IMAGE_AUG or not noise is None and noise > 0.0:
    #   # _, mask = cv2.threshold(image, 0, 1, cv2.THRESH_BINARY)
    #   image = aug_pipe.augment_image(image)
    #   # image = image * mask

    start_time = time.time()
    detections = net.detect([image])[0]
    elapsed_time = time.time() - start_time
    fps = 1.0 / elapsed_time

    class_ids, scores, boxes, rotated_boxes = \
        detections['class_ids'], detections['scores'], detections['boxes'], detections['rotated_boxes']
    class_ids = [dataset.class_info[cls_id]['id'] for cls_id in class_ids]

    # flip vertices
    boxes = vertices_fliplr(boxes)
    rotated_boxes = vertices_fliplr(rotated_boxes) if not rotated_boxes is None else None

    # load image info
    image_info = dataset.image_info[image_id]
    img_path = image_info['path']

    # extract detections
    detections = extract_detections(class_ids, scores, boxes, rotated_boxes)

    # extract annotations
    annotations = extract_annotations(image_info["annotations"])

    image_w, image_h, image_d = image.shape
    dt_matches, gt_matches, scores, overlaps = compute_match(
        annotations,
        detections,
        (image_w, image_h),
        iou_threshold=0.5,
        evaltype="segm")

    ious = np.copy(overlaps)
    for i, dt in enumerate(detections):
      for j, gt in enumerate(annotations):
        if dt['id'] != gt['id']:
          ious[i, j] = 0
    ious = np.reshape(ious, (-1)).tolist()

    result = {
        'image_info': {
            "filepath": img_path,
            'width': image_w,
            'height': image_h,
            'depth': image_d
        },
        'elapsed_time': elapsed_time,
        'annotations': annotations,
        'detections': detections
    }

    results += [result]

    count += 1

    if VERBOSE:
      print("Prediction {0}/{1}:".format(count, total_images))
      print("FPS: {0:0.4f}".format(fps))
      if not DISABLE_ROTATED_BOXES and not ious is None:
        for iou in ious:
          print("IoU: {0:0.4f}".format(iou))

    # show detections
    if SHOW_DETECTION:
      # drawing.draw_boxes(
      #     image, boxes, class_ids=class_ids, scores=scores, labels=labels)

      for ann in annotations:
        drawing.draw_rotated_boxes(image, [ann['rbox']], colors=(0, 255, 0))

      if not rotated_boxes is None:
        image = drawing.draw_rotated_detections(image, detections, labels=['SSIV', 'Vessel', 'Ferry'], ious=ious)

      # save detection results
      cv2.imshow("image", image)
      cv2.imwrite("result.png", image)
      if cv2.waitKey(15) & 0xFF == ord('q'):
        break

    # if SAVE_IMAGE:
    #   # img_filepath = "noise-ssiv_bahia-{:03.0f}.png".format(noise * 100)
    #   # img_filepath = "noise-jequitaia-{:03.0f}.png".format(noise * 100)
    #   # img_filepath = "noise-balsa-{:03.0f}.png".format(noise * 100)
    #   x1, y1, w, h = annotations[0]['bbox']
    #   # pad = -30
    #   # ypad = -5
    #   ypad = 5
    #   # xpad = 25
    #   xpad = 15
    #   x1 = x1-xpad
    #   y1 = y1-ypad
    #   x2 = x1+w+xpad*2
    #   y2 = y1+h+ypad*2
    #   cropped = crop_roi(image, [x1, y1, x2, y2])
    #   print(cropped.shape)
    #   cv2.imshow("cropped", cropped)
    #   cv2.waitKey(15)
    #   cv2.imwrite(img_filepath, cropped)

  if len(results) > 0 and SAVE_RESULTS:
    result_filepath = ""
    prefix_filepath = ""
    if not noise is None:
      prefix_filepath = "{}_{:.3f}_{}".format(config.NAME.lower(), noise, "imgaug")
    elif IMAGE_AUG:
      prefix_filepath = "{}_{}".format(config.NAME.lower(), "imgaug")
    else:
      prefix_filepath = config.NAME.lower()

    result_filepath = "{}{:%Y%m%dT%H%M}.json".format(prefix_filepath,
                                                     datetime.datetime.now())
    print("Result saved in: {0}".format(result_filepath))
    json.dump(results, open(result_filepath, 'w'))

    evalcoco(result_filepath)
  # plot_average_recall(all_ious)

if __name__ == '__main__':
  load_weights()
  predict()