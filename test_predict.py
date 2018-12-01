#%% [markdown]
# ## Rboxnet - Test prediction
#
#%%

import os
import json
import random
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon
import rboxtrain
import rboxnet
import rboxnet.dataset
from rboxnet import inference, config, model

# Root directory of the project
# ROOT_DIR = os.getcwd()
ROOT_DIR = "/home/gustavoneves/sources/rboxnet/"

# Trained model directory
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to Shapes trained weights
RBOXNET_MODEL_PATH = os.path.join(
    ROOT_DIR,
    "logs/gemini_resnet50_deltas20181126T2216/rboxnet_gemini_0160.h5")

# Path to configuration file
CONFIG_PATH = os.path.join(ROOT_DIR, "cfg/gemini_deltas.json")


#%% [markdown]
# ## Create inference configuration.
#
#%%
class InferenceConfig(rboxtrain.TrainingConfig):
  GPU_COUNT = 1
  IMAGES_PER_GPU = 1
  DETECTION_MIN_CONFIDENCE = 0
  BACKBONE = "resnet50"


config = InferenceConfig()

#%% [markdown]
# ## Load dataset
#
#%%
with open(CONFIG_PATH) as f:
  cfg = json.load(f)
  anns_path = os.path.join(ROOT_DIR, cfg['annotations']['test'])
  dataset = rboxnet.dataset.gemini_dataset(anns_path, shuffle=False)
  config.regressor = cfg['regressor']

print("Images: {0}\nClasses: {1}".format(
    len(dataset.image_ids), dataset.class_names))

image_id = random.choice(dataset.image_ids)
image = dataset.load_image(image_id)

DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0
with tf.device(DEVICE):
  net = inference.Inference(config)

#%% [markdown]
# Load trained weights
net.load_weights(RBOXNET_MODEL_PATH, by_name=True)

#%% [markdown]
# run detector
molded_images, image_metas, windows = net.mold_inputs([image])

rpn_rois, rpn_class, rpn_bbox, rbox_dts = net.keras_model.predict(
    [molded_images, image_metas], verbose=0)

print(rbox_dts.shape)

image_shape = image.shape


def unmold_detections(dts, image_shape, window, config):
  zero_ix = np.where(dts[:, 0] == 0)[0]
  N = zero_ix[0] if zero_ix.shape[0] > 0 else dts.shape[0]
  class_ids = dts[:N, 0].astype(np.int32)
  scores = dts[:N, 1]
  boxes = dts[:N, 2:6]

  # Compute scale and shift to translate coordinates to image domain.
  h_scale = image_shape[0] / (window[2] - window[0])
  w_scale = image_shape[1] / (window[3] - window[1])

  scale = min(h_scale, w_scale)
  shift = window[:2]  # y, x
  scales = np.array([scale, scale, scale, scale])
  shifts = np.array([shift[0], shift[1], shift[0], shift[1]])

  # Filter out detections with zero area. Often only happens in early
  # stages of training when the network weights are still a bit random.
  exclude_ix = np.where(
      (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]

  # Translate bounding boxes to image domain
  boxes = np.multiply(boxes - shifts, scales).astype(np.int32)
  if config.regressor == "deltas":
    tlines = dts[:N, 6:10]
    tlines = np.multiply(tlines - shifts, scales).astype(np.int32)

  outputs = []
  if exclude_ix.shape[0] > 0:
    boxes = np.delete(boxes, exclude_ix, axis=0)
    class_ids = np.delete(class_ids, exclude_ix, axis=0)
    scores = np.delete(scores, exclude_ix, axis=0)

    if config.regressor == "deltas":
      tlines = np.delete(tlines, exclude_ix, axis=0)

  outputs = [class_ids, scores, boxes]

  if config.regressor == "deltas":
    outputs += [tlines]

  return outputs


def tlines_to_rboxes(tlines, boxes):
  y1 = tlines[:, 0]
  x1 = tlines[:, 1]
  y2 = tlines[:, 2]
  x2 = tlines[:, 3]
  y3 = boxes[:, 2] - (y1 - boxes[:, 0])
  x3 = boxes[:, 3] - (x1 - boxes[:, 1])
  y4 = boxes[:, 0] + (boxes[:, 2] - y2)
  x4 = boxes[:, 1] + (boxes[:, 3] - x2)
  return np.stack([y1, x1, y2, x2, y3, x3, y4, x4], axis=1)


#%%
# show detection results
tlines = None
rboxes = None
if config.regressor == "deltas":
  class_ids, scores, boxes, tlines = unmold_detections(
      rbox_dts[0], image.shape, windows[0], config)
  rboxes = tlines_to_rboxes(tlines, boxes)
else:
  class_ids, scores, boxes = unmold_detections(rbox_dts[0], image.shape,
                                               windows[0], config)

color = "red"
style = "solid"
alpha = 1

fig, ax1 = plt.subplots(1, figsize=(12, 12))
ax1.imshow(image.astype(np.uint8))

for i, cls_id in enumerate(class_ids):
  print("{0}: {1}".format(dataset.class_info[cls_id]['name'], scores[i]))
  y1, x1, y2, x2 = boxes[i]
  p = patches.Rectangle((x1, y1),
                        x2 - x1,
                        y2 - y1,
                        linewidth=2,
                        alpha=alpha,
                        linestyle=style,
                        edgecolor=color,
                        facecolor='none')
  ax1.add_patch(p)


# if rboxes:
#   fig, ax2 = plt.subplots(1, figsize=(12, 12))

#   for i, cls_id in enumerate(class_ids):
#     print("{0}: {1}".format(dataset.class_info[cls_id]['name'], scores[i]))
#     y1, x1, y2, x2 = boxes[i]
#     p = patches.Rectangle((x1, y1),
#                           x2 - x1,
#                           y2 - y1,
#                           linewidth=2,
#                           alpha=alpha,
#                           linestyle=style,
#                           edgecolor=color,
#                           facecolor='none')
#     ax1.add_patch(p)

#     verts = np.reshape(rboxes[i], (-1, 2))
#     verts = np.fliplr(verts)
#     p = Polygon(verts, facecolor="none", linewidth=2, edgecolor=color)
#     ax2.add_patch(p)

#   ax2.imshow(image.astype(np.uint8))
#   plt.show()