#%% [markdown]
# ## Rboxnet - Inspect Trained Model
#
# Inspect rboxnet model
#%%

import os
import json
import random
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import rboxtrain
import rboxnet
import rboxnet.dataset
from rboxnet import inference, config, model

# Root directory of the project
ROOT_DIR = os.getcwd()

# Trained model directory
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to Shapes trained weights
RBOXNET_MODEL_PATH = os.path.join(ROOT_DIR,
                                  "logs/gemini20181126T2216/rboxnet_gemini_0067.h5")

# Path to configuration file
CONFIG_PATH = os.path.join(ROOT_DIR, "cfg/gemini.json")


#%% [markdown]
# ## Create inference configuration.
#
#%%
class InferenceConfig(rboxtrain.TrainingConfig):
  GPU_COUNT = 1
  IMAGES_PER_GPU = 1
  DETECTION_MIN_CONFIDENCE = 0


config = InferenceConfig()

#%% [markdown]
# ## Create inference model
# Device to load the neural network on.
# Useful if you're training a model on the same
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
with tf.device(DEVICE):
  net = inference.Inference(config)

#%% [markdown]
# ## Load trained weights
#%%
net.load_weights(RBOXNET_MODEL_PATH, by_name=True)

#%% [markdown]
# ## Load dataset
#
#%%
with open(CONFIG_PATH) as f:
  cfg = json.load(f)
  anns_path = cfg['annotations']['test']
  dataset = rboxnet.dataset.gemini_dataset(
      cfg['annotations']['test'], shuffle=False)

print("Images: {0}\nClasses: {1}".format(
    len(dataset.image_ids), dataset.class_names))

image_id = random.choice(dataset.image_ids)
image, image_meta, _, _, _, _, _ = model.load_image_gt(
    dataset, config, image_id, use_mini_mask=False)


#%% [markdown]
# ## Run detector
#

rpn_rois, rpn_class, rpn_bbox = net.detect([image], verbose=1)

#%% [markdown]
# ## Plot anchors
#
limit = 10
rpn_class_sorted = np.argsort(rpn_class[:, :, 1].flatten())[::-1]
boxes = net.anchors[rpn_class_sorted[:limit]]
N = boxes.shape[0]

#%% [markdown]
# ## Show anchors
#
color = "red"
style = "dotted"
alpha = 1
fig,ax = plt.subplots(1, figsize=(12, 12))

for i in range(N):
  y1, x1, y2, x2 = boxes[i]
  p = patches.Rectangle((x1, y1),
                        x2 - x1,
                        y2 - y1,
                        linewidth=2,
                        alpha=alpha,
                        linestyle=style,
                        edgecolor=color,
                        facecolor='none')
  ax.add_patch(p)

# Display the image
ax.imshow(image.astype(np.uint8))
plt.show()