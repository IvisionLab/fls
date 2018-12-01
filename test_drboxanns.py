#%% [markdown]
# Create annotations for DRBOX
import os
import fnmatch
import random
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shutil import copyfile
from rboxnet import utils, config
from rboxnet.anns.common import get_rbbox_annotation, rbox2points

#%%
# Load parameters and configurations
TOTAL_FILES = -1
BASE_FOLDER = "/home/gustavoneves/data/gemini/dataset/train/jequitaia"
OUTPUT_FOLDER = "/home/gustavoneves/sources/drbox/data/jequitaia/"
class Config(config.Config):
  IMAGE_MAX_DIM = 448
conf = Config()

#%%
# List image files
files = os.listdir(BASE_FOLDER)
files = fnmatch.filter(files, '*.png')
files = fnmatch.filter(files, '*[!-mask].png')
random.shuffle(files)
if TOTAL_FILES > 0:
  files = files[:TOTAL_FILES]
else:
  TOTAL_FILES = len(files)

#%%
# Load random image
index = random.randint(0, TOTAL_FILES-1)
if index > 0:
  index = 0
filename = files[index]
img_path = os.path.join(BASE_FOLDER, filename)
image = skimage.io.imread(img_path)

#%%
# Load annotation
clsid, gt = get_rbbox_annotation(img_path)
rbbox = gt[4:]
rbbox[0:2] += gt[0:2]

print("Class ID: ", clsid)
print("Ground Truth: ", rbbox[:4])
print("Angle: ", rbbox[4])

#%%
# Draw random rotated bouding box
fig, ax = plt.subplots(1, figsize=(12, 12))
p = patches.Circle((rbbox[0], rbbox[1]), 5, linewidth=2, facecolor="red")
ax.add_patch(p)
pts = rbox2points(rbbox).astype(int)
p = patches.Polygon(pts, facecolor="none", edgecolor="red", linewidth=2)
ax.add_patch(p)
ax.imshow(image.astype(np.uint8))

#%%
# Resize the image to have the same size
image, window, scale, padding = utils.resize_image(
    image,
    min_dim=conf.IMAGE_MIN_DIM,
    max_dim=conf.IMAGE_MAX_DIM,
    padding=conf.IMAGE_PADDING)

#%%
# resize rotated bounding-box
rbbox[:4] *= scale
rbbox[:2] += [window[1], window[0]]

#%%
# Draw resized rotated bouding box
fig, ax = plt.subplots(1, figsize=(12, 12))
p = patches.Circle((rbbox[0], rbbox[1]), 5, linewidth=2, facecolor="red")
ax.add_patch(p)
pts = rbox2points(rbbox).astype(int)
p = patches.Polygon(pts, facecolor="none", edgecolor="red", linewidth=2)
ax.add_patch(p)
ax.imshow(image.astype(np.uint8))

#%%
# Generate drbox annotation files
if not os.path.exists(OUTPUT_FOLDER):
  os.makedirs(OUTPUT_FOLDER)
anns_folder = os.path.join(OUTPUT_FOLDER, "train_data")
if not os.path.exists(anns_folder):
  os.makedirs(anns_folder)

trainval_items = []
for i, filename in enumerate(files):
  # source image path
  src_img_path = os.path.join(BASE_FOLDER, filename)
  print("Source image: ", src_img_path)
  # load source image
  image = skimage.io.imread(src_img_path)
  # resize source image
  image, window, scale, padding = utils.resize_image(
      image,
      min_dim=conf.IMAGE_MIN_DIM,
      max_dim=conf.IMAGE_MAX_DIM,
      padding=conf.IMAGE_PADDING)

  # load rotated bounding-box annotation
  clsid, gt = get_rbbox_annotation(src_img_path)
  rbbox = gt[4:]
  rbbox[0:2] += gt[0:2]
  # resize rotated bounding-box annotation
  rbbox[:4] *= scale
  rbbox[:2] += [window[1], window[0]]
  cx, cy, w, h, angle = rbbox

  # destination image path
  dst_img_path = os.path.join(anns_folder, filename)
  # save resized image to destination folder
  skimage.io.imsave(dst_img_path, image)

  # # Draw resized rotated bouding box
  # fig, ax = plt.subplots(1, figsize=(12, 12))
  # p = patches.Circle((rbbox[0], rbbox[1]), 5, linewidth=2, facecolor="red")
  # ax.add_patch(p)
  # pts = rbox2points(rbbox).astype(int)
  # p = patches.Polygon(pts, facecolor="none", edgecolor="red", linewidth=2)
  # ax.add_patch(p)
  # ax.imshow(image.astype(np.uint8))

  anns_filename = filename + ".rbox"
  anns_path = os.path.join(anns_folder, anns_filename)
  # add trainval file item
  trainval_items.append("{0} {1}\n".format(filename, anns_filename))
  # get rotated bounding-box annotation
  clsid, gt = get_rbbox_annotation(src_img_path)
  rbbox = gt[4:]
  rbbox[0:2] += gt[0:2]
  cx, cy, w, h, angle = rbbox
  with open(anns_path, "w") as anns_file:
    anns_file.write("{0:.6f} {1:.6f} {2:.6f} {3:.6f} {4} {5:.6f}".format(
        cx, cy, w, h, clsid, angle * -1))

# write trainval file
with open(os.path.join(OUTPUT_FOLDER, "trainval.txt"), "w") as trainval_file:
  trainval_file.writelines(trainval_items)
