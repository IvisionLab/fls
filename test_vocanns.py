#%% [markdown]
# Read VOC annotation
import os
import numpy as np
import skimage.io
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#%%
# Load parameters and configurations
BASE_FOLDER = "/home/gustavoneves/data/gemini/dataset/train"
ANNS_FOLDER = "assets/annotations/voc/train"
ANNS_FILE = "ssiv_bahia-sample-0000002.xml"

#%%
# Load XML file
tree = ET.parse(os.path.join(ANNS_FOLDER, ANNS_FILE))

img = {'objects': []}
for elem in tree.iter():
  if 'filename' in elem.tag:
    img['filename'] = os.path.join(BASE_FOLDER, elem.text)
  if 'object' in elem.tag:
    obj = {}
    for attr in list(elem):
      if 'bndbox' in attr.tag:
        for dim in list(attr):
          if 'xmin' in dim.tag:
            obj['xmin'] = int(round(float(dim.text)))
          if 'ymin' in dim.tag:
            obj['ymin'] = int(round(float(dim.text)))
          if 'xmax' in dim.tag:
            obj['xmax'] = int(round(float(dim.text)))
          if 'ymax' in dim.tag:
            obj['ymax'] = int(round(float(dim.text)))

      if 'name' in attr.tag:
        obj['name'] = attr.text
        img['objects'] += [obj]

#%%
# Load image
image = skimage.io.imread(img['filename'])

#%%
# Draw annotation bounding box
fig, ax = plt.subplots(1, figsize=(12, 12))
for obj in img['objects']:
  print("Object name: ", obj['name'])

  x1 = obj['xmin']
  y1 = obj['ymin']
  x2 = obj['xmax']
  y2 = obj['ymax']

  p = patches.Rectangle((x1, y1),
                        x2 - x1,
                        y2 - y1,
                        linewidth=2,
                        edgecolor="red",
                        facecolor='none')
  ax.add_patch(p)

ax.imshow(image.astype(np.uint8))
plt.show()