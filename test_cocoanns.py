#%% [markdown]
# ## Rboxnet - Test coco annotation
#
#%%
import os
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from pycocotools.coco import COCO

COCO_ANNS_PATH = "/home/gustavoneves/data/gemini/annotations"
BASE_FOLDER = "/home/gustavoneves/data/gemini/dataset/test"

coco=COCO(COCO_ANNS_PATH)
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))
nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

#%%
# select a random image
catIds = coco.getCatIds(catNms=[])
imgIds = coco.getImgIds(catIds=catIds)
imgIds = coco.getImgIds(imgIds)
idx = np.random.randint(0,len(imgIds))
img = coco.loadImgs(imgIds[idx])[0]

# %%
# load and display image
img_file_path = os.path.join(BASE_FOLDER, img['file_name'])
I = io.imread(img_file_path)
plt.axis('off')
plt.imshow(I)
plt.show()

#%%
# load and display instance annotations
plt.imshow(I); plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)

