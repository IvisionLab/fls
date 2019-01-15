#%%
# Evaluate results using COCO
import os
import cv2
import json
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
from rboxnet.eval import rotated_box_mask
from rboxnet.drawing import draw_verts
from rboxnet.anns.common import adjust_rotated_boxes
from pycocotools.coco import COCO
from pycocotools.mask import encode
from pycocotools.cocoeval import COCOeval

# COCO evaluation

#######################################
# original
#######################################

# deltas - resnet50
# RESULTS_FILEPATH = "assets/results/gemini_resnet50_deltas20181205T1744.json"

# rotdim - resnet50
# RESULTS_FILEPATH = "assets/results/gemini_resnet50_rotdim20181205T1807.json"

# deltas - resnet101
# RESULTS_FILEPATH = "assets/results/gemini_resnet101_deltas20181205T2226.json"

# rotdim - resnet101
# RESULTS_FILEPATH = "assets/results/gemini_resnet101_rotdim20181205T2248.json"

# mask-rcnn - resnet50
# RESULTS_FILEPATH = "assets/results/mask_rcnn_gemini_resnet50_20181207T1921.json"

# mask-rcnn - resnet101
# RESULTS_FILEPATH = "assets/results/mask_rcnn_gemini_resnet101_20181206T1858.json"

# yolo - deltas
# RESULTS_FILEPATH = "assets/results/yolo_rbox_deltas_20181205T1953.json"

# yolo - rotdim
# RESULTS_FILEPATH = "assets/results/yolo_rbox_rotdim20181211T1954.json"

# HOG+SVM
# RESULTS_FILEPATH = "assets/results/hog_svm_all_20181212T0952.json"
#######################################
# with pertubations
#######################################

# deltas - resnet50
# RESULTS_FILEPATH = "assets/results/gemini_resnet50_deltas_imgaug20181208T1715.json"
# RESULTS_FILEPATH = "assets/results/gemini_resnet50_deltas_imgaug20181208T1352.json"
# RESULTS_FILEPATH = "assets/results/gemini_resnet50_deltas_imgaug20181208T1820.json"

# rotdim - resnet50
# RESULTS_FILEPATH = "assets/results/gemini_resnet50_rotdim_imgaug20181208T1453.json"

# deltas - resnet101
# RESULTS_FILEPATH = "assets/results/gemini_resnet101_deltas_imgaug20181208T1529.json"
# RESULTS_FILEPATH = "assets/results/gemini_resnet101_deltas_imgaug20181208T1901.json"

# rotdim -resnet101
# RESULTS_FILEPATH = "assets/results/gemini_resnet101_rotdim_imgaug20181208T1616.json"

# mask-rcnn - resnet50
# RESULTS_FILEPATH = "assets/results/mask_rcnn_gemini_resnet50_imgaug20181208T1957.json"

# mask-rcnn - resnet101
# RESULTS_FILEPATH = "assets/results/mask_rcnn_gemini_resnet101_imgaug20181208T2029.json"

# image base folder
BASE_FOLDER = "/home/gustavoneves/data/gemini/dataset/test/"
BASE_FOLDER_WORSPACE = "/workspace/data/gemini/dataset/test/"

# coco annotation path
COCO_ANNS_PATH = "assets/annotations/coco_annotations_test.json"
MASK_RCNN_COCO_ANNS_PATH = "assets/annotations/mask_rcnn_coco_annotations_test.json"

# source labels
LABELS = ["ssiv_bahia", "jequitaia", "balsa"]

# max results
TOTAL_RESULTS = -1

# show detections
SHOW_DETECTIONS = False

# show annotations
SHOW_ANNOTATIONS = False

# plot coco annotations
COCO_PLOT_ANNOTATIONS = False

# get coco id from source class id
def coco_catId(idx):
  source_ids = {0: 1, 1: 3, 2: 2}
  return source_ids[idx]

def evalcoco(results_filepath, evaltype="segm", class_id=-1, enable_adjustment=False):
  # load results file
  with open(results_filepath) as jsonfile:
    results = json.load(jsonfile)

  is_mask_rcnn = True if not results_filepath.find("mask_rcnn") == -1 else False

  # load images from coco annotations
  if is_mask_rcnn and evaltype=="bbox":
    coco = COCO(MASK_RCNN_COCO_ANNS_PATH)
  else:
    coco = COCO(COCO_ANNS_PATH)

  if class_id > -1:
    catId = coco_catId(class_id)
    cats = coco.loadCats(coco.getCatIds(catIds=[catId]))
    imgIds = coco.getImgIds(coco.getImgIds(catIds=[catId]))
  else:
    imgIds = coco.getImgIds(coco.getImgIds())
  imgs = coco.loadImgs(imgIds)

  # build coco results
  coco_results = []
  coco_image_ids = []
  N = TOTAL_RESULTS if TOTAL_RESULTS > 0 else len(results)
  for r in results[:N]:
    image_info = r['image_info']

    # load image id from coco annotations
    file_name = image_info['filepath'].replace(BASE_FOLDER, '')
    image_ids = [img['id'] for img in imgs if img["file_name"] == file_name]

    if not image_ids:
      file_name = image_info['filepath'].replace(BASE_FOLDER_WORSPACE, '')
      image_ids = [img['id'] for img in imgs if img["file_name"] == file_name]
      if not image_ids:
        continue

    image_id = image_ids[0]
    coco_image_ids += [image_id]

    # detections results
    detections = r['detections']
    if not detections == [] and enable_adjustment == True:
      for dt in detections:
        dt["rbox"] = adjust_rotated_boxes(dt["rbox"], dt["bbox"])


    # annotations
    annotations = r['annotations']

    # image shape
    image_shape = (image_info["height"], image_info["width"])

    if COCO_PLOT_ANNOTATIONS and not TOTAL_RESULTS > 10 and TOTAL_RESULTS > 0:
      img = coco.loadImgs(image_id)[0]
      I = skimage.io.imread(os.path.join(BASE_FOLDER, img['file_name']))
      plt.imshow(I); plt.axis('off')
      annIds = coco.getAnnIds(imgIds=img['id'], catIds=[], iscrowd=None)
      anns = coco.loadAnns(annIds)
      coco.showAnns(anns)
      plt.show()

    # build coco results
    if not detections:
      coco_results.append({
            "image_id": image_id,
            "category_id": None,
            "bbox": [-1, -1, -1, -1],
            "score": None,
            "segmentation": []
        })
    else:
      for dt in detections:
        category_id = coco_catId(dt['id'])
        score = dt['score']
        bbox = np.around(dt['bbox'])
        mask = rotated_box_mask(image_shape, [dt['rbox']])
        rle = encode(np.asfortranarray(mask))
        rle['counts'] = str(rle['counts'], 'utf-8')

        x1, y1, x2, y2 = bbox.astype(int)

        coco_results.append({
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "score": float(score),
            "segmentation": rle
        })

      if SHOW_DETECTIONS or SHOW_ANNOTATIONS:
        image = cv2.imread(image_info['filepath'])
        if SHOW_ANNOTATIONS:
          for gt in annotations:
            draw_verts(image, gt['rbox'], colors=(255, 0, 0))

        if SHOW_DETECTIONS:
          for dt in detections:
            draw_verts(image, dt['rbox'], colors=(0, 0, 255))

        cv2.imshow("results", image)
        if cv2.waitKey() & 0xFF == ord('q'):
          break

  # coco evaluation
  print("Total coco results: ", len((coco_results)))
  cocoEval = COCOeval(coco, coco.loadRes(coco_results), iouType=evaltype)
  cocoEval.params.imgIds = coco_image_ids
  cocoEval.evaluate()
  cocoEval.accumulate()
  return cocoEval

if __name__ == '__main__':
  evalcoco(RESULTS_FILEPATH)