#%% Test result
import cv2
import os
import json
import math
import numpy as np
from rboxnet.anns.common import adjust_rotated_boxes
from rboxnet.drawing import draw_verts, draw_boxes

# base folders
replace_folder = "/workspace/data/gemini/dataset/test/"
base_folder = "/home/gustavoneves/data/gemini/dataset/test/"

# result directory
result_dir = "assets/experiments/20181213/"

# labels
labels = ["ssiv_bahia", "jequitaia", "balsa"]

# result filename
result_filename = "gemini_resnet50_deltas/gemini_resnet50_deltas_0.200_imgaug20181213T1550.json"
# result_filename = "mask_rcnn_gemini_resnet50/mask_rcnn_gemini_resnet50_0.200_imgaug_20181213T1816.json"
# result_filename = "yolo_rbox_deltas/yolo_rbox_deltas_0.200_imgaug20181213T2107.json"
# result_filename = "yolo_rbox_rotdim/yolo_rbox_rotdim_0.200_imgaug20181213T2123.json"

filepath = os.path.join(result_dir, result_filename)

# load results file
with open(filepath) as fs:
  results = json.load(fs)
  results = [ r for r in results if not r["detections"] == [] ]

total_results = len(results)


# detections
for r in results:
  info, dts, gts = r["image_info"], r["detections"], r['annotations']

  filepath = ""
  # load image id from coco annotations
  base_name = info['filepath'].replace(replace_folder, '')
  filepath = os.path.join(base_folder, base_name)

  image = cv2.imread(filepath)

  if dts == []:
    continue

  for gt in gts:
    draw_verts(image, gt['rbox'], colors=(0, 0, 255))


  for dt in dts:
    rbox = dt['rbox']
    bbox = dt['bbox']
    rbox = adjust_rotated_boxes(rbox, bbox)
    draw_verts(image, rbox, colors=(255, 0, 0))
    draw_boxes(image, [dt['bbox']], [dt['id']], [dt['score']], labels)

  cv2.imshow("results", image)
  if cv2.waitKey() & 0xFF == ord('q'):
    break
