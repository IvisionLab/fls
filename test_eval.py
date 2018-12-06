#%%
# Evaluate results using COCO
import cv2
import json
import numpy as np
from rboxnet import utils, visualize
from rboxnet.drawing import draw_verts
from rboxnet.eval import rotated_box_mask, calc_iou_from_masks, calc_ious, plot_average_recall

# file path to the results
# RESULTS_FILEPATH = "assets/results/gemini_resnet50_deltas20181205T1744.json"
# RESULTS_FILEPATH = "assets/results/gemini_resnet50_rotdim20181205T1807.json"
# RESULTS_FILEPATH = "assets/results/gemini_resnet50_verts20181205T1828.json"
# RESULTS_FILEPATH = "assets/results/gemini_resnet101_deltas20181205T2226.json"
# RESULTS_FILEPATH = "assets/results/gemini_resnet101_rotdim20181205T2248.json"
# RESULTS_FILEPATH = "assets/results/gemini_resnet101_verts20181205T2310.json"
# RESULTS_FILEPATH = "assets/results/mask_rcnn_gemini_resnet10120181205T2030.json"
RESULTS_FILEPATH = "assets/results/yolo_rbox20181205T1953.json"

# total of results to be processed
TOTAL_RESULTS = -1

# show detections
SHOW_DETECTIONS = False

# show annotations
SHOW_ANNOTATIONS = False

# number of classes
NB_CLASS = 3

# load results file
with open(RESULTS_FILEPATH) as jsonfile:
  results = json.load(jsonfile)


all_ious = []
cnt = 0
fps = 0.0
for result in results[:TOTAL_RESULTS]:
  ellapsed_time = result['elapsed_time']
  image_info = result['image_info']
  detections = result['detections']
  annotations = result['annotations']
  fps += 1.0 / ellapsed_time
  cnt = cnt +1


  image_shape = (image_info["height"], image_info["width"])

  ious = calc_ious(annotations, detections, image_shape, NB_CLASS)
  all_ious += [ious]

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

fps = fps / cnt
print("Average FPS: {0}".format(fps))
plot_average_recall(all_ious)