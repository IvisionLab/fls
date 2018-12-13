#%%
# Evaluate results using COCO
import cv2
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from rboxnet import utils, visualize
from rboxnet.drawing import draw_verts
from rboxnet.eval import rotated_box_mask, calc_iou_from_masks, calc_ious
from rboxnet.eval import average_recall, compute_match, compute_ap
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from evalcoco import evalcoco

# Plot IOU/Recall

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
# RESULTS_FILEPATH = "assets/results/gemini_resnet50_deltas_imgaug20181208T1820.json"

# rotdim - resnet50
# RESULTS_FILEPATH = "assets/results/gemini_resnet50_rotdim_imgaug20181208T2138.json"

# deltas - resnet101
# RESULTS_FILEPATH = "assets/results/gemini_resnet101_deltas_imgaug20181208T1901.json"

# rotdim -resnet101
# RESULTS_FILEPATH = "assets/results/gemini_resnet101_rotdim_imgaug20181208T1616.json"

# mask-rcnn - resnet50
# RESULTS_FILEPATH = "assets/results/mask_rcnn_gemini_resnet50_imgaug20181208T1957.json"

# mask-rcnn - resnet101
# RESULTS_FILEPATH = "assets/results/mask_rcnn_gemini_resnet101_imgaug20181208T2029.json"

# yolo - deltas
# RESULTS_FILEPATH = "assets/results/yolo_rbox_deltas_imgaug20181208T2054.json"

# yolo - rotdim
# RESULTS_FILEPATH = "assets/results/yolo_rbox_rotdim_imgaug20181211T1942.json"

# total of results to be processed
TOTAL_RESULTS = -1

# show detections
SHOW_DETECTIONS = False

# show annotations
SHOW_ANNOTATIONS = False

FILTER_CLSID = -1

LABELS = ["ssiv_bahia", "jequitaia", "balsa"]


def evaluate(result_filepath):
  # load results file
  with open(result_filepath) as jsonfile:
    results = json.load(jsonfile)

  cnt = 0
  fps = 0.0

  # results = [r for r in results if len(r["detections"]) >= 2]

  if FILTER_CLSID > -1:
    label = LABELS[FILTER_CLSID]
    print("Label: ", label)
    results = [
        r for r in results if r["image_info"]["filepath"].find(label) != -1
    ]

  print("Result file: ", result_filepath)
  print("Total: ", len(results))

  all_dt_matches = []
  all_dt_scores = []
  all_gt_matches = []
  all_overlaps = []
  all_ious_matches = []

  N = TOTAL_RESULTS if TOTAL_RESULTS > 0 else len(results)

  for result in results[:N]:
    ellapsed_time = result['elapsed_time']
    image_info = result['image_info']
    detections = result['detections']
    annotations = result['annotations']

    image_shape = (image_info["height"], image_info["width"])

    dt_matches, gt_matches, scores, overlaps = compute_match(
        annotations, detections, image_shape, iou_threshold=0.5)

    iou = np.copy(overlaps)
    for i, dt in enumerate(detections):
      for j, gt in enumerate(annotations):
        if dt['id'] != gt['id']:
          iou[i, j] = 0

    all_ious_matches += np.reshape(iou, (-1)).tolist()

    all_dt_matches += dt_matches.tolist()
    all_gt_matches += gt_matches.tolist()
    all_dt_scores += scores.tolist()
    all_overlaps += np.reshape(overlaps, (-1)).tolist()

    fps += 1.0 / ellapsed_time
    cnt = cnt + 1

    if SHOW_DETECTIONS or SHOW_ANNOTATIONS:
      image = cv2.imread(image_info['filepath'])

      if SHOW_ANNOTATIONS:
        for gt in annotations:
          draw_verts(image, gt['rbox'], colors=(255, 0, 0))

      if SHOW_DETECTIONS:
        for dt in detections:
          draw_verts(image, dt['rbox'], colors=(0, 0, 255))

      cv2.imshow("results", image)
      if cv2.waitKey(15) & 0xFF == ord('q'):
        break

  # convert matches to array
  all_gt_matches = np.array(all_gt_matches, dtype=np.int32)
  all_dt_matches = np.array(all_dt_matches, dtype=np.int32)
  all_dt_scores = np.array(all_dt_scores)
  all_overlaps = np.array(all_overlaps)
  all_ious_matches = np.array(all_ious_matches)

  # sort predictions by score from high to low
  indices = np.argsort(all_dt_scores)[::-1]
  all_dt_scores = all_dt_scores[indices]
  all_dt_matches = all_dt_matches[indices]
  all_overlaps = all_overlaps[indices]
  all_ious_matches = all_ious_matches[indices]

  return all_dt_matches, all_dt_scores, all_overlaps, all_ious_matches, all_gt_matches


def save_matches(result_filepath, dt_matches, dt_scores, overlaps, ious):
  csv_lines = []
  for i in range(len(dt_matches)):
    csv_lines.append("{}, {}, {}, {}\n".format(dt_matches[i], dt_scores[i],
                                               overlaps[i], ious[i]))

  path, filename = os.path.split(result_filepath)
  base_name = os.path.splitext(filename)[0]

  if FILTER_CLSID > -1:
    base_name = "{}_{}".format(LABELS[FILTER_CLSID], base_name)

  csv_filename = "{}_matches.csv".format(base_name)
  print("Save matches in {}".format(csv_filename))

  with open(csv_filename, 'w') as csv_file:
    csv_file.writelines(csv_lines)


if __name__ == '__main__':
  result_files = [
    # deltas - resnet50
    # "assets/results/gemini_resnet50_deltas20181205T1744.json",
    "gemini_resnet50_deltas_0.450_imgaug20181213T1022.json",
    # # rotdim - resnet50
    # "assets/results/gemini_resnet50_rotdim20181205T1807.json",
    # # deltas - resnet101
    # "assets/results/gemini_resnet101_deltas20181205T2226.json",
    # # rotdim - resnet101
    # "assets/results/gemini_resnet101_rotdim20181205T2248.json",
    # # mask-rcnn - resnet50
    # "assets/results/mask_rcnn_gemini_resnet50_20181207T1921.json",
    # # mask-rcnn - resnet101
    # "assets/results/mask_rcnn_gemini_resnet101_20181206T1858.json",
    # # yolo - deltas
    # "assets/results/yolo_rbox_deltas_20181205T1953.json",
    # # yolo - rotdim
    # "assets/results/yolo_rbox_rotdim20181211T1954.json",
    # # HOG+SVM
    # "assets/results/hog_svm_all_20181212T0952.json"
  ]

  # result_files = [
  #     # deltas - resnet50
  #     "assets/results/gemini_resnet50_deltas_imgaug20181208T1820.json",
  #     # rotdim - resnet50
  #     "assets/results/gemini_resnet50_rotdim_imgaug20181208T2138.json",
  #     # deltas - resnet101
  #     "assets/results/gemini_resnet101_deltas_imgaug20181208T1901.json",
  #     # rotdim -resnet101
  #     "assets/results/gemini_resnet101_rotdim_imgaug20181208T1616.json",
  #     # mask-rcnn - resnet50
  #     "assets/results/mask_rcnn_gemini_resnet50_imgaug20181208T1957.json",
  #     # mask-rcnn - resnet101
  #     "assets/results/mask_rcnn_gemini_resnet101_imgaug20181208T2029.json",
  #     # yolo - deltas
  #     "assets/results/yolo_rbox_deltas_imgaug20181208T2054.json",
  #     # yolo - rotdim
  #     "assets/results/yolo_rbox_rotdim_imgaug20181211T1942.json"
  # ]

  stats = []
  for filepath in result_files:
    dt_matches, dt_scores, overlaps, ious, gt_matches = evaluate(filepath)
    save_matches(filepath, dt_matches, dt_scores, overlaps, ious)

    # coco eval
    cocoEval = evalcoco(filepath, class_id=FILTER_CLSID)
    cocoEval.summarize()

    overlap, recall, AR = average_recall(ious)
    print("Average Recall: {}".format(AR))

    s = [AR, cocoEval.stats[1], cocoEval.stats[2], cocoEval.stats[0], cocoEval.stats[8]]
    stats += [s]

  print("AR\t AP.5\tAP.75\tAP:.95\tAR:.95")
  for s in stats:
    print("{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}".format(
        s[0], s[1], s[2], s[3], s[4]))
