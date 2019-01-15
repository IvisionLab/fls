#%%
# Evaluate results using COCO
import cv2
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from rboxnet import utils, visualize
from rboxnet.drawing import draw_verts, draw_boxes, draw_rotated_detections
from rboxnet.anns.common import adjust_rotated_boxes
from rboxnet.eval import rotated_box_mask, calc_iou_from_masks, calc_ious
from rboxnet.eval import average_recall, compute_match, compute_ap
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from evalcoco import evalcoco

# Plot IOU/Recall

# total of results to be processed
TOTAL_RESULTS = -1

# show detections
SHOW_DETECTIONS = True

# show annotations
SHOW_ANNOTATIONS = True

# show bbox
SHOW_BBOX = True

FILTER_CLSID = 2

ENABLE_ADJUSTMENT = False

LABELS = ["ssiv_bahia", "jequitaia", "balsa"]

SAVE_MATCHES = False

replace_folder = "/workspace/data/gemini/dataset/test/"
base_folder = "/home/gustavoneves/data/gemini/dataset/test/"


def evaluate(result_filepath, enable_adjustment=False, evaltype="segm"):
  # load results file
  with open(result_filepath) as jsonfile:
    results = json.load(jsonfile)

  cnt = 0
  fps = 0.0


  if FILTER_CLSID > -1:
    label = LABELS[FILTER_CLSID]
    print("Label: ", label)
    results = [
        r for r in results if r["image_info"]["filepath"].find(label) != -1
    ]

  results = [r for r in results if len(r["detections"]) >= 2]

  # results = [
  #     r for r in results if r["image_info"]["filepath"].find("0000932.png") != -1
  # ]

  # bbox ssiv
  # results = [
  #     r for r in results if r["image_info"]["filepath"].find("0001410.png") != -1
  # ]

  # # bbox jequitaia
  # results = [
  #     r for r in results if r["image_info"]["filepath"].find("0000500.png") != -1
  # ]

  # ssiv bahia
  # results = [
  #     r for r in results if r["image_info"]["filepath"].find("0002005.png") != -1
  # ]

  filepaths = [r["image_info"]["filepath"] for r in results]

  indices = sorted(range(len(filepaths)), key=lambda k: filepaths[k])

  print("Result file: ", result_filepath)
  print("Total: ", len(results))

  all_dt_matches = []
  all_dt_scores = []
  all_gt_matches = []
  all_overlaps = []
  all_ious_matches = []

  N = TOTAL_RESULTS if TOTAL_RESULTS > 0 else len(results)

  is_mask_rcnn = True if not result_filepath.find("mask_rcnn") == -1 else False

  for idx in indices[:N]:
    result = results[idx]
    ellapsed_time = result['elapsed_time']
    image_info = result['image_info']
    detections = result['detections']
    annotations = result['annotations']

    if not is_mask_rcnn:
      for gt in annotations:
        bb = gt['bbox']
        bb[2] = bb[0] + bb[2]
        bb[3] = bb[1] + bb[3]

    if not detections == [] and enable_adjustment == True:
      for dt in detections:
        dt["rbox"] = adjust_rotated_boxes(dt["rbox"], dt["bbox"])

    image_shape = (image_info["height"], image_info["width"])

    dt_matches, gt_matches, scores, overlaps = compute_match(
        annotations,
        detections,
        image_shape,
        iou_threshold=0.5,
        evaltype=evaltype)

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

      # load image id from coco annotations
      if not os.path.exists(image_info['filepath']):
        base_name = image_info['filepath'].replace(replace_folder, '')
        image_info['filepath'] = os.path.join(base_folder, base_name)

      print(image_info['filepath'])
      image = cv2.imread(image_info['filepath'])

      if SHOW_ANNOTATIONS:
        if SHOW_BBOX and evaltype == "bbox":
          for gt in annotations:
            bb = gt['bbox']
            image = draw_boxes(image, [bb], color=(0, 255, 0))

        if evaltype == "segm":
          for gt in annotations:
            draw_verts(image, gt['rbox'], colors=(0, 255, 0))

      if SHOW_DETECTIONS:
        ious = np.reshape(iou, (-1)).tolist()
        if SHOW_BBOX and evaltype == "bbox":
          for i, dt in enumerate(detections):
            image = draw_boxes(image, [dt['bbox']],
                                color=(0, 0, 255),
                                class_ids=[dt['id']],
                                # scores=[dt['score']],
                                scores=None,
                                labels=["SSIV", "Vessel", "Ferry"],
                                ious=[ious[i]])

        if evaltype == "segm":
          image = draw_rotated_detections(image, detections, labels=["SSIV", "Vessel", "Ferry"], ious=None, score=False)

      cv2.imshow("results", image)
      if cv2.waitKey() & 0xFF == ord('q'):
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

  fps = fps / cnt
  print("fps: {}".format(fps))
  return all_dt_matches, all_dt_scores, all_overlaps, all_ious_matches, all_gt_matches, fps


def save_matches(result_filepath,
                 dt_matches,
                 dt_scores,
                 overlaps,
                 ious,
                 evaltype="segm"):
  csv_lines = []
  for i in range(len(dt_matches)):
    csv_lines.append("{}, {}, {}, {}\n".format(dt_matches[i], dt_scores[i],
                                               overlaps[i], ious[i]))

  path, filename = os.path.split(result_filepath)
  base_name = os.path.splitext(filename)[0]

  if FILTER_CLSID > -1:
    base_name = "{}_{}".format(LABELS[FILTER_CLSID], base_name)

  if not evaltype == "segm":
    base_name = "{}_{}".format(base_name, evaltype)

  if SAVE_MATCHES:
    csv_filename = "{}_matches.csv".format(base_name)
    print("Save matches in {}".format(csv_filename))

    with open(csv_filename, 'w') as csv_file:
      csv_file.writelines(csv_lines)


if __name__ == '__main__':

  # result_files = [
  #   "assets/experiments/20181222/mask_rcnn_gemini_resnet50/mask_rcnn_gemini_resnet50_20181222T1841-256.json",
  #   "assets/experiments/20181222/mask_rcnn_gemini_resnet50/mask_rcnn_gemini_resnet50_20181222T1905-320.json",
  #   "assets/experiments/20181222/mask_rcnn_gemini_resnet50/mask_rcnn_gemini_resnet50_20181222T1929-384.json",
  #   "assets/experiments/20181222/mask_rcnn_gemini_resnet50/mask_rcnn_gemini_resnet50_20181207T1921-448.json"
  # ]

  # result_files = [
  #     "assets/experiments/20181222/yolo_rbox_deltas/yolo_rbox_deltas20181222T1733-256.json",
  #     "assets/experiments/20181222/yolo_rbox_deltas/yolo_rbox_deltas20181222T1617-320.json",
  #     "assets/experiments/20181222/yolo_rbox_deltas/yolo_rbox_deltas20181222T1628-384.json",
  #     "assets/experiments/20181222/yolo_rbox_deltas/yolo_rbox_deltas20181205T1953-416.json"
  # ]


  # result_files = [
  #   "assets/experiments/20181222/mask_rcnn_gemini_resnet101/mask_rcnn_gemini_resnet101_20181222T2038-256.json",
  #   "assets/experiments/20181222/mask_rcnn_gemini_resnet101/mask_rcnn_gemini_resnet101_20181222T2100-320.json",
  #   "assets/experiments/20181222/mask_rcnn_gemini_resnet101/mask_rcnn_gemini_resnet101_20181222T2122-384.json",
  #   "assets/experiments/20181222/mask_rcnn_gemini_resnet101/mask_rcnn_gemini_resnet101_20181206T1858-448.json"
  # ]

  # result_files = [
  #   "assets/experiments/20181222/yolo_rbox_rotdim/yolo_rbox_rotdim20181222T2135-256.json",
  #   "assets/experiments/20181222/yolo_rbox_rotdim/yolo_rbox_rotdim20181222T2146-320.json",
  #   "assets/experiments/20181222/yolo_rbox_rotdim/yolo_rbox_rotdim20181222T2211-384.json",
  #   "assets/experiments/20181222/yolo_rbox_rotdim/yolo_rbox_rotdim20181211T0126-416.json",
  # ]

  # result_files = [
    # "assets/experiments/20181222/gemini_resnet101_deltas/gemini_resnet101_deltas20181222T2342-256.json",
    # "assets/experiments/20181222/gemini_resnet101_deltas/gemini_resnet101_deltas20181222T2358-320.json",
    # "assets/experiments/20181222/gemini_resnet101_deltas/gemini_resnet101_deltas20181223T0019-384.json",
    # "assets/experiments/20181222/gemini_resnet101_deltas/gemini_resnet101_deltas20181205T2226-448.json"
  # ]

  # result_files = [
  #   "assets/experiments/20181222/gemini_resnet101_rotdim/gemini_resnet101_rotdim20181223T0836-256.json",
  #   "assets/experiments/20181222/gemini_resnet101_rotdim/gemini_resnet101_rotdim20181223T0901-320.json",
  #   "assets/experiments/20181222/gemini_resnet101_rotdim/gemini_resnet101_rotdim20181223T0929-384.json",
  #   "assets/experiments/20181222/gemini_resnet101_rotdim/gemini_resnet101_rotdim20181205T2248-448.json"
  # ]

  # result_files = [
  #   "assets/experiments/20181222/gemini_resnet50_deltas/gemini_resnet50_deltas20181223T1018-256.json",
  #   "assets/experiments/20181222/gemini_resnet50_deltas/gemini_resnet50_deltas20181223T1034-320.json",
  #   "assets/experiments/20181222/gemini_resnet50_deltas/gemini_resnet50_deltas20181223T1106-384.json",
  #   "assets/experiments/20181222/gemini_resnet50_deltas/gemini_resnet50_deltas20181205T1744-448.json"
  # ]

  # result_files = [
  #   "assets/experiments/20181222/gemini_resnet50_rotdim/gemini_resnet50_rotdim20181223T1151-256.json",
  #   "assets/experiments/20181222/gemini_resnet50_rotdim/gemini_resnet50_rotdim20181223T1210-320.json",
  #   "assets/experiments/20181222/gemini_resnet50_rotdim/gemini_resnet50_rotdim20181223T1255-384.json",
  #   "assets/experiments/20181222/gemini_resnet50_rotdim/gemini_resnet50_rotdim20181205T1807-448.json"
  # ]

  # result_files = [
  #     "assets/results/gemini_resnet50_verts/gemini_resnet50_verts20181223T1522.json",
  #     "assets/results/gemini_resnet101_verts/gemini_resnet101_verts20181223T1552.json"
  # ]

  # result_files = [
    # "assets/results/faster_rcnn_gemini_resnet50/faster_rcnn_gemini_resnet50_20181219T1200.json",
    # "assets/results/faster_rcnn_gemini_resnet50/faster_rcnn_gemini_resnet50_20181218T1240.json"
  # ]

  # result_files = [
    # "assets/results/faster_rcnn_gemini_resnet101/faster_rcnn_gemini_resnet101_20181218T1308.json",
    # "assets/results/faster_rcnn_gemini_resnet101/faster_rcnn_gemini_resnet101_20181219T1303.json"
  # ]

  # result_files = [
  #   "assets/results/yolo_rbox_deltas/yolo_rbox_deltas_20181205T1953.json",
    # "assets/results/yolo_rbox_deltas/yolo_rbox_deltas20181220T1445.json",
    # "assets/results/yolo_rbox_rotdim/yolo_rbox_rotdim20181211T0126.json",
    # "assets/results/yolo_rbox_rotdim/yolo_rbox_rotdim20181211T1954.json"
  # ]

  # result_files = [
  #   "assets/results/gemini_resnet101_verts/gemini_resnet101_verts20181223T1552.json",
  #   "assets/results/gemini_resnet101_verts/gemini_resnet101_verts_ssiv_bahia_20181224T1719.json",
  #   "assets/results/gemini_resnet101_verts/gemini_resnet101_verts_jequitaia_20181224T1726.json",
  #   "assets/results/gemini_resnet101_verts/gemini_resnet101_verts_balsa_20181224T1740.json",
  #   "assets/results/gemini_resnet50_verts/gemini_resnet50_verts20181223T1522.json",
  #   "assets/results/gemini_resnet50_verts/gemini_resnet50_verts_ssiv_bahia_20181223T1606.json",
  #   "assets/results/gemini_resnet50_verts/gemini_resnet50_verts_jequitaia_20181223T1614.json",
  #   "assets/results/gemini_resnet50_verts/gemini_resnet50_verts_balsa_20181223T1630.json",
  # ]


  # result_files = [
  #   "assets/experiments/20181222/gemini_resnet50_deltas/gemini_resnet50_deltas20181205T1744-448.json",
  #   "assets/experiments/20181222/gemini_resnet50_rotdim/gemini_resnet50_rotdim20181205T1807-448.json",
  #   "assets/results/gemini_resnet50_verts/gemini_resnet50_verts20181223T1522.json",
  #   "assets/experiments/20181222/gemini_resnet101_deltas/gemini_resnet101_deltas20181205T2226-448.json",
  #   "assets/experiments/20181222/gemini_resnet101_rotdim/gemini_resnet101_rotdim20181205T2248-448.json",
  #   "assets/results/gemini_resnet101_verts/gemini_resnet101_verts20181223T1552.json"
  # ]

  result_files = [
    "assets/results/faster_rcnn_gemini_resnet50/faster_rcnn_gemini_resnet50_20181218T1240.json",
    "assets/results/faster_rcnn_gemini_resnet101/faster_rcnn_gemini_resnet101_20181219T1303.json",
    # "assets/experiments/20181222/gemini_resnet50_deltas/gemini_resnet50_deltas20181205T1744-448.json",
    # "assets/experiments/20181222/gemini_resnet101_deltas/gemini_resnet101_deltas20181205T2226-448.json",
    "assets/results/yolo_rbox_deltas/yolo_rbox_deltas_20181205T1953.json"
  ]

  # result_files = [
  #   "assets/experiments/20181222/gemini_resnet101_deltas/gemini_resnet101_deltas20181205T2226-448.json"
  # ]


  evaltype = "bbox"
  # evaltype = "segm"

  stats = []
  for filepath in result_files:
    dt_matches, dt_scores, overlaps, ious, gt_matches, fps = \
        evaluate(filepath, enable_adjustment=ENABLE_ADJUSTMENT, evaltype=evaltype)

    save_matches(filepath, dt_matches, dt_scores, overlaps, ious, evaltype=evaltype)

  #   # coco eval
  #   cocoEval = evalcoco(
  #       filepath,
  #       class_id=FILTER_CLSID,
  #       enable_adjustment=ENABLE_ADJUSTMENT,
  #       evaltype=evaltype)
  #   cocoEval.summarize()

  #   overlap, recall, AR = average_recall(ious)
  #   print("Average Recall: {}".format(AR))

  #   s = [
  #       fps, AR, cocoEval.stats[1], cocoEval.stats[2], cocoEval.stats[0],
  #       cocoEval.stats[8]
  #   ]
  #   stats += [s]

  # print("FPS\t AR\t AP.5\tAP.75\tAP:.95\tAR:.95")
  # for s in stats:
  #   print("{:0.3f}, {:0.3f}, {:0.3f}, {:0.3f}, {:0.3f}, {:0.3f}".format(
  #       s[0], s[1], s[2], s[3], s[4], s[5]))
