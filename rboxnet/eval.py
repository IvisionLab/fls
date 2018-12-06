import cv2
import numpy as np
import matplotlib.pyplot as plt

# create mask from rodated bounding-box
def rotated_box_mask(shape, rotated_boxes):
  mask = np.zeros(shape, dtype=np.uint8)
  for rbox in rotated_boxes:
    rbox = np.reshape(rbox, (-1, 2))
    rbox = rbox.astype(int)
    cv2.fillConvexPoly(mask, rbox, 255)
  return mask


# calculate the IoU between two mask
def calc_iou_from_masks(gt, dt):
  inter = np.zeros(gt.shape, dtype=np.uint8)
  cv2.bitwise_and(gt, dt, inter)
  inter_area = np.sum(inter) / 255.
  gt_area = np.sum(gt) / 255.
  dt_area = np.sum(dt) / 255.
  return inter_area / (gt_area + dt_area - inter_area)


# calculate IoU between annotations and detections
def calc_ious(annotations, detections, image_shape, nb_class):
  ious = []
  for cls_id in range(nb_class):
    dts = [dt for dt in detections if dt['id'] == cls_id]
    gts = [gt for gt in annotations if gt['id'] == cls_id]

    if not dts and not gts:
      continue

    dt_mask = rotated_box_mask(
        image_shape,
        [dt['rbox'] for dt in dts if not dt is None and len(dt) > 0])

    gt_mask = rotated_box_mask(
        image_shape,
        [gt['rbox'] for gt in gts if not gt is None and len(gt) > 0])

    iou = calc_iou_from_masks(gt_mask, dt_mask)
    ious = [iou]
  return ious


# calculate average recall based on IoU
def average_recall(iou):
  all_iou = sorted(iou)
  num_pos = len(all_iou)
  dx = 0.001

  overlap = np.arange(0, 1, dx)
  overlap[-1] = 1

  N = len(overlap)
  recall = np.zeros(N, dtype=np.float32)
  for i in range(N):
    recall[i] = (all_iou > overlap[i]).sum() / float(num_pos)

  good_recall = recall[np.where(overlap > 0.5)]
  AR = 2 * dx * np.trapz(good_recall)
  return overlap, recall, AR

def plot_average_recall(ious):
  overlap, recall, AR = average_recall(ious)
  print("Average Recall: {}".format(AR))
  plt.figure()
  plt.step(overlap, recall, color='b', alpha=0.2, where='post')
  plt.fill_between(overlap, recall, step='post', alpha=0.2, color='b')
  plt.xlabel('IoU')
  plt.ylabel('Recall')
  plt.ylim([0.0, 1.05])
  plt.xlim([0.0, 1.0])
  plt.title('Recall-IoU curve: AR={0:0.5f}'.format(AR))
  plt.show()