import cv2
import numpy as np
from rboxnet.anns.common import vert2points


def draw_verts(im, verts, colors=(255, 0, 0)):
  pts = vert2points(verts)
  n = len(pts)
  for i in range(n):
    pt1 = tuple(np.round(pts[i % n]).astype(int))
    pt2 = tuple(np.round(pts[(i + 1) % n]).astype(int))
    cv2.line(im, pt1, pt2, colors, 5)


def draw_rotated_boxes(im, rboxes_verts, colors=(255, 0, 0)):
    for verts in rboxes_verts:
      draw_verts(im, verts, colors)


def draw_boxes(image, boxes, class_ids, scores, labels, only_label=False):
  image_h, image_w, _ = image.shape

  for i, box in enumerate(boxes):
    x1, y1, x2, y2 = box

    if not only_label:
      cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    label = labels[class_ids[i]]
    score = scores[i]
    cv2.putText(image,
                label + ' ' + str(score),
                (x1, y1 - 13),
                cv2.FONT_HERSHEY_SIMPLEX,
                1e-3 * image_h,
                (0,255,0), 2)

  return image
