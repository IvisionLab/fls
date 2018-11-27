#!/usr/bin/python3
import json
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt



def calc_iou(gt, dt):
    inter = np.zeros(gt.shape, dtype=np.uint8)
    cv2.bitwise_and(gt, dt, inter)
    inter_area = np.sum(inter)/255.
    gt_area = np.sum(gt)/255.
    dt_area = np.sum(dt)/255.
    return inter_area / (gt_area+dt_area-inter_area)

def average_recall(iou):
    all_iou= sorted(iou)
    num_pos = len(all_iou)
    dx = 0.001

    overlap = np.arange(0, 1, dx)
    overlap[-1]= 1

    N = len(overlap)
    recall = np.zeros(N, dtype=np.float32)
    for i in range(N):
        recall[i] = (all_iou > overlap[i]).sum() / float(num_pos)

    good_recall = recall[np.where(overlap > 0.5)]
    AR = 2 * dx * np.trapz(good_recall)
    return overlap, recall, AR

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Evaluate results.')

    parser.add_argument('--annotations', required=True,
                        metavar="/path/to/annotations",
                        help='Path to json file with the annotations')

    parser.add_argument('--results', required=True,
                        metavar="/path/to/results",
                        help='Path to json file with the results')

    args = parser.parse_args()

    annotations = []
    with open(args.annotations) as f:
        annotations = json.load(f)

    results = []
    with open(args.results) as f:
        results = json.load(f)

    iou = []
    for result_item in results:
        img_w = result_item['width']
        img_h = result_item['height']
        dts = result_item['detections']

        N = 3
        for cls_id in range(N):
            dts = [dt for dt in result_item['detections'] if dt['class_id'] == cls_id]
            gts = [gt for gt in result_item['ground_truths'] if gt['id'] == cls_id]

            dt_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            for dt in dts:
                dt_rbox = dt["rbox"]
                dt_rbox = np.reshape(dt_rbox, (-1, 2))
                dt_rbox = dt_rbox[:,[1,0]]
                cv2.fillConvexPoly(dt_mask, dt_rbox, 255)

            gt_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            for gt in gts:
                gt_rbox = gt["segmentation"]
                gt_rbox = np.reshape(gt_rbox, (-1, 2))
                cv2.fillConvexPoly(gt_mask, gt_rbox, 255)

            if not dts and not gts:
                continue

            iou += [calc_iou(gt_mask, dt_mask)]


    overlap, recall, AR = average_recall(iou)
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