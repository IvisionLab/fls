#%%
# Plot matches results
import os
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from scipy.interpolate import interp1d

# experiment dir

# exp_dir = "assets/tables/bbox"
xmin = 0.02
exp_dir = "assets/tables/segm"
# xmin = 0.0

# method_names = {
#     "gemini_resnet50_deltas": "RESNET50/Box Distances",
#     "gemini_resnet50_rotdim": "RESNET50/Angles+Dimensions",
#     "gemini_resnet101_deltas": "RESNET101/Box Distances",
#     "gemini_resnet101_rotdim": "RESNET101/Angles+Dimensions",
#     "mask_rcnn_gemini_resnet50": "RESNET50/Mask-RCNN",
#     "mask_rcnn_gemini_resnet101": "RESNET101/Mask-RCNN",
#     "yolo_rbox_deltas": "Darknet-19/YOLO/Box Distances",
#     "yolo_rbox_rotdim": "Darknet-19/YOLO/Angles+Distance"
# }

method_names = {
    "gemini_resnet50_deltas": "frcnn-res50-dbox",
    "gemini_resnet101_deltas": "frcnn-res101-dbox",
    "yolo_rbox_deltas": "yolo-dark19-dbox",
    "gemini_resnet50_rotdim": "frcnn-res50-obox",
    "gemini_resnet101_rotdim": "frcnn-res101-obox",
    "yolo_rbox_rotdim": "yolo-dark19-obox",
    "mask_rcnn_gemini_resnet50": "mrcnn-res50-segm",
    "mask_rcnn_gemini_resnet101": "mrcnn-res101-segm"
}

styles = {
    "gemini_resnet50_deltas": "d--",
    "gemini_resnet50_rotdim": "o--",
    "gemini_resnet101_deltas": "d--",
    "gemini_resnet101_rotdim": "o--",
    "mask_rcnn_gemini_resnet50": "s--",
    "mask_rcnn_gemini_resnet101": "s--",
    "yolo_rbox_deltas": "p--",
    "yolo_rbox_rotdim": "p--"
}


# load experiments
def load_exp(filepath):
  stats = []
  with open(filepath, 'r') as fs:
    reader = csv.reader(fs)
    for row in reader:
      row = [float(elem) for elem in row]
      ar, ap5, ap75, ap5_95, ar5_95 = row
      s = [ar, ap5, ap75, ap5_95, ar5_95]
      stats += [s]

  stats = np.vstack(stats)
  return stats


files = glob.glob(os.path.join(exp_dir, "*.csv"))

#%%
mstats = {}
for f in files:
  _, filename = os.path.split(f)
  base_name = os.path.splitext(filename)[0]
  stats = load_exp(f)
  mstats[base_name] = stats

keys = sorted(mstats.keys())

#%%
# Plot original test dataset results

xmax = 0.2
xstep = xmax / 10.0
x = np.arange(xmin, xmax + xstep, xstep)


def plot(methods, ax):
  table = []

  for name in methods:
    # if not name in methods:
    #   continue

    print(name)
    print("AR\t AP.5\tAP.75\tAP:.95\tAR:.95")
    for s in mstats[name]:

      print("{0:.3f}, {1:.3f}, {2:.3f}, {3:.3f}, {4:.3f}".format(
          s[0], s[1], s[2], s[3], s[4]))

    ap5_95 = mstats[name][:, 3]
    row = "{}, ".format(name)
    for i in range(ap5_95.shape[0]):
      row += "{:.3f},".format(ap5_95[i])

    row += "{:.3f}".format(np.mean(ap5_95))
    table += [row]

    ax.set_ylim(0.2, 1.1)
    ax.set_xlim(0.02, 0.2)
    ax.set_xlabel('Noise Level', fontsize=18)
    ax.set_ylabel('COCO mAP', fontsize=18)

    label = method_names[name]
    ax.plot(x, ap5_95, styles[name], label=label)
    # ax.legend(ncol=2)
    ax.legend(
        bbox_to_anchor=(0.0, 1.0, 1.0, 0.0),
        loc=2,
        ncol=4,
        mode="expand",
        prop={
            'family': 'monospace',
            'size': 12
        })

  for row in table:
    print(row)


# method_table = [
#   [
#     [
#       "gemini_resnet50_deltas",
#       "gemini_resnet50_rotdim",
#       "gemini_resnet101_deltas",
#       "gemini_resnet101_rotdim",
#       "mask_rcnn_gemini_resnet50",
#       "mask_rcnn_gemini_resnet101",
#       "yolo_rbox_deltas",
#       "yolo_rbox_rotdim"
#     ],
#     [
#       "gemini_resnet50_deltas",
#       "gemini_resnet50_rotdim",
#       "mask_rcnn_gemini_resnet50",
#     ]
#   ],
#   [
#     [
#       "gemini_resnet101_deltas",
#       "gemini_resnet101_rotdim",
#       "mask_rcnn_gemini_resnet101",
#     ],
#     [
#       "yolo_rbox_deltas",
#       "yolo_rbox_rotdim"
#     ]
#   ]
# ]

# methods = [
#   "mask_rcnn_gemini_resnet50",
#   "mask_rcnn_gemini_resnet101",
#   "yolo_rbox_deltas"
# ]

methods = [
    "gemini_resnet50_deltas",
    "gemini_resnet50_rotdim",
    "gemini_resnet101_deltas",
    "gemini_resnet101_rotdim",
    "yolo_rbox_deltas",
    "yolo_rbox_rotdim",
    "mask_rcnn_gemini_resnet50",
    "mask_rcnn_gemini_resnet101",
]

nrows = 1
ncols = 1
_, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 6))
plot(methods, axes)

# for i in range(2):
#   for j in range(2):
#     methods = method_table[i][j]
#     plot(methods, axes[i, j])

plt.show()