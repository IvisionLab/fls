#%%
# Plot matches results
import numpy as np
import matplotlib.pyplot as plt
from rboxnet.eval import  load_matches, compute_ap
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve


def plot_precision_recall(ax, filepath, cstyle, label=None, markevery=1):
  y_true, score, overlaps, _ = load_matches(filepath)

  y_true[np.where(overlaps <= 0.9)] = 0
  # precision, recall, mAP = compute_ap(y_true)

  # y_true[np.where(overlaps < 0.90)] = 0
  # ap = average_precision_score(y_true, score)
  # print("Average Precision: {}".format(ap))
  # precision, recall, thresholds = precision_recall_curve(y_true, score)

  ax.set_ylim(0, 1.05)
  ax.set_xlim(0, 1.05)
  ax.set_xlabel('Recall')
  ax.set_ylabel('Precision')
  ax.plot(recall, precision, cstyle, label=label, markevery=markevery)

#%%
# Plot original test dataset results

print("Original test dataset")
_, ax = plt.subplots()

ax.set_title("RESNET50-FPN")
# plot_precision_recall(
#     ax,
#     "assets/matches/gemini_resnet50_deltas20181205T1744_matches.csv",
#     label="Faster-RCNN/ROI distances",
#     cstyle="r")

# plot_precision_recall(
#     ax,
#     "assets/matches/gemini_resnet50_rotdim20181205T1807_matches.csv",
#     label="Faster-RCNN/Angles+Dimensions",
#     cstyle="b")

# plot_precision_recall(
#     ax,
#     "assets/matches/mask_rcnn_gemini_resnet50_20181207T1921_matches.csv",
#     label="Mask-RCNN/Segmentation based",
#     cstyle="g")

plot_precision_recall(
  ax,
  "assets/matches/hog_svm_jequitaia_20181211T2150_matches.csv",
  label="HOG+SVM",
  cstyle="g")

ax.legend()


plt.show()