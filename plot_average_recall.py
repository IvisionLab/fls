#%%
# Plot precision recall
import matplotlib.pyplot as plt
from rboxnet.eval import average_recall, load_matches

def plot_average_recall(ax, filepath, cstyle, label=None, markevery=1):
  _, _, _, ious = load_matches(filepath)
  overlap, recall, AR = average_recall(ious)

  ax.set_ylim(0, 1.05)
  ax.set_xlim(0, 1.05)
  ax.set_xlabel('IoU')
  ax.set_ylabel('Recall')
  ax.plot(overlap, recall, cstyle, label=label, markevery=markevery)
  print("Average Recall: {}".format(AR))


#%%
# Plot original test dataset results

print("Original test dataset")
_, ax = plt.subplots()

ax.set_title("RESNET50-FPN")
plot_average_recall(
    ax,
    "assets/matches/gemini_resnet50_deltas20181205T1744_matches.csv",
    label="Faster-RCNN/ROI distances",
    cstyle="r")

plot_average_recall(
    ax,
    "assets/matches/gemini_resnet50_rotdim20181205T1807_matches.csv",
    label="Faster-RCNN/Angles+Dimensions",
    cstyle="b")

plot_average_recall(
    ax,
    "assets/matches/mask_rcnn_gemini_resnet50_20181207T1921_matches.csv",
    label="Mask-RCNN/Segmentation based",
    cstyle="g")
ax.legend()

_, ax = plt.subplots()

ax.set_title("RESNET50-FPN")

plot_average_recall(
    ax,
    "assets/matches/gemini_resnet101_deltas20181205T2226_matches.csv",
    label="Faster-RCNN/ROI distances",
    cstyle="r")

plot_average_recall(
    ax,
    "assets/matches/gemini_resnet101_rotdim20181205T2248_matches.csv",
    label="Faster-RCNN/Angles+Dimensions",
    cstyle="g")

plot_average_recall(
    ax,
    "assets/matches/mask_rcnn_gemini_resnet101_20181206T1858_matches.csv",
    label="Mask-RCNN/Segmentation based",
    cstyle="b")

ax.legend()

_, ax = plt.subplots()

ax.set_title("Darknet-19")

plot_average_recall(
    ax,
    "assets/matches/yolo_rbox_deltas_20181205T1953_matches.csv",
    label="YOLO/ROI distances",
    cstyle="r")

plot_average_recall(
    ax,
    "assets/matches/yolo_rbox_rotdim20181211T1954_matches.csv",
    label="YOLO/Angles+Dimensions",
    cstyle="g")

ax.legend()

#%%
# Plot modified test dataset results
print("Modified test dataset")

_, ax = plt.subplots()

ax.set_title("RESNET50-FPN")
plot_average_recall(
    ax,
    "assets/matches/gemini_resnet50_deltas_imgaug20181208T1820_matches.csv",
    label="Faster-RCNN/ROI distances",
    cstyle="r")

plot_average_recall(
    ax,
    "assets/matches/gemini_resnet50_rotdim_imgaug20181208T2138_matches.csv",
    label="Faster-RCNN/Angles+Dimensions",
    cstyle="b")

plot_average_recall(
    ax,
    "assets/matches/mask_rcnn_gemini_resnet50_imgaug20181208T1957_matches.csv",
    label="Mask-RCNN/Segmentation based",
    cstyle="g")
ax.legend()

_, ax = plt.subplots()

ax.set_title("RESNET101-FPN")
plot_average_recall(
    ax,
    "assets/matches/gemini_resnet101_deltas_imgaug20181208T1901_matches.csv",
    label="Faster-RCNN/ROI distances",
    cstyle="r")

plot_average_recall(
    ax,
    "assets/matches/gemini_resnet101_rotdim_imgaug20181208T1616_matches.csv",
    label="Faster-RCNN/Angles+Dimensions",
    cstyle="b")

plot_average_recall(
    ax,
    "assets/matches/mask_rcnn_gemini_resnet101_imgaug20181208T2029_matches.csv",
    label="Mask-RCNN/Segmentation based",
    cstyle="g")
ax.legend()

_, ax = plt.subplots()

ax.set_title("Darknet-19")

plot_average_recall(
    ax,
    "assets/matches/yolo_rbox_deltas_imgaug20181208T2054_matches.csv",
    label="YOLO/ROI distances",
    cstyle="r")

plot_average_recall(
    ax,
    "assets/matches/yolo_rbox_rotdim_imgaug20181211T1942_matches.csv",
    label="YOLO/Angles+Dimensions",
    cstyle="g")

ax.legend()

#%%
# Plot modified test dataset results
print("HOG+SVM")
_, ax = plt.subplots()

plot_average_recall(
    ax,
    "assets/matches/hog_svm_all_20181212T0952_matches.csv",
    label="HOG+SVM",
    cstyle="r")

ax.legend()

plt.show()