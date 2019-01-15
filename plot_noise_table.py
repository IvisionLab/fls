#%%
# Plot matches results
import os
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex":
    True,
    "text.latex.preamble": [
        r"\usepackage{array}",
        r"\usepackage{tabularx}",
        r"\newcolumntype{M}[1]{>{\centering\arraybackslash}m{#1}}",
        r"\newcolumntype{R}[1]{>{\raggedleft\arraybackslash}p{#1}}"
    ],
})

# methods list
methods = {
    "RBoxNet-101-D": {
        "filepath": "assets/tables/20181225/gemini_resnet101_deltas.csv",
        "style": "*-",
        'color': 'orange'
    },
    "RBoxNet-101-OS": {
        "filepath":"assets/tables/20181225/gemini_resnet101_rotdim.csv",
        "style":"*-",
        'color':'red'
    },
    "RBoxNet-50-D": {
        "filepath":"assets/tables/20181225/gemini_resnet50_deltas.csv",
        "style":">-",
        'color':'blue'
    },
    "RBoxNet-50-OS": {
        "filepath":"assets/tables/20181225/gemini_resnet50_rotdim.csv",
        "style":">-",
        'color':'green'
    },
    "YOLOv2+RBoxDNet-D": {
        "filepath": "assets/tables/segm/yolo_rbox_deltas.csv",
        "style": "o-",
        'color': 'grey'
    },
    "YOLOv2+RBoxDNet-OS": {
        "filepath": "assets/tables/segm/yolo_rbox_rotdim.csv",
        "style": "o-",
        'color': 'tan'
    },
    "Mask-RCNN-50": {
        "filepath": "assets/tables/segm/mask_rcnn_gemini_resnet50.csv",
        "style": "D-",
        'color': 'darkblue'
    },
    "Mask-RCNN-101": {
        "filepath": "assets/tables/segm/mask_rcnn_gemini_resnet101.csv",
        "style": "D-",
        'color': 'sienna'
    },
}

def set_legend(tabledef='m{4.25cm}R{0.75cm}', makered=None, legend_out=False):

  if legend_out:
    l = plt.legend(
      loc=3,
      ncol=1,
      prop={'size': 11})
  else:
    l = plt.legend(
      ncol=1,
      prop={'size': 11})

  title = r''
  title += r'\begin{tabular}{m{0.5cm}'+tabledef+'} '
  title += r' & \bfseries Method & \bfseries mAP \\ '
  title += r'\hline '
  title += r'\end{tabular}'
  l.set_title(title, prop={'size':'large'})

  if not makered is None:
    texts = l.get_texts()
    for i in makered:
      texts[i].set_color('red')

def plot_map_noise(names, title=None, tabledef="m{4.25cm}R{0.5cm}"):
  # _, ax = plt.subplots(1, 1, figsize=(5,4))
  _, ax = plt.subplots(1, 1)
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  ax.tick_params(labelsize="12")
  ax.set_ylim(0.1, 1.05)
  ax.set_xlim(0.015, 0.105)
  # ax.set_xlim(0.015, 0.205)
  ax.set_xlabel('level of additive noise', fontsize=18)
  ax.set_ylabel('mAP', fontsize=18)
  if not title is None:
    ax.set_title(title, fontsize=14)

  xmin = 0.02
  xmax = 0.10
  xstep = xmax / 5.0
  # xmax = 0.20
  # xstep = xmax / 10.0
  xvals = np.arange(xmin, xmax+xstep, xstep)

  for i, name in enumerate(names):
    data = methods[name]
    stats = load_stats(data['filepath'])
    print("AR\t AP.5\tAP.75\tAP:.95\tAR:.95")
    for s in stats:
      print("{0:.3f}, {1:.3f}, {2:.3f}, {3:.3f}, {4:.3f}".format(
          s[0], s[1], s[2], s[3], s[4]))
    # mAPs = stats[:, 3]
    mAPs = stats[0:5, 3]
    m = np.mean(mAPs) * 100

    label = r''

    label += r'\begin{tabular}{'+tabledef+'}'
    if i == 0:
      label += r'\textbf{' + str(name) + r'} & \textbf{' + str("{:.1f}").format(m) + r'} '
    else:
      label += r'' + str(name) + r' & ' + str("{:.1f}").format(m) + r' '
    label += r'\end{tabular}'

    ax.plot(xvals, mAPs, data['style'], color=data['color'], label=label, linewidth=2, markersize=10)

# load stats
def load_stats(filepath):
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

def print_map_noise(names):
  for i, name in enumerate(names):
    data = methods[name]
    stats = load_stats(data['filepath'])
    mAPs = stats[0:5, 3] * 100
    text = ""
    for m in mAPs:
      text = "{0}{1:.1f} & ".format(text, m)
    print("{0} & {1}{2:.1f} & {3:.1f} \\\\".format(name, text, np.mean(mAPs), np.std(mAPs)))

print_map_noise([
  "RBoxNet-101-D",
  "RBoxNet-50-D",
  "YOLOv2+RBoxDNet-D",
  "RBoxNet-101-OS",
  "YOLOv2+RBoxDNet-OS",
  "RBoxNet-50-OS",
  "Mask-RCNN-101",
  "Mask-RCNN-50"])

# plot_map_noise(["RBoxNet-101-D", "RBoxNet-101-OS", "RBoxNet-50-D", "RBoxNet-50-OS"])
# set_legend()

# plot_map_noise(["RBoxNet-101-D", "RBoxNet-50-D", "YOLOv2+RBoxDNet-D"], title="Box Offset")
# set_legend()
# plt.savefig('noise_level_deltas.eps', bbox_inches='tight', dpi=300)

# plot_map_noise(["RBoxNet-101-OS", "RBoxNet-50-OS", "YOLOv2+RBoxDNet-OS"], title="Orientation and Size")
# set_legend()
# plt.savefig('noise_level_rotdim.eps', bbox_inches='tight', dpi=300)

# plot_map_noise(["Mask-RCNN-101", "Mask-RCNN-50"], title="Segmentation Mask")
# set_legend()
# plt.savefig('noise_level_segm.eps', bbox_inches='tight', dpi=300)

# plot_map_noise(["RBoxNet-101-D", "RBoxNet-101-OS", "Mask-RCNN-101"], title="Comparison of the top-3 methods")
# set_legend()
# plt.savefig('noise_level_best.eps', bbox_inches='tight', dpi=300)

# plot_map_noise(["YOLOv2+RBoxDNet-D", "YOLOv2+RBoxDNet-OS"])
# set_legend()

# plot_map_noise(["RBoxNet-101-D", "RBoxNet-50-D", "RBoxNet-101-OS", "RBoxNet-50-OS"])
# set_legend()

# plot_map_noise(["RBoxNet-50-D", "RBoxNet-50-OS", "Mask-RCNN-50"])
# set_legend()

# plot_map_noise(["RBoxNet-101-D", "RBoxNet-101-OS", "Mask-RCNN-101", "RBoxNet-50-D", "RBoxNet-50-OS", "Mask-RCNN-50", "YOLOv2+RBoxDNet-D", "YOLOv2+RBoxDNet-OS"])
# set_legend()


plt.show()