#%%
import csv
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

def load_stats(filepath):
  stats = []
  with open(filepath, 'r') as fs:
    reader = csv.reader(fs)
    for row in reader:
      row = [float(elem) for elem in row]
      stats += [row]

  stats = np.vstack(stats)
  return stats

methods = {
  "Mask-RCNN-50": {
    "filepath": "assets/experiments/20181222/mask_rcnn_gemini_resnet50/mask_rcnn_resnet50.csv",
    "style": "D-",
    "annotate": True,
    'color': 'darkblue'
  },
  "Mask-RCNN-101": {
    "filepath": "assets/experiments/20181222/mask_rcnn_gemini_resnet101/mask_rcnn_gemini_resnet101.csv",
    "style": "D-",
    "annotate": True,
    'color': 'sienna'
  },
  "YOLOv2+RBoxDNet-D": {
    "filepath": "assets/experiments/20181222/yolo_rbox_deltas/yolo_rbox_deltas.csv",
    "style": "o-",
    "annotate": True,
    'color': 'grey'
  },
  "YOLOv2+RBoxDNet-OS": {
    "filepath": "assets/experiments/20181222/yolo_rbox_rotdim/yolo_rbox_rotdim.csv",
    "style": "o-",
    "annotate": False,
    'color': 'tan'
  },
 "RBoxNet-101-D": {
    "filepath": "assets/experiments/20181222/gemini_resnet101_deltas/gemini_resnet101_deltas.csv",
    "style": "*-",
    "annotate": True,
    'color': 'orange'
  },
 "RBoxNet-101-OS": {
    "filepath": "assets/experiments/20181222/gemini_resnet101_rotdim/gemini_resnet101_rotdim.csv",
    "style": "*-",
    "annotate": False,
    'color': 'red'
  },
 "RBoxNet-50-D": {
    "filepath": "assets/experiments/20181222/gemini_resnet50_deltas/gemini_resnet50_deltas.csv",
    "style": ">-",
    "annotate": True,
    'color': 'blue'
  },
 "RBoxNet-50-OS": {
    "filepath": "assets/experiments/20181222/gemini_resnet50_rotdim/gemini_resnet50_rotdim.csv",
    "style": ">-",
    "annotate": False,
    'color': 'green'
  },
}

sizes = [256, 320, 384, 448]

custom_preamble = {
    "text.usetex": True,
    "text.latex.preamble": [
        r"\usepackage{array}",
        r"\usepackage{tabularx}",
        r"\newcolumntype{M}[1]{>{\centering\arraybackslash}m{#1}}",
        r"\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}"
        r"\newcolumntype{R}[1]{>{\raggedleft\arraybackslash}p{#1}}"
        ],
    }
plt.rcParams.update(custom_preamble)

all_fps = []
all_mAP = []
all_stats = []
names = []
for k, v in methods.items():
  stats = load_stats(v['filepath'])
  fps = 1/stats[:,0]*1000
  mAP = stats[:,4] * 100.0
  all_mAP += [np.mean(mAP)]
  all_fps += [round(np.mean(fps))]

  s = '{0}'.format(k)
  for i, v in enumerate(mAP):
    s = '{0},{1:.1f},{2:.1f}'.format(s, v, fps[i])
  print(s)

  names += [k]
  all_stats += [stats]

_, ax = plt.subplots(1, 1, figsize=(10.75, 5))

indices = np.argsort(-np.array(all_mAP))
bbox_args = dict(boxstyle="round", fc="1.0")

for idx in indices:
  stats = all_stats[idx]
  fps = 1/stats[:,0]*1000
  mAP = stats[:,4]
  ax.set_ylim(0.2, 0.95)
  ax.set_xlim(22.5, 175)
  ax.set_xlabel('average inference time (ms)', fontsize=18)
  ax.set_ylabel('mAP', fontsize=18)

  txt_ap = str("{0:.1f}".format(all_mAP[idx]))
  txt_fps = str("{0:.0f}".format(all_fps[idx]))
  txt_fps = "{0: <5}".format(txt_fps)

  label = r''
  label += r'\begin{tabular}{m{3.7cm}L{0.75cm}R{0.5cm}}'
  label += r''+str(names[idx])+' & '+ txt_ap +' & '+ txt_fps +' '
  label += r'\end{tabular}'

  data = methods[names[idx]]

  ax.plot(fps, mAP, data['style'], label=label, linewidth=2, markersize=10, color=data['color'])

  if data['annotate']:
    for i in range(len(mAP)):
      ax.annotate(str(sizes[i]), (fps[i], mAP[i]), xytext=(-27, 0), textcoords='offset points', bbox=bbox_args)

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(labelsize="12")

leg = plt.legend(loc=4)

title = r''
title += r'\begin{tabular}{m{0.75cm}m{3.75cm}m{0.75cm}m{0.75cm}} '
title += r' & \bfseries Method & \bfseries mAP & \bfseries time \\ '
title += r'\hline '
title += r'\end{tabular}'

leg.set_title(title, prop={'size':'large'})

plt.savefig('map_vs_time.eps', bbox_inches='tight')

plt.show()