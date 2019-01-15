#%%
# Plot precision recall
import os
import matplotlib
import matplotlib.pyplot as plt
from rboxnet.eval import average_recall, load_matches

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
    "Mask-RCNN-50": {
        "matches": "assets/matches/segm/mask_rcnn_gemini_resnet50/mask_rcnn_gemini_resnet50_0.000_20181207T1921_matches.csv",
        "style": "D-",
        'color': 'darkblue'
    },
    "Mask-RCNN-101": {
        "matches": "assets/matches/segm/mask_rcnn_gemini_resnet101/mask_rcnn_gemini_resnet101_0.000_20181206T1858_matches.csv",
        "style": "D-",
        'color': 'sienna'
    },
    "YOLOv2+RBoxDNet-D": {
        "matches": "assets/matches/segm/yolo_rbox_deltas/yolo_rbox_deltas_0.000_20181205T1953_matches.csv",
        "style": "o-",
        'color': 'grey'
    },
    "YOLOv2+RBoxDNet-OS": {
        "matches": "assets/matches/segm/yolo_rbox_rotdim/yolo_rbox_rotdim_0.000_20181211T1954_matches.csv",
        "style": "o-",
        'color': 'tan'
    },
    "RBoxNet-101-D": {
        "matches": "assets/matches/segm/gemini_resnet101_deltas/gemini_resnet101_deltas_0.000_20181205T2226_matches.csv",
        "style": "*-",
        'color': 'orange'
    },
    "RBoxNet-101-OS": {
        "matches":"assets/matches/segm/gemini_resnet101_rotdim/gemini_resnet101_rotdim_0.000_20181205T2248_matches.csv",
        "style":"*-",
        'color':'red'
    },
    "RBoxNet-50-D": {
        "matches":"assets/matches/segm/gemini_resnet50_deltas/gemini_resnet50_deltas_0.000_20181205T1744_matches.csv",
        "style":">-",
        'color':'blue'
    },
    "RBoxNet-50-OS": {
        "matches":"assets/matches/segm/gemini_resnet50_rotdim/gemini_resnet50_rotdim_0.000_20181205T1807_matches.csv",
        "style":">-",
        'color':'green'
    },
    "RBoxNet-50-V": {
        "matches": "assets/matches/segm/gemini_resnet50_verts/gemini_resnet50_verts20181223T1522_matches.csv",
        "style": "<-.",
        'color': 'black'
      },
    "RBoxNet-101-V": {
        "matches": "assets/matches/segm/gemini_resnet101_verts/gemini_resnet101_verts20181223T1552_matches.csv",
        "style": "H-.",
        'color': 'teal'
      },
    "HOG+SVM": {
        "matches":"assets/matches/segm/hog_svm/hog_svm_all_20181212T0952_matches.csv",
        "style":"H-.",
        'color':'magenta'
    },
}

# deltas methods
methods_deltas = ["RBoxNet-101-D", "RBoxNet-50-D", "YOLOv2+RBoxDNet-D"]

# rotdim methods
methods_rotdim = ["RBoxNet-101-OS", "RBoxNet-50-OS", "YOLOv2+RBoxDNet-OS"]

# verts methods
methods_verts = ["RBoxNet-101-V", "RBoxNet-50-V", "HOG+SVM"]

# best results
methods_best = ["RBoxNet-101-D", "RBoxNet-101-OS", "Mask-RCNN-101", "Mask-RCNN-50"]

def plot_average_recall(filepath, ax, cstyle, method, tabledef="m{4.25cm}R{0.5cm}", color=None, bold=False):
  _, _, _, ious = load_matches(filepath)
  overlap, recall, AR = average_recall(ious)

  AR *= 100.0

  label = r''
  label += r'\begin{tabular}{'+tabledef+'}'
  if bold:
    label += r'\textbf{' + str(method) + r'} & \textbf{' + str("{:.1f}").format(AR) + r'} '
  else:
    label += r'' + str(method) + r' & ' + str("{:.1f}").format(AR) + r' '
  label += r'\end{tabular}'

  ax.set_ylim(0, 1.05)
  ax.set_xlim(0.5, 1.0)
  ax.set_xlabel('IoU', fontsize=18)
  ax.set_ylabel('Recall', fontsize=18)
  ax.plot(overlap, recall, cstyle, label=label, color=color, markevery=50)
  print("{0:<20} (AR={1:.2f})".format(method, AR))

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
  title += r' & \bfseries Method & \bfseries AR \\ '
  title += r'\hline '
  title += r'\end{tabular}'
  l.set_title(title, prop={'size':'large'})

  if not makered is None:
    texts = l.get_texts()
    for i in makered:
      texts[i].set_color('red')

def setup_plot(names, title=None, figsize=None, tabledef="m{4.25cm}R{0.75cm}", makebold=None):
  fig, ax = plt.subplots(1, 1, figsize=figsize)
  for i, name in enumerate(names):
    matchesfile = methods[name]['matches']
    style = methods[name]['style']
    color = methods[name]['color']
    bold = True if not makebold is None and i in makebold else False
    plot_average_recall(
      matchesfile, ax, style,
      method=name, color=color, tabledef=tabledef, bold=bold)

  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  ax.tick_params(labelsize="12")

  if not title is None:
    ax.set_title(title, fontsize=14)

#%% Plot Box Offsets
setup_plot(methods_deltas, figsize=(6, 4), title="Box Offset comparison", makebold=[0])
set_legend()
plt.savefig('recall_iou_deltas.eps', bbox_inches='tight')

# Plot Orientation and Size
setup_plot(methods_rotdim, figsize=(6, 4), title="Orientation and Size comparison", makebold=[0])
set_legend()
plt.savefig('recall_iou_rotdim.eps', bbox_inches='tight')

# HOG+SVM
setup_plot(methods_verts, figsize=(6, 4), title="Vertices vs. HOG+SVM", tabledef='m{2.75cm}R{0.75cm}')
set_legend(tabledef='m{2.75cm}R{0.75cm}', legend_out=True)
plt.savefig('recall_iou_verts.eps', bbox_inches='tight')

# Mask-RCNN-50 / Mask-RCNN-101
setup_plot(methods_best, figsize=(6, 4), tabledef='m{3.15cm}R{0.75cm}', title="Best RboxNets vs. Mask-RCNN", makebold=[0, 1])
set_legend(tabledef='m{3.15cm}R{0.75cm}')
plt.savefig('recall_iou_bests.eps', bbox_inches='tight')
plt.tight_layout()
plt.show()
