#%%
# Evaluate results using COCO
import os
import numpy as np
import matplotlib.pyplot as plt
from rboxnet.eval import average_recall
from evalcoco import evalcoco
from evaluate import evaluate, save_matches

# experiments directory
base_dir = "assets/experiments/20181221"

def save_stats(exp_dir, stats):
  lines = []

  for s in stats:
    line = ""
    for v in s[:-1]:
      line += "{:.3f}, ".format(v)
    line += "{:.3f}\n".format(s[-1])
    lines += [line]

  filename = exp_dir+".csv"
  print("Saving stats: ", filename)
  with open(filename, 'w') as fs:
    fs.writelines(lines)

def evalexp(exp_dir):
  stats = []
  cnt = 0
  _, ax = plt.subplots()
  _, dirname = os.path.split(exp_dir)
  ax.set_title(dirname)

  evaltype = "segm"

  for i, filename in enumerate(sorted(os.listdir(exp_dir))):
    filepath = os.path.join(exp_dir, filename)
    dts, scores, overlaps, ious, _, _= evaluate(filepath, evaltype=evaltype)
    save_matches(filepath, dts, scores, overlaps, ious, evaltype=evaltype)
    cocoEval = evalcoco(filepath, evaltype=evaltype)
    cocoEval.summarize()
    overlap, recall, AR = average_recall(ious)
    s = [AR, cocoEval.stats[1], cocoEval.stats[2], cocoEval.stats[0], cocoEval.stats[8]]
    stats += [s]

    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('IoU')
    ax.set_ylabel('Recall')
    ax.plot(overlap, recall)

  if not stats == []:
    print("AR\t AP.5\tAP.75\tAP:.95\tAR:.95")
    for s in stats:
      print("{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}".format(
          s[0], s[1], s[2], s[3], s[4]))

  return stats

# list experiments folders
for dirname in sorted(os.listdir(base_dir)):
  exp_dir = os.path.join(base_dir, dirname)

  if not os.path.isdir(exp_dir):
    continue

  print("Loading experiments from: ", exp_dir)
  stats = evalexp(exp_dir)
  if not stats == []:
    save_stats(exp_dir, stats)

plt.show()