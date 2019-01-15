#%%
import os
import numpy as np
from rboxnet.eval import average_recall, load_matches

total_elems = 8748

# base_dir = "assets/matches/segm"
base_dir = "assets/matches/bbox"

def accumulate(matches_dir, method):
  res = []
  for i, filename in enumerate(sorted(os.listdir(matches_dir))):
    filepath = os.path.join(matches_dir, filename)
    dts, scores, overlaps, ious = load_matches(filepath)
    tp = len(dts[np.where(dts == 1)])
    fp = len(dts[np.where(dts == 0)])
    fn = total_elems-tp

    res += [[tp, fp, fn]]

  res = np.vstack(res)

  row = []
  row = "{}, ".format(method)
  for i in range(res.shape[0]):
    for j in range(res.shape[1]):
      row += "{}, ".format(res[i, j])

  print(row)



for dirname in sorted(os.listdir(base_dir)):
  matches_dir = os.path.join(base_dir, dirname)

  if not os.path.isdir(matches_dir):
    continue

  accumulate(matches_dir, dirname)