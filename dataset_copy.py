#!/usr/bin/env python3
import os
import sys
import argparse
import fnmatch
import shutil

def get_paths(root, name):
  base_name = os.path.splitext(name)[0]
  imgpath = os.path.join(root, name)
  annpath = os.path.join(root, "{0}.txt".format(base_name))
  mskpath = os.path.join(root, "{0}-mask.png".format(base_name))
  return imgpath, annpath, mskpath

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description="copy and rename files from dataset")

  parser.add_argument("input",
                      help="The path to a folder with images and annotations")

  parser.add_argument("output",
                      help="The path to a folder with images and annotations")

  args = parser.parse_args()
  input_folder = args.input
  output_folder = args.output

  if not os.path.exists(output_folder):
    os.makedirs(output_folder)

  cnt = 1
  for root, dirs, files in os.walk(input_folder):
    files = fnmatch.filter(files, '*.png')
    files = fnmatch.filter(files, '*[!-mask].png')
    files = sorted(files)
    for src_name in files:
      dst_name = "sample-{0}.png".format(str(cnt).zfill(7))
      src_imgpath, src_annpath, src_mskpath = get_paths(root, src_name)
      dst_imgpath, dst_annpath, dst_mskpath = get_paths(output_folder, dst_name)

      print("Copying {0} -> {1}".format(src_imgpath, dst_imgpath))
      shutil.copyfile(src_imgpath, dst_imgpath)
      print("Copying {0} -> {1}".format(src_annpath, dst_annpath))
      shutil.copyfile(src_annpath, dst_annpath)
      print("Copying {0} -> {1}".format(src_mskpath, dst_mskpath))
      shutil.copyfile(src_mskpath, dst_mskpath)
      cnt = cnt + 1
