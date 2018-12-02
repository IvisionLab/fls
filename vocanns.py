#!/usr/bin/env python3
import argparse
from rboxnet import anns

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description="Generate coco annotations")

  parser.add_argument(
      "--base-folder",
      required=True,
      metavar="/path/to/dataset/base/folder",
      type=str,
      help="Folder containg the image dataset")

  parser.add_argument(
      "--output-folder",
      required=False,
      metavar="/path/to/annotation/output/folder",
      type=str,
      help="Path to output folder")


  parser.add_argument(
      "--split",
      const=True,
      nargs='?',
      type=bool,
      metavar="<True/False>",
      default=False,
      help="Split dataset into train and validation")

  parser.add_argument(
      "--test",
      const=True,
      nargs='?',
      type=bool,
      metavar="<True/False>",
      default=False,
      help="Generate in test folder")

  parser.add_argument(
      "--limit",
      required=False,
      type=int,
      default=-1,
      help="Maximum number of images per classes")

  args = parser.parse_args()

  anns.voc.generate(args)
