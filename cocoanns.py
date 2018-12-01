#!/usr/bin/env python3
from rboxnet import anns

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(description="Generate coco annotations")

  parser.add_argument(
      "--base-folder",
      required=True,
      metavar="/path/to/dataset/base/folder",
      type=str,
      help="Folder containg the image dataset")

  parser.add_argument(
      "--use-rbbox",
      const=True,
      nargs='?',
      type=bool,
      metavar="<True/False>",
      default=False,
      help="Use rotated bounding box as mask")

  parser.add_argument(
      "--split",
      const=True,
      nargs='?',
      type=bool,
      metavar="<True/False>",
      default=False,
      help="Split dataset into train and validation")

  parser.add_argument(
    "--limit",
    required=False,
    type=int,
    default=-1,
    help="Maximum number of images per classes")

  args = parser.parse_args()

  anns.coco.generate(args)
