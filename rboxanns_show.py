#!/usr/bin/env python3
import sys
import json
import rboxnet.annotation.rbox as rboxanns

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(description="Load a json file with annotations")

  parser.add_argument("json_file",
                      help="The json file with annotations")

  args = parser.parse_args()

  with open(args.json_file) as f:
      dataset = json.load(f)
      print(len(dataset))