#!/usr/bin/env python3
import sys
import rboxnet.annotation.rbox as rboxanns

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Generate a json file containing annotations")

    parser.add_argument("dataset_folder",
                        help="The path to a folder with images and annotations")

    parser.add_argument("--output",
                        default="annotations.json",
                        help="The output filepath")

    parser.add_argument("--validation",
                        type=bool,
                        default=False,
                        help="Save validation dataset")

    parser.add_argument("--limit",
                        type=int,
                        help="Maximum number of images per classes")

    parser.add_argument("--rbox_mask",
                        type=bool,
                        default=True,
                        help="Save annotation mask")

    args = parser.parse_args()

    rboxanns.exec(args)
