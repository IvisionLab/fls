#!/usr/bin/python3

import os
import fnmatch
import cv2
import json
import numpy as np
from pycocotools.mask import encode, decode
from pycocotools.coco import COCO

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

labels = ["ssiv_bahia", "jequitaia", "balsa"]

class COCOAnnotationHolder:
    def __init__(self):
        self.coco = dict()
        self.coco['images'] = []
        self.coco['type'] = 'instances'
        self.coco['annotations'] = []
        self.coco['categories'] = []
        self.category_set = dict()
        self.image_set = set()
        self.category_item_id = 0
        self.annotation_id = 0
        self.image_id = 20180000000

    def addCatItem(self, name):
        category_item = dict()
        category_item['supercategory'] = 'none'
        self.category_item_id += 1
        category_item['id'] = self.category_item_id
        category_item['name'] = name
        self.coco['categories'].append(category_item)
        self.category_set[name] = self.category_item_id
        return self.category_item_id

    def addImgItem(self, file_name, coco_url, size):
        self.image_id += 1
        image_item = dict()
        image_item['id'] = self.image_id
        image_item['file_name'] = file_name
        image_item['coco_url'] = coco_url
        image_item['width'] = size[0]
        image_item['height'] = size[1]
        self.coco['images'].append(image_item)
        self.image_set.add(file_name)
        return self.image_id

    def addAnnoItem(self, object_name, image_id, category_id, bbox, seg):
        annotation_item = dict()
        annotation_item['segmentation'] = []
        annotation_item['segmentation'].append(seg)
        annotation_item['area'] = bbox[2] * bbox[3]
        annotation_item['iscrowd'] = 0
        annotation_item['ignore'] = 0
        annotation_item['image_id'] = image_id
        annotation_item['bbox'] = bbox
        annotation_item['category_id'] = category_id
        self.annotation_id += 1
        annotation_item['id'] = self.annotation_id
        self.coco['annotations'].append(annotation_item)

def build_coco_annotations(annotations):
    holder = COCOAnnotationHolder()
    for l in labels:
        holder.addCatItem(l)

    for elem in annotations:
        coco_url = os.path.join(elem['basefolder'], elem['filepath'])
        file_name = elem['filepath']
        current_image_id = None
        current_category_id = None
        object_name = None

        size = (elem['width'], elem['height'])
        if file_name not in holder.image_set:
            current_image_id = holder.addImgItem(file_name, coco_url, size)
            print('add image with {} and {}'.format(file_name, size))
        else:
            raise Exception('duplicated image: {}'.format(file_name))

        for a in elem['annotations']:
            label_id = a['id']
            object_name = labels[label_id]
            current_category_id = holder.category_set[object_name]

            bbox = a['bbox']
            seg = a['segmentation']
            holder.addAnnoItem(object_name, current_image_id, current_category_id, bbox, seg)

    return holder.coco

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Convert to COCO format")

    parser.add_argument('input_file',
                        help='Path to json file with the annotations')

    args = parser.parse_args()

    with open(args.input_file) as f:
        input_file = json.load(f)

    coco = build_coco_annotations(input_file)
    path, filename = os.path.split(args.input_file)
    base_name = os.path.splitext(filename)[0]
    filepath = os.path.join(path, "{}_coco.json".format(base_name))
    json.dump(coco, open(filepath, 'w'))
