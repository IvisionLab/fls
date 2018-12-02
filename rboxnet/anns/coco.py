import os
import cv2
import json
import numpy as np
from sklearn.utils import shuffle
from rboxnet.anns import common


################################
# Holder class
################################
class Holder:
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

  def addImgItem(self, file_name, size):
    if file_name is None:
      raise Exception('Could not find filename tag in xml file.')
    if size['width'] is None:
      raise Exception('Could not find width tag in xml file.')
    if size['height'] is None:
      raise Exception('Could not find height tag in xml file.')
    self.image_id += 1
    image_item = dict()
    image_item['id'] = self.image_id
    image_item['file_name'] = file_name
    image_item['width'] = size['width']
    image_item['height'] = size['height']
    self.coco['images'].append(image_item)
    self.image_set.add(file_name)
    return self.image_id

  def addAnnoItem(self, object_name, image_id, category_id, bbox, segm):
    annotation_item = dict()
    annotation_item['segmentation'] = []
    annotation_item['segmentation'].append(segm)
    annotation_item['area'] = bbox[2] * bbox[3]
    annotation_item['iscrowd'] = 0
    annotation_item['ignore'] = 0
    annotation_item['image_id'] = image_id
    annotation_item['bbox'] = bbox
    annotation_item['category_id'] = category_id
    self.annotation_id += 1
    annotation_item['id'] = self.annotation_id
    self.coco['annotations'].append(annotation_item)


def unpack_bbox(gt):
  gt = gt.astype(int)
  bbox = [int(gt[0]), int(gt[1]), int(gt[2] - gt[0]), int(gt[3] - gt[1])]

  return bbox


def unpack_rbbox(gt, bbox):
  rbbox = gt[4:]
  rbbox[0:2] += bbox[0:2]
  rbbox = common.rbox2points(rbbox).astype(int)
  polygon = []
  #left_top
  polygon.append(rbbox[0][0])
  polygon.append(rbbox[0][1])
  #left_bottom
  polygon.append(rbbox[1][0])
  polygon.append(rbbox[1][1])
  #right_bottom
  polygon.append(rbbox[2][0])
  polygon.append(rbbox[2][1])
  #right_top
  polygon.append(rbbox[3][0])
  polygon.append(rbbox[3][1])


def find_mask_polygon(mask_filepath):
  mask = cv2.imread(mask_filepath)
  mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
  _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)
  return contours[0].flatten().tolist()


def get_mask_filepath(img_file_path):
  path, filename = os.path.split(img_file_path)
  base_name = os.path.splitext(filename)[0]
  return os.path.join(path, base_name + '-mask.png')


def build_coco_annotations(base_folder, all_anns, labels, use_rbbox):
  holder = Holder()
  for anns in all_anns:
    file_name = anns['file_name']
    clsid = anns['clsid']
    gt = anns['gt']
    size = {}
    img_file_path = os.path.join(base_folder, file_name)
    img = cv2.imread(img_file_path)

    size['height'], size['width'], size['depth'] = img.shape
    object_name = labels[clsid]

    if object_name not in holder.category_set:
      current_category_id = holder.addCatItem(object_name)
    else:
      current_category_id = holder.category_set[object_name]

    if file_name not in holder.image_set:
      current_image_id = holder.addImgItem(file_name, size)
      print('add image with {} and {}'.format(file_name, size))
    else:
      raise Exception('duplicated image: {}'.format(file_name))

    bbox = unpack_bbox(gt.astype(np.int32))

    if use_rbbox:
      polygon = unpack_rbbox(gt, bbox)
    else:
      polygon = find_mask_polygon(get_mask_filepath(img_file_path))

    print('add annotation with {},{},{},{}'.format(
        object_name, current_image_id, current_category_id, bbox))
    holder.addAnnoItem(object_name, current_image_id, current_category_id,
                       bbox, polygon)

  return holder.coco


def generate(args):

  all_anns = common.list_files(args.base_folder, args.limit)
  coco_anns = build_coco_annotations(args.base_folder, all_anns,
                                     ["ssiv_bahia", "jequitaia", "balsa"],
                                     args.use_rbbox)

  json.dump(coco_anns, open("coco_annotations.json", 'w'))

  if args.split:
    shuffled_anns = shuffle(all_anns)
    n = int(0.8 * len(shuffled_anns))
    train_anns = shuffled_anns[:n]
    valid_anns = shuffled_anns[n:]
    coco_train_anns = build_coco_annotations(
        args.base_folder, train_anns, ["ssiv_bahia", "jequitaia", "balsa"],
        args.use_rbbox)

    json.dump(coco_anns, open("coco_annotations_train.json", 'w'))

    coco_valid_anns = build_coco_annotations(
        args.base_folder, valid_anns, ["ssiv_bahia", "jequitaia", "balsa"],
        args.use_rbbox)

    json.dump(coco_anns, open("coco_annotations_valid.json", 'w'))