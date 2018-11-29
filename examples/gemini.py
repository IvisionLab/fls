#!/usr/bin/python3
import os
import fnmatch
import sys
import json
import random
import cv2
import numpy as np
import time
from rboxnet import utils
from rboxnet import model as modellib
from rboxnet.config import Config
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

ROOT_DIR = os.getcwd()
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

class GeminiConfig(Config):
    NAME = "gemini"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4
    NUM_CLASSES = 1 + 3
    IMAGE_MAX_DIM = 448
    BACKBONE = "resnet50"

class InferenceConfig(GeminiConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0

class CocoDataset(utils.Dataset):
    def load_coco(self, anns_file):
        coco = COCO(anns_file)

        # All classes
        class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        random.shuffle(image_ids)

        # Add classes
        for i in class_ids:
            self.add_class("gemini", i, coco.loadCats(i)[0]["name"])
        print(self.class_info)

        # Add images
        for i in image_ids:
            self.add_image(
                "gemini",
                image_id=i,
                path=coco.imgs[i]['coco_url'],
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(imgIds=[i], catIds=class_ids, iscrowd=None)))
        return coco

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "gemini":
            return super(CocoDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "gemini.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(CocoDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "gemini":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(CocoDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

class GeminiDataset(utils.Dataset):
    LABELS = ["ssiv_bahia", "jequitaia", "balsa"]
    def load_gemini(self, dataset):
        for i in range(len(GeminiDataset.LABELS)):
            self.add_class('gemini', i, GeminiDataset.LABELS[i])

        print("Total images: {}".format(len(dataset)))

        labels_count = np.zeros(len(self.LABELS), dtype=np.int32)

        for item in dataset:
            image_path = os.path.join(item['basefolder'], item['filepath'])
            self.add_image(
                "gemini",
                image_id=item['filepath'],
                path=image_path,
                width=item['width'],
                height=item['height'],
                annotations=item['annotations'])

            for a in item['annotations']:
                labels_count[a['id']] += 1

        for i, c in enumerate(labels_count):
            print("{} - {}".format(self.LABELS[i], c))

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info["source"] != "gemini":
            return super(GeminiDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        image_info = self.image_info[image_id]
        for a in image_info['annotations']:
            h = image_info["height"]
            w = image_info["width"]
            m = self.annToMask(a, h, w)
            instance_masks.append(m)
            class_id = self.map_source_class_id("gemini.{}".format(a['id']))
            class_ids.append(class_id)

        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            return super(GeminiDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "gemini":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

    def annToRLE(self, ann, height, width):
        segm = ann['segmentation']
        if isinstance(segm, list):
            rles = maskUtils.frPyObjects([segm], height, width)
            rle = maskUtils.merge(rles)
            return rle
        return None

    def annToMask(self, ann, height, width):
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

def load_dataset_json(filepath, shuffle=False):
    with open(filepath) as f:
        annotations = json.load(f)
        if shuffle:
            random.shuffle(annotations)
        return annotations

def load_dataset_splitted(filepath):

    dataset = load_dataset_json(filepath, shuffle=True)
    n = int(0.8*len(dataset))

    dataset_train = GeminiDataset()
    dataset_train.load_gemini(dataset[:n])
    dataset_train.prepare()

    dataset_val = GeminiDataset()
    dataset_val.load_gemini(dataset[n:])
    dataset_val.prepare()

    return dataset_train, dataset_val

def train(model, config, train_annotations, valid_annotations=None):

    if valid_annotations == None:
        print("Load dataset and split")
        dataset_train, dataset_val = load_dataset_splitted(train_annotations)
    else:
        print("Load training dataset...")

        dataset_train = GeminiDataset()
        dataset_train.load_gemini(load_dataset_json(train_annotations, shuffle=True))
        dataset_train.prepare()

        print("Load validation dataset...")
        dataset_val = GeminiDataset()
        dataset_val.load_gemini(load_dataset_json(valid_annotations, shuffle=True))
        dataset_val.prepare()

    # train_generator = modellib.data_generator(
    #     dataset_train, config, shuffle=False, batch_size=config.BATCH_SIZE, augment=False)

    # for inputs, output in train_generator:
    #     # print("inputs")
    #     img = (inputs[0][0]+config.MEAN_PIXEL) / 255.0
    #     print("rpn_match: {}".format(inputs[2][0].shape)) # rpn_match
    #     print("rpn_bbox: {}".format(inputs[3][0].shape)) # rpn_bbox
    #     for i in range(len(inputs[4][0])):
    #         if inputs[4][0][i] > 0:
    #             gt_id = inputs[4][0][i]
    #             gt_box = inputs[5][0][i]
    #             gt_rbox = inputs[6][0][i]
    #             gt_rbox = np.reshape(gt_rbox, (-1, 2))
    #             p = [1, 0]
    #             gt_rbox = gt_rbox[:,p]
    #             cv2.drawContours(img, [gt_rbox], 0, (0,0,255), 3)

    #             print("gt_class_id: {}".format(gt_id))
    #             print("gt_boxes: {}".format(gt_box))

    #             cv2.rectangle(img, (gt_box[1], gt_box[0]), (gt_box[3], gt_box[2]), (0,255,0), 2)

    #             for rb in gt_rbox:
    #                 cv2.circle(img, (rb[0], rb[1]) , 5, (255, 0, 0), 3)

    #             cv2.imshow("img", img)
    #             if cv2.waitKey(100) & 0xFF == ord('q'):
    #                 exit()

    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                layers='heads')

    # print("Fine tune Resnet stage 4 and up")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE,
    #             epochs=120,
    #             layers='4+')

    # print("Fine tune all layers")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE / 10,
    #             epochs=160,
    #             layers='all')

def predict_single_image(model, config, image):
    start_time = time.time()
    detections = model.detect([image], verbose=0)[0]
    fps = 1.0 / (time.time() - start_time)
    print("FPS: ", fps)
    return detections

def prepare_rbox(rbox):
    rbox = np.reshape(rbox, (-1, 2))
    rbox = rbox[:,[1,0]]
    return rbox

def draw_rbox(image, rbox, color):
    rbox = prepare_rbox(rbox)
    cv2.drawContours(image, [rbox], 0, color, 3)

def predict(model, config, annotations_path, target_id = None, show_result = False):
    annotations = load_dataset_json(annotations_path, shuffle=False)
    results = []
    for item in annotations:
        if target_id != None and not target_id in [a['id'] for a in item['annotations']]:
            continue

        image_path = os.path.join(item['basefolder'], item['filepath'])
        print("Image: {}".format(image_path))

        image = cv2.imread(image_path)
        detections = predict_single_image(model, config, image)

        image_data = dict()
        image_data['basefolder'] = item['basefolder']
        image_data['filepath'] = item['filepath']
        image_data["height"] = item['height']
        image_data["width"] = item['width']
        image_data["detections"] = []

        for i in range(len(detections['rois'])):
            print("{}:{}".format(
                detections['class_ids'][i]-1,
                detections['scores'][i]))
            roi = detections['rois'][i]
            rbox = detections['rboxes3'][i]

            image_data["detections"].append({
                'class_id':  int(detections['class_ids'][i]-1),
                'scores': float(detections['scores'][i]),
                'box': roi.astype(int).tolist(),
                'rbox': rbox.astype(int).tolist()
                })

            if show_result:
                cv2.rectangle(image, (roi[1], roi[0]), (roi[3], roi[2]), (0,255,0), 2)
                draw_rbox(image, detections['rboxes1'][i], (0, 0, 255))
                draw_rbox(image, detections['rboxes2'][i], (0, 255, 0))
                draw_rbox(image, detections['rboxes3'][i], (255, 0, 0))


        image_data["ground_truths"] = item['annotations']

        if show_result:
            cv2.imshow("", image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        results += [image_data]

    path, filename = os.path.split(annotations_path)
    base_name = os.path.splitext(filename)[0]
    result_filepath = os.path.join(path, "{}_results.json".format(base_name))
    json.dump(results, open(result_filepath, 'w'))
    print("Save result with success in {}".format(result_filepath))

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]
            rle = maskUtils.encode(np.asfortranarray(mask))
            rle['counts'] = str(rle['counts'], 'utf-8')
            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "gemini"),
                "bbox": [int(bbox[1]), int(bbox[0]), int(bbox[3] - bbox[1]), int(bbox[2] - bbox[0])],
                "score": float(score),
                "segmentation": rle
            }
            results.append(result)
    return results

def build_mask_from_rboxes(h, w, rboxes):
    full_masks = []
    for rbox in rboxes:
        mask = np.zeros((h, w), dtype=np.uint8)
        rbox = np.reshape(rbox, (-1, 2)).astype(int)
        rbox = rbox[:,[1, 0]]
        cv2.fillConvexPoly(mask, rbox, 255)
        full_masks.append(mask)

    if full_masks:
        full_masks = np.stack(full_masks, axis=-1)

    return full_masks

def summarize(results, coco, eval_type, coco_image_ids):
    coco_results = coco.loadRes(results)
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

def evaluate_coco(model, dataset, coco, eval_type="segm", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results_rboxes1 = []
    results_rboxes2 = []
    results_rboxes3 = []
    results_rboxes4 = []
    for i, image_id in enumerate(image_ids):
        print("image_id: {}".format(image_id))

        # Load image
        image = dataset.load_image(image_id)
        h, w, _ = image.shape

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=1)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        full_masks = build_mask_from_rboxes(h, w, r["rboxes1"])
        image_results_rboxes1 = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"], full_masks)

        results_rboxes1.extend(image_results_rboxes1)

        full_masks = build_mask_from_rboxes(h, w, r["rboxes2"])
        image_results_rboxes2 = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"], full_masks)

        results_rboxes2.extend(image_results_rboxes2)

        full_masks = build_mask_from_rboxes(h, w, r["rboxes3"])
        image_results_rboxes3 = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"], full_masks)

        results_rboxes3.extend(image_results_rboxes3)

        # for i in range(r['rois'].shape[0]):

        #     mask, _ = dataset.load_mask(image_id)

        #     for i in range(mask.shape[-1]):

        #         bbox = dataset.image_info[image_id]['annotations'][i]['bbox']

        #         cv2.rectangle(
        #             image,
        #             (r['rois'][i][1], r['rois'][i][0]), (r['rois'][i][3], r['rois'][i][2]),
        #             (0,0,255), 2)

        #         draw_rbox(image, r['rboxes1'][i], (0, 0, 255))
        #         draw_rbox(image, r['rboxes2'][i], (0, 255, 0))
        #         draw_rbox(image, r['rboxes3'][i], (255, 0, 0))

        #         cv2.rectangle(
        #             image,
        #             (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]),
        #             (0,255,0), 2)

        #         m = mask[:, :, i].copy()
        #         m = m[:,:,np.newaxis]

        #         _, contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #         cv2.drawContours(image, contours, 0, (0,255,255), 3)
        # cv2.imshow("image", image)
        # if cv2.waitKey(5) & 0xFF == ord('q'):
        #     break

    print("results_rboxes1: {}".format(results_rboxes1))

    with open("gemini_rboxes1_results.json", 'w') as f:
        json.dump(results_rboxes1, f, ensure_ascii=False)

    with open("gemini_rboxes2_results.json", 'w') as f:
        json.dump(results_rboxes2, f, ensure_ascii=False)

    with open("gemini_rboxes3_results.json", 'w') as f:
        json.dump(results_rboxes3, f, ensure_ascii=False)

    summarize(results_rboxes1, coco, eval_type, coco_image_ids)
    summarize(results_rboxes2, coco, eval_type, coco_image_ids)
    summarize(results_rboxes3, coco, eval_type, coco_image_ids)

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Train RBBOX-RCNN using Gemini dataset.')

    parser.add_argument("command",
                        metavar="<command>",
                        help="'train', 'predict' or 'evaluate' on Gemini dataset")

    parser.add_argument('--train-annotations', required=False,
                        metavar="/path/to/gemini/annotations",
                        help='Path to json file with the gemini training annotations')

    parser.add_argument('--valid-annotations', required=False,
                        metavar="/path/to/gemini/annotations",
                        default=None,
                        help='Path to json file with the gemini validation annotations')

    parser.add_argument('--test-annotations', required=False,
                        metavar="/path/to/gemini/annotations",
                        default=None,
                        help='Path to json file with the gemini validation annotations')

    parser.add_argument('--coco-annotations', required=False,
                        metavar="/path/to/gemini/annotations",
                        default=None,
                        help='Path to json file with the gemini validation annotations')

    parser.add_argument('--show', required=False,
                        default=False,
                        help='Show the detected result')

    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')

    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")

    args = parser.parse_args()

    # parameters validation
    if args.command == "train":
        assert args.train_annotations, "Argument --annotations is required for training"
    elif args.command == "predict":
        assert args.test_annotations, "Argument --test-annotations is required for prediction"
    elif args.command == "evaluate":
        assert args.coco_annotations, "Argument --coco-annotations is required for evaluation"

    # create model
    if args.command == "train":
        config = GeminiConfig()
        model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)
        model.keras_model.summary()
    elif args.command == "predict" or args.command == "evaluate":
        config = InferenceConfig()
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)
        model.keras_model.summary()

    # select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "last":
        model_path = model.find_last()[1]
    else:
        model_path = args.model

    print("Model: {}".format(model_path))
    # load weights
    if args.model.lower() == "coco":
        model.load_weights(model_path, by_name=True, exclude=[
            "mrcnn_class_logits",
            "mrcnn_bbox_fc","mrcnn_bbox",
            "mrcnn_rotated_bbox_fc","mrcnn_rotated_bbox",
            "mrcnn_mask"])
    else:
        model.load_weights(model_path, by_name=True)

    # train or predict
    if args.command == "train":
        train(model, config, args.train_annotations, args.valid_annotations)
    elif args.command == "predict":
        predict(model, config, args.test_annotations, show_result=args.show, target_id=0)
    elif args.command == "evaluate":
        dataset_test = CocoDataset()
        coco = dataset_test.load_coco(args.coco_annotations)
        dataset_test.prepare()
        evaluate_coco(model, dataset_test, coco, "segm", limit=None)


