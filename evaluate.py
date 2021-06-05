import torch
import json
import glob
from collections import defaultdict
from PIL import Image
from vizer.draw import draw_boxes
import numpy as np
import os
import cv2
import warnings
from torch.serialization import SourceChangeWarning
from coco import COCO
from tqdm import tqdm
from pycocotools import coco
from pycocotools.cocoeval import COCOeval
import time
from models.ssd_mobilenet_v1 import create_mobilenetv1_ssd


# disable source change warning
warnings.filterwarnings("ignore", category=SourceChangeWarning)

COCO_NUM_CLASSES = 91


def pre_process_coco_mobilenet(img, dims=None, need_transpose=False):
    """Preprocess image for model digestion. zero mean and scale to [-1, 1]"""
    img = maybe_resize(img, dims)
    img -= 127.5
    img /= 127.5
    # transpose if needed
    if need_transpose:
        img = img.transpose([2, 0, 1])
    return torch.tensor(img, dtype=torch.float32)


def maybe_resize(img, dims):
    img = np.array(img, dtype=np.float32)
    if len(img.shape) < 3 or img.shape[2] != 3:
        # some images might be grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if dims != None:
        im_height, im_width, _ = dims
        img = cv2.resize(img, (im_width, im_height), interpolation=cv2.INTER_LINEAR)
    return img


def transform_coco_box_for_drawing(box):
    """Transform [xmin, ymin, w,h] tensor to [xmin, ymin, xmax, ymax] tuple"""
    x_min = int(box[0])
    y_min = int(box[1])
    x_max = x_min + int(box[2])
    y_max = y_min + int(box[3])
    return x_min, y_min, x_max, y_max


def evaluate_results(detection_results, annotation_filepath):
    """Evaluate metrics of detection results"""
    coco_GT = coco.COCO(annotation_file=annotation_filepath)
    coco_DT = coco_GT.loadRes(detection_results)
    coco_eval = COCOeval(cocoGt=coco_GT, cocoDt=coco_DT, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


def postprocess_example_coco(image_name, image_boxes, image_labels, image_scores, dataset_meta):
    """Final postprocess for a single example. This is purely for generating the COCO result format.
    Args:
        image_name: image name in dataset directory. i.e. 'xxxx.jpg'
        image_boxes: relative bounding boxes coordinates as predicted by the model after model postprocessing
        image_labels: category labels for each bounding box
        image_scores: predicted class score for the predicted class at each bounding box
        dataset: dataset object that has dataet metadata
    Returns:
        detection_results: [detection_result1, detection_result2, ...] where
                            detection result = {image_id, category_id, bbox, score} as requested in COCO
    """
    # unpack

    # extract metadata
    width, height = dataset_meta.meta_dict_by_filename[image_name]['width'], dataset_meta.meta_dict_by_filename[image_name]['height']
    image_id = dataset_meta.meta_dict_by_filename[image_name]['id']

    # process boxes (comes from model as xmax, ymax, xmin, ymin)
    def process_box(box):
        x_min = round(float(box[0] * width), ndigits=1)  # round to nearest tenth of a pixel to reduce result file size
        y_min = round(float(box[1] * height), ndigits=1)
        x_max = round(float(box[2] * width), ndigits=1)
        y_max = round(float(box[3] * height), ndigits=1)
        w = x_max - x_min
        h = y_max - y_min
        return [x_min, y_min, w, h]

    boxes = [process_box(box) for box in image_boxes]

    # append labels and scores
    detection_results = []

    for idx, box in enumerate(boxes):
        detection_results.append({
            'image_id': image_id,
            'category_id': int(image_labels[idx]),
            'bbox': boxes[idx],
            'score': float(image_scores[idx])
        })

    return detection_results


def postprocess_batch(model_predictions, batch_filenames, model, dataset):
    """All the postprocessing needed after generating raw model predictions"""

    # perform NMS and get relative bbox coordinates from model predictions
    batch_processed = model.model_post_process(model_predictions)
    batch_boxes, batch_labels, batch_scores = batch_processed

    # produce coco results format for each bounding box for each image
    # it acts linearly on every image because image shape varies
    batch_coco_results = []
    for image_filename, image_boxes, image_labels, image_scores in zip(batch_filenames,
                                                                       batch_boxes, batch_labels, batch_scores):
        # produce coco result for that image and aggregate
        image_coco_results = postprocess_example_coco(image_filename,
                                                      image_boxes, image_labels, image_scores, dataset)
        batch_coco_results += image_coco_results

    return batch_coco_results


@torch.no_grad()
def evaluate(model_path, dataset, batch_size=32, output_dir=None, save_images=False):
    """Main benchmark method.
    Args:
        model_path: path to serialized model
        dataset: dataset pytorch object
        batch_size: batch size
        output_dir: OPTIONAL, where to save detection results (required for saving images also)
        save_images: flag, whether to save images with bboxes
    """

    start_time = time.time()

    # build mobilenetv1-ssd
    model = create_mobilenetv1_ssd(len(dataset.class_names))

    # load pretrained model weights
    model.load_state_dict(torch.load(model_path))

    # prepare dataloader
    loader = iter(torch.utils.data.DataLoader(dataset, batch_size=batch_size))
    # prepare model for inference
    model.eval()
    coco_results = []

    # predict
    for batch_idx in tqdm(range(len(loader))):
        # load and preprocess batch of images (reshape, normalize and zero mean)
        batch_images, batch_filenames = next(loader)

        # get DNN prediction
        results = model(batch_images)

        # postprocess
        # do NMS, get relative bbox coordinates and then transform to COCO foramt
        batch_coco_results = postprocess_batch(results, batch_filenames, model, dataset)
        coco_results += batch_coco_results  # aggregate final results

        # save image with bboxes if save_images is True and output_dir is provided
        if save_images:
            assert output_dir, 'output_dir should be provided for saving images with bboxes'

            # dict for grouping coco detection results per image
            batch_detection_dict = defaultdict(lambda: {
                'boxes': [],
                'labels': [],
                'scores': []
            })
            for detection in batch_coco_results:
                # transform boxes to drawing format
                boxes = transform_coco_box_for_drawing(detection['bbox'])
                batch_detection_dict[detection['image_id']]['boxes'].append(boxes)
                batch_detection_dict[detection['image_id']]['labels'].append(detection['category_id'])
                batch_detection_dict[detection['image_id']]['scores'].append(detection['score'])

            for image_id in batch_detection_dict.keys():
                try:
                    image_filename = dataset.image_id_to_filename[image_id]
                    image_boxes  =batch_detection_dict[image_id]['boxes']
                    image_labels = batch_detection_dict[image_id]['labels']
                    image_scores = batch_detection_dict[image_id]['scores']
                    image = np.array(Image.open(os.path.join(images_dir, image_filename)).convert("RGB"))
                    drawn_image = draw_boxes(image, image_boxes, image_labels, image_scores, dataset.class_names)
                    Image.fromarray(drawn_image).save(os.path.join(output_dir, image_filename))
                except:
                    print(f'Cant draw {44} boxes because its grayscale')

    print(f'Finished inference in {time.time() - start_time} seconds.')

    # save if needed
    if output_dir:
        with open(f'{output_dir}/detection_results.json', 'w') as file:
            json.dump(coco_results, file)

        # evaluate metrics
        evaluate_results(f'{output_dir}/detection_results.json', dataset.annotaion_filepath)
    else:
        with open('.detection_results_temp.json', 'w') as file:
            json.dump(coco_results, file)

        # evaluate metrics
        evaluate_results('.detection_results_temp.json', dataset_meta.annotaion_filepath)
        os.remove('.detection_results_temp.json')


if __name__ == '__main__':
    coco_meta = COCO('datasets/val2017', 'datasets/annotations/instances_val2017.json', 'coco_labels.txt')
    model_path = 'trained_models/mobilenetv1-ssd.pt'
    images_dir = 'datasets/val2017'

    # create output dir
    if not os.path.exists('output'):
        os.mkdir('output')

    evaluate(model_path, coco_meta, output_dir='output', save_images=True)

