import torch
import glob
from PIL import Image
from vizer.draw import draw_boxes
import numpy as np
import os
import cv2
import warnings
from torch.serialization import SourceChangeWarning
from coco import COCO

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


def preprocess_dataset(dataset_dir, save_dir=None):
    """Preprocess all the images in the dataset and save them as arrays to save_dir. Reshape to (300,300),
    Zero mean,  scale and transpose to CHW"""

    image_paths = glob.glob(dataset_dir + '/*.jpg')

    if not save_dir:
        save_dir = dataset_dir + '/preprocessed'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for image_path in image_paths:
        image_name = os.path.basename(image_path).split('.')[0]
        image = np.array(Image.open(image_path))
        image = pre_process_coco_mobilenet(image, dims=[300, 300, 3], need_transpose=True)  # mobilenet specs
        np.save(os.path.join(save_dir, image_name), image)  # save


def transform_coco_box_for_drawing(box):
    """Transform [xmin, ymin, w,h] tensor to [xmin, ymin, xmax, ymax] tuple"""
    x_min = int(box[0])
    y_min = int(box[1])
    x_max = x_min + int(box[2])
    y_max = y_min + int(box[3])
    return x_min, y_min, x_max, y_max


def postprocess_example(prediction, image_name, dataset_meta):
    """Postprocess and SSD prediction.
    Args:
        prediction: output tensor from model prediction on a single image. shape [bboxes, labels, scores]
        image_name: image name in dataset directory. i.e. 'xxxx.jpg'
        dataset_meta: dataset object describing its metadata
    Returns:
        detection_results: [detection_result1, detection_result2, ...] where
                            detection result = {image_id, category_id, bbox, score} as requested in COCO
    """
    # unpack
    boxes, labels, scores = prediction
    boxes = boxes[0]
    labels = labels[0]
    scores = scores[0]

    # extract metadata
    width, height = dataset_meta.image_dict[image_name]['width'], dataset_meta.image_dict[image_name]['height']
    image_id = dataset_meta.image_dict[image_name]['id']

    # process boxes (comes from model as xmax, ymax, xmin, ymin)
    def process_box(box):
        x_min = round(float(box[0] * width), ndigits=1)  # round to nearest tenth of a pixel to reduce result file size
        y_min = round(float(box[1] * height), ndigits=1)
        x_max = round(float(box[2] * width), ndigits=1)
        y_max = round(float(box[3] * height), ndigits=1)
        w = x_max - x_min
        h = y_max - y_min
        return [x_min, y_min, w, h]

    boxes = [process_box(box) for box in boxes]

    # append labels and scores
    detection_results = []

    for idx, box in enumerate(boxes):
        detection_results.append({
            'image_id': image_id,
            'category_id': int(labels[idx]),
            'bbox': boxes[idx],
            'score': float(scores[idx])
        })

    return detection_results


@torch.no_grad()
def evaluate(model_path, images_dir, dataset_meta, output_dir=None):
    """Evaluate model on images.
    Args:
        model_path: path to serialized model
        images_dir: path to images directory
        dataset_meta: dataset object describing its metadata
        output_dir: OPTIONAL, where to save detection results
    """

    device = torch.device('cpu')
    # build mobilenetv1 ssd and cast to cpu
    # load model weights
    model = torch.load(model_path, map_location=device)
    model = model.to(device)

    # prepare model for inference
    model.eval()
    image_paths = glob.glob(images_dir + '/*.jpg')

    # predict
    for i, image_path in enumerate(image_paths):
        # load and preprocess image
        image = np.array(Image.open(image_path))
        image_tensor = pre_process_coco_mobilenet(image, dims=[300, 300, 3], need_transpose=True)
        image_tensor = image_tensor.unsqueeze(0)  # add batch dim
        image_name = os.path.basename(image_path)

        # predict
        result = model(image_tensor)
        detection_results = postprocess_example(result, image_name, dataset_meta)

        # save detection results if output_dir is provided
        if output_dir:
            boxes, labels, scores = [], [], []
            for detection in detection_results:
                boxes.append(detection['bbox'])
                labels.append(detection['category_id'])
                scores.append(detection['score'])
            boxes = [transform_coco_box_for_drawing(box) for box in boxes]

            # draw and save boxes
            try:
                drawn_image = draw_boxes(image, boxes, labels, scores, dataset_meta.class_names)
                Image.fromarray(drawn_image).save(os.path.join(output_dir, image_name))
            except:
                print(f'Cant draw {image_name} boxes because its grayscale')


if __name__ == '__main__':
    coco_meta = COCO('datasets/annotations/instances_val2017.json', 'coco_labels.txt')
    model_path = 'trained_models/ssd_mobilenet_v1.pytorch'
    images_dir = 'datasets/val2017'

    evaluate(model_path, images_dir, coco_meta, 'output')