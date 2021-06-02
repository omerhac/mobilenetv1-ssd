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
    img = maybe_resize(img, dims)
    img -= 127.5
    img /= 127.5
    # transpose if needed
    if need_transpose:
        img = img.transpose([2, 0, 1])
    return img


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


def postprocess_example(prediction, width, height):
    """Postprocess and SSD prediction. """
    # unpack
    boxes, labels, scores = prediction
    boxes = boxes[0]
    labels = labels[0]
    scores = scores[0]

    # process boxes (comes from model as xmax, ymax, xmin, ymin)
    def process_box(box):
        x_min = box[0] * width
        y_min = box[1] * height
        x_max = box[2] * width
        y_max = box[3] * height
        w = x_max - x_min
        h = y_max - y_min
        return np.array([x_min, y_min, w, h])

    boxes = [process_box(box) for box in boxes]

    return boxes, labels, scores


@torch.no_grad()
def evaluate(model_path, images_dir, class_names, output_dir):
    """Evaluate model on images"""

    device = torch.device('cpu')
    # build mobilenetv1 ssd and cast to cpu
    # load model weights
    model = torch.load(model_path, map_location=lambda storage, loc: storage)
    model = model.to(device)

    # prepare model for inference
    model.eval()
    image_paths = glob.glob(images_dir + '/*.jpg')

    # predict
    for i, image_path in enumerate(image_paths):
        # load and preprocess image
        image = np.array(Image.open(image_path))
        height, width = image.shape[:2]
        image_proc = pre_process_coco_mobilenet(image, dims=[300, 300, 3], need_transpose=True)
        image_tensor = torch.tensor(image_proc, dtype=torch.float32)
        image_tensor = image_tensor.unsqueeze(0)  # add batch dim
        image_name = os.path.basename(image_path)

        # predict
        result = model(image_tensor)
        boxes, labels, scores = postprocess_example(result, width, height)
        boxes = [transform_coco_box_for_drawing(box) for box in boxes]
        labels = [int(label.numpy()) for label in labels]

        # draw and save boxes
        try:
            drawn_image = draw_boxes(image, boxes, labels, scores, class_names)
            Image.fromarray(drawn_image).save(os.path.join(output_dir, image_name))
        except:
            print(f'Cant draw {image_name} boxes because its grayscale')


if __name__ == '__main__':
    coco = COCO('datasets/annotations/instances_val2017.json')
    print(coco.image_dict['000000397133'])
    model_path = 'trained_models/ssd_mobilenet_v1.pytorch'
    images_dir = 'datasets/val2017'

    with open('coco_labels.txt') as file:
        coco_labels = file.read().splitlines()

    class_names = {key+1: coco_labels[key] for key in range(len(coco_labels))}
    print(class_names)
    evaluate(model_path, images_dir, class_names, 'output')