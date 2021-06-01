import torch
import glob
from PIL import Image
from vizer.draw import draw_boxes
import numpy as np
import os
import cv2
import warnings
from torch.serialization import SourceChangeWarning
import json
import matplotlib.pyplot as plt

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


@torch.no_grad()
def evaluate(model_path, images_dir, class_names, output_dir):
    """Evaluate model on images"""

    device = torch.device('cpu')
    # build mobilenetv1 ssd and cast to cpu
    # load model weights
    model = torch.load(model_path, map_location=device)
    model = model.to(device)

    # prepare model for inference
    model.eval()
    image_paths = glob.glob(images_dir + '/*.jpg')

    # predict
    for i, image_path in enumerate(image_paths[2:]):
        # load and preprocess image
        image = np.array(Image.open(image_path).convert("RGB"))
        height, width = image.shape[:2]
        image_proc = pre_process_coco_mobilenet(image, dims=[300, 300, 3], need_transpose=True)
        image_tensor = torch.tensor(image_proc, dtype=torch.float32)
        image_tensor = image_tensor.unsqueeze(0)  # add batch dim
        image_name = os.path.basename(image_path)

        # predict
        result = model(image_tensor)
        # unpack
        boxes, labels, scores = result
        boxes = boxes[0]
        labels = labels[0]
        scores = scores[0]

        # process boxes
        def process_box(box):
            y_min = box[0] * height
            x_min = box[1] * width
            y_max = box[2] * height
            x_max = box[3] * width
            w = x_max - x_min
            h = y_max - y_min
            return x_min, y_min, w, h

        boxes = [process_box(box) for box in boxes]
        labels = [int(label.numpy()) for label in labels]
        ####
        b = boxes[0]
        b = [int(a.numpy()) for a in b]
        plt.imsave(output_dir + '/check.jpg', image[b[0]:b[0]+b[2],b[1]:b[1]+b[3],:])
        ####


        # draw and save boxes
        drawn_image = draw_boxes(image, boxes, labels, scores, class_names)
        Image.fromarray(drawn_image).save(os.path.join(output_dir, image_name))


if __name__ == '__main__':
    model_path = 'trained_models/mobilenetv1_ssd.model'
    images_dir = 'datasets/val2017'

    with open('datasets/annotations/instances_val2017.json') as file:
        annotations = json.load(file)
    class_names = annotations['categories']

    class_names = {key: class_names[key]['name'] for key in range(len(class_names))}
    print(class_names)
    evaluate(model_path, images_dir, class_names, 'output')