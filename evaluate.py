import torch
from ssd.modeling.detector import build_detection_model
from ssd.config import cfg
import glob
from PIL import Image
from vizer.draw import draw_boxes
from ssd.data.datasets import VOCDataset
import numpy as np
import os
from ssd.data.transforms import build_transforms


@torch.no_grad()
def evaluate(model_path, images_dir, output_dir):
    """Evaluate model on images"""

    device = torch.device('cpu')
    # build mobilenetv3 ssd and cast to cpu
    cfg.merge_from_file('configs/mobilenet_v3_ssd320_voc0712.yaml')  # load config file for mobilenetv3 SSD
    cfg.freeze()
    model = build_detection_model(cfg)
    model = model.to(device)

    # load model weights
    weights = torch.load(model_path, map_location=device)['model']
    model.load_state_dict(weights)

    # prepare model for inference
    model.eval()
    # inference preprocessing
    # resize, subtract mean, make channel first dim and make Tensor from np array
    data_preprocess = build_transforms(cfg, is_train=False)
    image_paths = glob.glob(images_dir + '/*.jpg')

    # predict
    for i, image_path in enumerate(image_paths):
        # load and preprocess image
        image = np.array(Image.open(image_path).convert("RGB"))
        image_tensor = data_preprocess(image)[0]  # use only the returned image (not boxes/labels)
        image_tensor = image_tensor.unsqueeze(0)  # add batch dim
        image_name = os.path.basename(image_path)
        height, width = image.shape[:2]

        # predict
        result = model(image_tensor)[0]
        result = result.resize((width, height)).numpy()
        boxes, labels, scores = result['boxes'], result['labels'], result['scores']  # unpack results

        # filter results
        indices = scores > 0.5
        boxes = boxes[indices]
        labels = labels[indices]
        scores = scores[indices]

        # draw and save boxes
        # VOC dataset was used for training and hence its class names are used
        drawn_image = draw_boxes(image, boxes, labels, scores, VOCDataset.class_names).astype(np.uint8)
        Image.fromarray(drawn_image).save(os.path.join(output_dir, image_name))


if __name__ == '__main__':
    model_path = 'models/mobilenetv3_ssd.model'
    images_dir = 'datasets/val2017'

    evaluate(model_path, images_dir, 'output')