import torch
import json
from collections import defaultdict
from PIL import Image
from vizer.draw import draw_boxes
import numpy as np
import os
import warnings
from torch.serialization import SourceChangeWarning
from coco import COCO
from tqdm import tqdm
from pycocotools import coco
from pycocotools.cocoeval import COCOeval
import time
from models.ssd_mobilenet_v1 import create_mobilenetv1_ssd
from torchvision.io.image import decode_image
import cv2
import torchvision


# disable source change warning
warnings.filterwarnings("ignore", category=SourceChangeWarning)


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


def postprocess_coco(images_ids, images_boxes, images_labels, images_scores):
    """Final postprocess for model predictions. This is purely for generating the COCO result format.
    Args:
        images_ids: list of COCO image ids in the order they appear in the dataset
        images_boxes: relative bounding boxes coordinates as predicted by the model after model postprocessing, list of
                      predicted boxes pre image
        images_labels: list of tensors depicting the labels predicted for each bounding box in the image
    Returns:
        coco_results: [detection_result1, detection_result2, ...] where
                            detection result = {image_id, category_id, bbox, score} as requested in COCO
    """

    # append labels and scores
    coco_results = []

    for image_id, image_boxes, image_labels, image_scores in zip(images_ids, images_boxes, images_labels,
                                                                 images_scores):
        # produce COCO results format per each image
        for idx, box in enumerate(image_boxes):
            coco_results.append({
                'image_id': image_id,
                'category_id': int(image_labels[idx]),
                'bbox': image_boxes[idx],
                'score': float(image_scores[idx])
            })

    return coco_results


class Pipeline:
    """COCO inference pipeline"""
    def __init__(self, model, image_size=[300, 300]):
        """Initializes a pipeline.
        Args:
            model: MobilenetV1-SSD instance
            image_size: image size to reshape the images to
        """

        self.image_size = image_size
        self.model = model

    def preprocess(self, batch_bytes):
        """Decode images from image bytes, reshape and normalize to [-1, 1].
        Args:
            batch_bytes: list of images bytes
        Returns:
            images, [images_shape]: stacked batch images and list of images shapes
        """
        # decode images
        images = [decode_image(image_bytes) for image_bytes in batch_bytes]

        def reshape_and_scale(image):
            """Helper function"""
            image = image.type(torch.FloatTensor)  # convert to float32

            # some images might be grayscale
            if len(image.shape) < 3 or image.shape[0] != 3:
                image = image.numpy()
                image = image.transpose([1, 2, 0])  # tanspose to HWC for cv2
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                image = image.transpose([2, 0, 1])  # tanspose batcb to CHW for model postprocessing
                image = torch.tensor(image, dtype=torch.float32)

            # normalize to zero mean
            image -= 127.5
            image /= 127.5

            return torchvision.transforms.Resize(size=self.image_size)(image), image.shape

        images, images_shapes = zip(*[reshape_and_scale(image) for image in images])
        return torch.stack(images, dim=0), images_shapes

    def model(self, batch_images):
        """Forward pass"""
        return self.model(batch_images)

    def postprocess(self, model_predictions, batch_shapes):
        """All the postprocessing needed after generating raw model predictions.
        Args:
            model_predictions: raw model predictions
            batch_shapes: shapes of the original images before resizing
        Returns:
            [batch_boxes, batch_labels, batch_scores]: boxes, labels and scores for each image
        """

        # perform NMS and get relative bbox coordinates from model predictions
        batch_processed = self.model.model_post_process(model_predictions)
        batch_boxes, batch_labels, batch_scores = batch_processed
        # transform relative coordinates into absolute ones
        batch_boxes = [self.process_box(boxes, shape) for (boxes, shape) in zip(batch_boxes, batch_shapes)]

        return batch_boxes, batch_labels, batch_scores

    def __call__(self, batch_bytes):
        """Full pass through the pipeline, preprocess, model forward pass and postprocess"""
        batch_images, batch_shapes = self.preprocess(batch_bytes)
        model_predictions = self.model(batch_images)
        batch_coco_results = self.postprocess(model_predictions, batch_shapes)
        return batch_coco_results

    @staticmethod
    def process_box(boxes, shape):
        """"Process boxes (comes from model as xmax, ymax, xmin, ymin). Process relative coordinates into
        absolute ones.
        Args:
            boxes: image predicted bounding boxes with relative coordinates
            shape: original image shape before resizing
        Returns:
            [transformed_boxes]: image boxes transformed into absolute coordinates
        """
        _, width, height = shape
        transformed_boxes = []
        for box in boxes:
            # round to nearest tenth of a pixel to reduce result file size
            x_min = round(float(box[0] * width), ndigits=1)
            y_min = round(float(box[1] * height), ndigits=1)
            x_max = round(float(box[2] * width), ndigits=1)
            y_max = round(float(box[3] * height), ndigits=1)
            w = x_max - x_min
            h = y_max - y_min
            transformed_boxes.append([x_min, y_min, w, h])
        return transformed_boxes


@torch.no_grad()
def evaluate(model_path, dataset, batch_size=32, coco_val=False, save_images=False):
    """Main benchmark method.
    Args:
        model_path: path to serialized model
        dataset: dataset pytorch object
        batch_size: batch size
        coco_val: flag, whether the dataset is COCO validation, to benchmark detections accuracy
        save_images: flag, whether to save images with bboxes
    """

    start_time = time.time()

    # build mobilenetv1-ssd
    model = create_mobilenetv1_ssd(len(dataset.class_names))

    # load pretrained model weights
    model.load_state_dict(torch.load(model_path))

    # prepare dataloader
    loader = iter(torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=lambda x: x))
    pipe = Pipeline(model)

    # prepare model for inference
    model.eval()
    bbox_results, label_results, score_results = [], [], []
    print(f'Finished loading model in {time.time() - start_time} seconds.')
    start_time = time.time()

    # predict
    for batch_id in tqdm(range(len(loader))):
        # Data Loader loads images bytes
        batch_image_bytes = next(loader)

        # move data through pipeline
        batch_bbox, batch_labels, batch_scores = pipe(batch_image_bytes)

        bbox_results += batch_bbox  # aggregate final results
        label_results += batch_labels
        score_results += batch_scores

        # save images with bboxes if save_images is True and output_dir is provided
        if save_images:
            assert output_dir, 'output_dir should be provided for saving images with bboxes'
            draw_bboxed_images(batch_coco_results, images_dir, dataset, output_dir)

    print(f'Finished inference in {time.time() - start_time} seconds.')

    if coco_val:
        # generate image ids
        images_ids = [dataset.meta_dict_by_filename[filename]['id'] for filename in dataset._file_names]
        # generate COCO results format
        coco_results = postprocess_coco(images_ids, bbox_results, label_results, score_results)

        # evaluate metrics
        with open('.detection_results_temp.json', 'w') as file:
            json.dump(coco_results, file)
        evaluate_results('.detection_results_temp.json', dataset.annotaion_filepath)
        os.remove('.detection_results_temp.json')


def draw_bboxed_images(batch_coco_results, images_dir, dataset, output_dir):
    """Save images with bounding boxes as depicted in the COCO results"""
    # dict for grouping coco detection results per image
    batch_detection_dict = defaultdict(lambda: {
        'boxes': [],
        'labels': [],
        'scores': []
    })
    for detection in batch_coco_results:
        # transform boxes to drawing format
        box = transform_coco_box_for_drawing(detection['bbox'])
        # append bounding box data
        batch_detection_dict[detection['image_id']]['boxes'].append(box)
        batch_detection_dict[detection['image_id']]['labels'].append(detection['category_id'])
        batch_detection_dict[detection['image_id']]['scores'].append(detection['score'])

    for image_id in batch_detection_dict.keys():
        image_filename = dataset.image_id_to_filename[image_id]
        try:
            image_boxes = batch_detection_dict[image_id]['boxes']
            image_labels = batch_detection_dict[image_id]['labels']
            image_scores = batch_detection_dict[image_id]['scores']
            # draw all of the images bounding boxes
            image = np.array(Image.open(os.path.join(images_dir, image_filename)).convert("RGB"))
            drawn_image = draw_boxes(image, image_boxes, image_labels, image_scores, dataset.class_names)
            Image.fromarray(drawn_image).save(os.path.join(output_dir, image_filename))
        except:
            print(f'Cant draw {image_filename} boxes because its grayscale')


if __name__ == '__main__':
    coco_data = COCO('datasets/val2017', 'datasets/annotations/instances_val2017.json', 'coco_labels.txt')
    model_path = 'trained_models/mobilenetv1-ssd.pt'
    images_dir = 'datasets/val2017'

    # create output dir
    if not os.path.exists('output'):
        os.mkdir('output')

    evaluate(model_path, coco_data, batch_size=32, coco_val=True, save_images=False)

