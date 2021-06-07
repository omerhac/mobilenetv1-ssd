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


def postprocess_image_coco(image_name, image_boxes, image_labels, image_scores, dataset_meta):
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


class Pipeline:
    """COCO inference pipeline"""
    def __init__(self, model, dataset, image_size=[300, 300]):
        """Initializes a pipeline.
        Args:
            model: MobilenetV1-SSD instance
            dataset: COCO dataset object
            image_size: image size to reshape the images to
        """

        self.image_size = image_size
        self.model = model
        self.dataset = dataset

    def preprocess(self, batch_bytes, batch_names):
        """Decode images from image bytes, reshape and normalize to [-1, 1]"""
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

            return torchvision.transforms.Resize(size=self.image_size)(image)

        images = [reshape_and_scale(image) for image in images]
        return torch.stack(images, dim=0), batch_names

    def model(self, batch_images):
        """Forward pass"""
        return self.model(batch_images)

    def postprocess(self, model_predictions, batch_filenames):
        """All the postprocessing needed after generating raw model predictions.
        Args:
            model_predictions: raw model predictions
            batch_filenames: filenames of the images fed to the model
        Returns:
            batch_coco_results: COCO formatted results -> {image_id, category_id, bbox, score}
        """

        # perform NMS and get relative bbox coordinates from model predictions
        batch_processed = self.model.model_post_process(model_predictions)
        batch_boxes, batch_labels, batch_scores = batch_processed

        # produce coco results format for each bounding box for each image
        # it acts linearly on every image because of generated COCO format and because image shape varies
        batch_coco_results = []
        for image_filename, image_boxes, image_labels, image_scores in zip(batch_filenames,
                                                                           batch_boxes, batch_labels, batch_scores):
            # produce coco result for that image and aggregate
            image_coco_results = postprocess_image_coco(image_filename,
                                                        image_boxes, image_labels, image_scores, self.dataset)
            batch_coco_results += image_coco_results

        return batch_coco_results

    def __call__(self, batch_bytes, batch_names):
        """Full pass through the pipeline, preprocess, model forward pass and postprocess"""
        batch_images, batch_names = self.preprocess(batch_bytes, batch_names)
        model_predctions = self.model(batch_images)
        batch_coco_results = self.postprocess(model_predctions, batch_names)
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
    loader = iter(torch.utils.data.DataLoader(coco_data, batch_size=batch_size, collate_fn=lambda x: zip(*x)))
    pipe = Pipeline(model, dataset)

    # prepare model for inference
    model.eval()
    coco_results = []
    print(f'Finished loading model in {time.time() - start_time} seconds.')
    start_time = time.time()

    # predict
    for batch_id in tqdm(range(len(loader))):
        # Data Loader loads images bytes and filenames
        batch_image_bytes, batch_image_names = next(loader)

        # move data through pipeline
        batch_coco_results = pipe(batch_image_bytes, batch_image_names)

        coco_results += batch_coco_results  # aggregate final results

        # save images with bboxes if save_images is True and output_dir is provided
        if save_images:
            assert output_dir, 'output_dir should be provided for saving images with bboxes'
            draw_bboxed_images(batch_coco_results, images_dir, dataset, output_dir)

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

    evaluate(model_path, coco_data, batch_size=32, output_dir='output', save_images=False)

