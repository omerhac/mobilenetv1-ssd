import json
import torch
from torch.utils.data import Dataset
import os
from torchvision.io import read_image
import torchvision
import cv2

class COCO(Dataset):
    """Class for storing COCO dataset image metadata"""
    def __init__(self, images_dir, annotation_filepath, labels_filepath, image_size=[300, 300]):
        """Initialize COCO metadata instance
        Args:
            annotation_filepath: path to annotation file
            labels_filepath: path to labels list textfile
            image_size: model input image size
            images_dir: path to image directory
        """
        super(COCO, self).__init__()

        self.image_size = image_size
        self.meta_dict_by_filename = {}
        self.annotaion_filepath = annotation_filepath
        self.images_dir = images_dir
        self._file_names = []

        # get labels
        with open(labels_filepath) as file:
            coco_labels = file.read().splitlines()

        self.class_names = {key + 1: coco_labels[key] for key in range(len(coco_labels))}

        # get image metadata from annotation file
        with open(annotation_filepath) as file:
            metadata = json.load(file)['images']

        self.count = len(metadata)

        # collect metadata
        for image_data in metadata:
            self.meta_dict_by_filename[image_data['file_name']] = {
                'id': image_data['id'],
                'width': image_data['width'],
                'height': image_data['height']
            }
            self._file_names.append(image_data['file_name'])

    def __getitem__(self, item):
        img_path = os.path.join(self.images_dir, self._file_names[item])
        image = read_image(img_path)
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
        return torchvision.transforms.Resize(size=self.image_size)(image), self._file_names[item]

    def __len__(self):
        return len(self._file_names)


if __name__ == '__main__':
    data = COCO('datasets/val2017', 'datasets/annotations/instances_val2017.json', 'coco_labels.txt')
    data_loader = torch.utils.data.DataLoader(data, batch_size=64)
    d = next(iter(data_loader))
    print(d.shape)

