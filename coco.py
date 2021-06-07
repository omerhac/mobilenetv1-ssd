import json
import torch
from torch.utils.data import Dataset
import os
from torchvision.io.image import read_file

class COCO(Dataset):
    """Class for storing COCO dataset image metadata"""
    def __init__(self, images_dir, annotation_filepath, labels_filepath):
        """Initialize COCO metadata instance
        Args:
            annotation_filepath: path to annotation file
            labels_filepath: path to labels list textfile
            images_dir: path to image directory
        """
        super(COCO, self).__init__()

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
        image = read_file(img_path)
        return image

    def __len__(self):
        return len(self._file_names)


if __name__ == '__main__':
    data = COCO('datasets/val2017', 'datasets/annotations/instances_val2017.json', 'coco_labels.txt')
    data_loader = torch.utils.data.DataLoader(data, batch_size=64, collate_fn=lambda x: x)

