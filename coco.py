import json


class COCO:
    """Class for storing COCO dataset image metadata"""
    def __init__(self, annotation_filepath, labels_filepath, image_size=[300,300]):
        """Initialize COCO metadata instance
        Args:
            annotation_filepath: path to annotation file
            labels_filepath: path to labels list textfile
            image_size: model input image size
        """
        self.image_size = image_size
        self.image_dict = {}
        self.label_list = []
        self.image_ids = []
        self.image_sizes = []

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
            self.image_dict[image_data['file_name']] = {
                'id': image_data['id'],
                'width': image_data['width'],
                'height': image_data['height']
            }