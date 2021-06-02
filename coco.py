import json


class COCO:
    """Class for storing COCO dataset image metadata"""
    def __init__(self, annotaion_filepath, image_size=[300,300]):
        self.image_size = image_size
        self.image_dict = {}
        self.label_list = []
        self.image_ids = []
        self.image_sizes = []

        # get image metadata from annotation file
        with open(annotaion_filepath) as file:
            metadata = json.load(file)['images']

        self.count = len(metadata)

        # collect metadata
        for image_data in metadata:
            self.image_dict[image_data['file_name'].split('.')[0]] = {  # use only filename, without.jpg
                'id': image_data['id'],
                'width': image_data['width'],
                'height': image_data['height']
            }