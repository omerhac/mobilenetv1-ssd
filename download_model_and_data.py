from urllib import request
import os
import zipfile


def download_model(model_url, models_dir, filename):
    """Download model from model url"""
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    print(f'Downloading model into {models_dir} .....')
    request.urlretrieve(model_url,
                        filename=f'{models_dir}/{filename}')
    print('Finished downloading!')


def download_dataset_images(dataset_url, datasets_dir, dataset_name):
    """Download dataset from dataset url and unzip"""
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    print(f'Downloading coco dataset into {datasets_dir} ..... (It might take some time)')
    request.urlretrieve(dataset_url,
                        filename=f'{datasets_dir}/{dataset_name}.zip')
    print('Finished downloading!')

    # unzip dataset
    print('Extracting ....')
    with zipfile.ZipFile(f'{datasets_dir}/{dataset_name}.zip', 'r') as zip:
        zip.extractall(datasets_dir)
    os.remove(f'{datasets_dir}/{dataset_name}.zip')
    print('Finished extracting.')


def download_dataset_annotations(dataset_url, datasets_dir):
    """Download dataset annotations from dataset url and unzip"""
    if not os.path.exists(f'{datasets_dir}/annotations'):
        os.mkdir(f'{datasets_dir}/annotations')
    print(f'Downloading coco dataset annotations into {datasets_dir}/annotations ..... (It might take some time)')
    request.urlretrieve(dataset_url,
                        filename=f'{datasets_dir}/annotations.zip')
    print('Finished downloading!')

    # unzip annotations
    print('Extracting ....')
    with zipfile.ZipFile(f'{datasets_dir}/annotations.zip', 'r') as zip:
        zip.extractall(f'{datasets_dir}')
    os.remove(f'{datasets_dir}/annotations.zip')
    print('Finished extracting.')


if __name__ == '__main__':
    model_url = 'https://github.com/lufficc/SSD/releases/download/1.2/mobilenet_v3_ssd320_voc0712.pth'
    datasets_url = 'http://images.cocodataset.org/zips/val2017.zip'
    annotations_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'

    # download model
    download_model(model_url, 'models', 'mobilenetv3_ssd.model')
    # download dataset
    download_dataset_images(datasets_url, 'datasets', 'val2017')
    download_dataset_annotations(annotations_url, 'datasets')


