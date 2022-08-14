import argparse
import os
import random
import sys
from pathlib import Path
import shutil
import numpy as np
import json
import os
import tqdm
import glob
from urllib.request import urlretrieve
import threading
import csv

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.general import LOGGER

def _download_file_per_thread(url, file_name):
    urlretrieve(url, file_name)

def _read_json(opt):
    with open(opt.aml_json, 'r') as f:
        data = json.load(f)

    # Create easy to use arrays
    images = {}
    for img in data['images']:
        images[img['id']] = {
            'name': os.path.basename(img['file_name']),
            'url': img['absolute_url'],
        }

    annotations = data['annotations']
    categories = data['categories']
    return images, annotations, categories


def download_dataset(opt, images, annotations, categories):
    # Internal function for threaded images download.

    LOGGER.info('Deleting downloads folder')
    RESULTS = Path.resolve(opt.results)
    DOWNLOAD = Path('download')

    # Delete dataset
    if os.path.exists(DOWNLOAD):
        shutil.rmtree(DOWNLOAD, ignore_errors=True)

    if os.path.exists(RESULTS):
        shutil.rmtree(RESULTS, ignore_errors=True)


    # Create yolov5 directory structure
    LOGGER.info('Create results yolov5 folder structure')
    os.makedirs(DOWNLOAD)
    os.makedirs(DOWNLOAD/'train')
    os.makedirs(DOWNLOAD/'train/images')
    os.makedirs(DOWNLOAD/'train/labels')

    os.makedirs(RESULTS)
    os.makedirs(RESULTS/'train')
    os.makedirs(RESULTS/'train/images')
    os.makedirs(RESULTS/'train/labels')
    os.makedirs(RESULTS/'valid')
    os.makedirs(RESULTS/'valid/images')
    os.makedirs(RESULTS/'valid/labels')

    # Create yolov5 directory structure
    LOGGER.info('Downloading images')

    threads = []
    tid = 0
    for image_id in tqdm.tqdm(images.keys()):
        tid += 1
        image = images[image_id]

        # image['name'], image['url']
        img_abspath = DOWNLOAD / f'train/images/{image["name"]}'
        t = threading.Thread(target=_download_file_per_thread, args=(image['url'], img_abspath))
        t.daemon = True
        threads.append(t)
        t.start()

        # Parallelize using the number of threads specified
        if tid % opt.threads == 0:
            for t in threads:
                t.join()
            threads = []

    LOGGER.info(f'✅ Download complete: {DOWNLOAD}')


def create_complete_dataset(opt, images, annotations, categories):

    DOWNLOAD = Path('download')
    RESULTS = Path.resolve(opt.results)

    # Create image dict
    # images = {'%g' % x['id']: x for x in annotations}

    # # Class to label mapping (convert everything to shelf, bottle, void)
    # cls_to_label = {
    #     1: 1, # bottle
    #     2: 1, # can
    #     3: 1, # carton
    #     4: 1, # container
    #     5: 2, # void
    #     6: 0, # shelf
    #     7: 1, # packet
    #     8: 1, # box
    #     9: 1, # cup
    # }

    # Write labels file
    LOGGER.info('Writing annotations')
    for x in tqdm.tqdm(annotations):
        img = images[x['image_id']]

        f = DOWNLOAD / 'train/labels' / img['name']

        # The COCO box format is [top left x, top left y, width, height]
        box = np.array(x['bbox'], dtype=np.float64)

        # AML is (corner leftx, corner lefty, w, h) Yolov5 format is (center leftx, center lefty, w, h)
        box[:2] += box[2:] / 2

        # Write
        if box[2] > 0 and box[3] > 0:  # if w > 0 and h > 0
            # cls = cls_to_label[x['category_id']] # label
            cls = x['category_id'] - 1

            # Map classes
            line = cls, *(box)  # cls, box or segments
            with open(f.with_suffix('.txt'), 'a') as file:
                file.write(('%g ' * len(line)).rstrip() % line + '\n')

    # Write data.yaml file
    LOGGER.info('Writing data.yaml')
    # labels = ['shelf', 'bottle', 'void']
    labels = [cat['name'] for cat in categories]
    labels_cnt = len(labels)
    labels = json.dumps(labels)
    yaml_data = [
        f'train: {RESULTS.name}/train\n',
        f'val: {RESULTS.name}/valid\n\n',
        f'nc: {labels_cnt}\n',
        f'names: {labels}\n'
    ]
    with open(RESULTS/'data.yaml', 'w') as f:
        f.writelines(yaml_data)

    # Split dataset between train & val
    LOGGER.info('Splitting dataset')
    for image_fname in tqdm.tqdm(glob.glob(f'{DOWNLOAD.name}/train/images/*.jpg')):
        result_image_fname = image_fname.replace(DOWNLOAD.name, RESULTS.name)
        ann_fname = image_fname.replace('images/', 'labels/').replace('.jpg', '.txt')
        result_ann_fname = ann_fname.replace(DOWNLOAD.name, RESULTS.name)

        if (not os.path.exists(image_fname) or
            not os.path.exists(ann_fname)):
            LOGGER.error(f'Image/Annotation does not exist: {image_fname}: {ann_fname}')

        rand_num = random.random()
        if rand_num < 0.2:
            val_image_fname = result_image_fname.replace('train/', 'valid/')
            val_ann_fname = result_ann_fname.replace('train/', 'valid/')
            shutil.copy(image_fname, val_image_fname)
            shutil.copy(ann_fname, val_ann_fname)
        else:
            shutil.copy(image_fname, result_image_fname)
            shutil.copy(ann_fname, result_ann_fname)


    LOGGER.info(f'✅ Yolov5 Dataset creation complete: {RESULTS}')

def create_shelf_dataset(opt, images, annotations, categories):
    # Basically creating shelf dataset is similar to creating entire dataset.
    # We just need to ignore all the object annotations. We will only focus on
    # Shelf annotations.
    DOWNLOAD = Path('download')
    RESULTS = Path('shelf-dataset')

    if os.path.exists(RESULTS):
        shutil.rmtree(RESULTS, ignore_errors=True)
    os.makedirs(RESULTS)
    os.makedirs(RESULTS/'train')
    os.makedirs(RESULTS/'train/images')
    os.makedirs(RESULTS/'train/labels')
    os.makedirs(RESULTS/'valid')
    os.makedirs(RESULTS/'valid/images')
    os.makedirs(RESULTS/'valid/labels')

    for image_fname in tqdm.tqdm(glob.glob(f'{DOWNLOAD.name}/train/images/*.jpg')):
        ann_fname = image_fname.replace('images/', 'labels/').replace('.jpg', '.txt')
        result_image_fname = image_fname.replace(DOWNLOAD.name, RESULTS.name)
        result_ann_fname = ann_fname.replace(DOWNLOAD.name, RESULTS.name)

        if (not os.path.exists(image_fname) or
            not os.path.exists(ann_fname)):
            LOGGER.error(f'Image/Annotation does not exist: {image_fname}: {ann_fname}')

        image_annotations = []
        with open(ann_fname, 'r') as f:
            ann_csv = csv.reader(f, delimiter=' ')
            for row in ann_csv:
                # Only consider shelf class
                if len(row) > 4 and int(row[0]) == 5:
                    row[0] = 1
                    image_annotations.append(row)

        rand_num = random.random()
        if rand_num < 0.2:
            val_image_fname = result_image_fname.replace('train/', 'valid/')
            val_ann_fname = result_ann_fname.replace('train/', 'valid/')
            shutil.copy(image_fname, val_image_fname)
            with open(val_ann_fname, 'w') as f:
                csvwriter = csv.writer(f)
                csvwriter.writerows(image_annotations)
        else:
            shutil.copy(image_fname, result_image_fname)
            with open(result_ann_fname, 'w') as f:
                csvwriter = csv.writer(f)
                csvwriter.writerows(image_annotations)

    # Write data.yaml file
    LOGGER.info('Writing data.yaml')
    labels = ['background', 'shelf']
    labels_cnt = len(labels)
    labels = json.dumps(labels)
    yaml_data = [
        f'train: {RESULTS.name}/train\n',
        f'val: {RESULTS.name}/valid\n\n',
        f'nc: {labels_cnt}\n',
        f'names: {labels}\n'
    ]
    with open(RESULTS/'data.yaml', 'w') as f:
        f.writelines(yaml_data)

    LOGGER.info(f'✅ Yolov5 Shelf dataset creation complete: {RESULTS}')


def create_object_dataset(opt, images, annotations, categories):
    pass


def main(opt):
    if opt.aml_json is not None and os.path.exists(opt.aml_json):
        LOGGER.info('Using AML json file: {}'.format(opt.aml_json))
    else:
        LOGGER.error('Unable to find JSON file: {}'.format(opt.aml_json))
        return -1

    # Check if any actions are specified.
    if not (opt.download_images or opt.create_dataset or opt.create_shelf_dataset or opt.create_object_dataset) is True:
        LOGGER.info('Error: no actions specify.')
        return -1

    # Read the data information
    images, annotations, categories = _read_json(opt)

    if opt.download_images is True:
        download_dataset(opt, images, annotations, categories)

    if opt.create_dataset is True:
        create_complete_dataset(opt, images, annotations, categories)

    if opt.create_shelf_dataset is True:
        create_shelf_dataset(opt, images, annotations, categories)

    if opt.create_object_dataset is True:
        create_object_dataset(opt, images, annotations, categories)



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aml-json', type=str, default=ROOT/'azure-export-data.json', help='Path to input JSON file')
    parser.add_argument('--download-images', action='store_true', help='Download images specified in JSON')
    parser.add_argument('--create-dataset', action='store_true', help='Create complete yolov5 dataset (all classes)')
    parser.add_argument('--create-shelf-dataset', action='store_true', help='Create shelf-dataset')
    parser.add_argument('--create-object-dataset', action='store_true', help='Create object-dataset')
    parser.add_argument('--results', type=str, default=ROOT/'results', help='Results dataset location.')
    parser.add_argument('--threads', type=int, default=100, help='Download dataset max threads')
    opt = parser.parse_args()
    opt.create_shelf_dataset = True
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)


