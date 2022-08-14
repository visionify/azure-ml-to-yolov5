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
from urllib.request import urlretrieve
import threading

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

def _delete_and_download_images(opt, images):
    LOGGER.info('Deleting results folder')
    RESULTS = Path.resolve(opt.results)

    # # Delete dataset
    if os.path.exists(RESULTS):
        shutil.rmtree(RESULTS, ignore_errors=True)

    # Create yolov5 directory structure
    LOGGER.info('Create results yolov5 folder structure')
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
        img_abspath = RESULTS / f'train/images/{image["name"]}'
        t = threading.Thread(target=_download_file_per_thread, args=(image['url'], img_abspath))
        t.daemon = True
        threads.append(t)
        t.start()

        # Parallelize using the number of threads specified
        if tid % opt.threads == 0:
            for t in threads:
                t.join()
            threads = []

def download_dataset(opt, images, annotations, categories):
    # Internal function for threaded images download.

    RESULTS = Path.resolve(opt.results)
    _delete_and_download_images(opt, images)

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

        f = RESULTS / 'train/labels' / img['name']

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
    yaml_data = [
        f'train: "{RESULTS.name}/train"\n',
        f'val: "{RESULTS.name}/valid"\n',
        f'nc: {labels_cnt}\n',
        f'names: {labels}\n'
    ]
    with open(RESULTS/'data.yaml', 'w') as f:
        f.writelines(yaml_data)

    LOGGER.info(f'✅ Results available at: {RESULTS}')

def main(opt):
    if opt.aml_json is not None and os.path.exists(opt.aml_json):
        LOGGER.info('Using AML json file: {}'.format(opt.aml_json))
    else:
        LOGGER.error('Unable to find JSON file: {}'.format(opt.aml_json))
        return -1

    # Read the data information
    images, annotations, categories = _read_json(opt)

    if opt.force_download is True:
        download_dataset(opt, images, annotations, categories)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aml-json', type=str, default=ROOT/'azure-export-data.json', help='Path to input JSON file')
    parser.add_argument('--force-download', action='store_true', help=' Force re-download of images based on JSON')
    parser.add_argument('--shelf-dataset', action='store_true', help='Create shelf-dataset')
    parser.add_argument('--object-dataset', action='store_true', help='Create object-dataset')
    parser.add_argument('--results', type=str, default=ROOT/'results', help='Results dataset location.')
    parser.add_argument('--threads', type=int, default=100, help='Download dataset max threads')
    opt = parser.parse_args()
    opt.force_download=True  # Debugging only.
    opt.threads = 200
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)


