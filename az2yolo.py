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
import pandas as pd
from PIL import Image
import multiprocessing

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # az2yolo root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.general import LOGGER

# Paths
DOWNLOAD = Path('download')
SHELF_OBJECT = Path('shelf-object-dataset')
OBJECT = Path('object-dataset')
SHELF = Path('shelf-dataset')

def _download_file_per_thread(url, file_name):
    urlretrieve(url, file_name)

def _read_json(opt):
    with open(opt.json, 'r') as f:
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

def delete_dataset(opt):
    # Internal function for threaded images download.

    LOGGER.info('Deleting download folder')
    if os.path.exists(DOWNLOAD):
        shutil.rmtree(DOWNLOAD, ignore_errors=True)

    LOGGER.info('Deleting results folder')
    if os.path.exists(SHELF_OBJECT):
        shutil.rmtree(SHELF_OBJECT, ignore_errors=True)

    LOGGER.info('Deleting shelf-dataset')
    SHELF = Path('shelf-dataset')
    if os.path.exists(SHELF):
        shutil.rmtree(SHELF, ignore_errors=True)

    LOGGER.info('Deleting object-dataset')
    OBJECT = Path('object-dataset')
    if os.path.exists(OBJECT):
        shutil.rmtree(OBJECT, ignore_errors=True)

    LOGGER.info(f'✅ Done.')


def download_dataset(opt, images, annotations, categories):
    # Internal function for threaded images download.

    LOGGER.info('Deleting downloads folder')

    # Delete dataset
    if os.path.exists(DOWNLOAD):
        shutil.rmtree(DOWNLOAD, ignore_errors=True)

    if os.path.exists(SHELF_OBJECT):
        shutil.rmtree(SHELF_OBJECT, ignore_errors=True)


    # Create yolov5 directory structure
    LOGGER.info('Create results yolov5 folder structure')
    os.makedirs(DOWNLOAD)
    os.makedirs(DOWNLOAD/'train')
    os.makedirs(DOWNLOAD/'train/images')
    os.makedirs(DOWNLOAD/'train/labels')

    os.makedirs(SHELF_OBJECT)
    os.makedirs(SHELF_OBJECT/'train')
    os.makedirs(SHELF_OBJECT/'train/images')
    os.makedirs(SHELF_OBJECT/'train/labels')
    os.makedirs(SHELF_OBJECT/'valid')
    os.makedirs(SHELF_OBJECT/'valid/images')
    os.makedirs(SHELF_OBJECT/'valid/labels')

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
        f'train: {SHELF_OBJECT.name}/train\n',
        f'val: {SHELF_OBJECT.name}/valid\n\n',
        f'nc: {labels_cnt}\n',
        f'names: {labels}\n'
    ]
    with open(SHELF_OBJECT/'data.yaml', 'w') as f:
        f.writelines(yaml_data)

    # Split dataset between train & val
    LOGGER.info('Splitting dataset')
    for image_fname in tqdm.tqdm(glob.glob(f'{DOWNLOAD.name}/train/images/*.jpg')):
        result_image_fname = image_fname.replace(DOWNLOAD.name, SHELF_OBJECT.name)
        ann_fname = image_fname.replace('images/', 'labels/').replace('.jpg', '.txt')
        result_ann_fname = ann_fname.replace(DOWNLOAD.name, SHELF_OBJECT.name)

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


    LOGGER.info(f'✅ Yolov5 Shelf-Object Dataset creation complete: {SHELF_OBJECT}')

def create_shelf_dataset(opt):
    # Basically creating shelf dataset is similar to creating entire dataset.
    # We just need to ignore all the object annotations. We will only focus on
    # Shelf annotations.
    SHELF = Path('shelf-dataset')

    if os.path.exists(SHELF):
        shutil.rmtree(SHELF, ignore_errors=True)
    os.makedirs(SHELF)
    os.makedirs(SHELF/'train')
    os.makedirs(SHELF/'train/images')
    os.makedirs(SHELF/'train/labels')
    os.makedirs(SHELF/'valid')
    os.makedirs(SHELF/'valid/images')
    os.makedirs(SHELF/'valid/labels')

    for image_fname in tqdm.tqdm(glob.glob(f'{DOWNLOAD.name}/train/images/*.jpg')):
        ann_fname = image_fname.replace('images/', 'labels/').replace('.jpg', '.txt')
        result_image_fname = image_fname.replace(DOWNLOAD.name, SHELF.name)
        result_ann_fname = ann_fname.replace(DOWNLOAD.name, SHELF.name)

        if (not os.path.exists(image_fname) or
            not os.path.exists(ann_fname)):
            LOGGER.error(f'Image/Annotation does not exist: {image_fname}: {ann_fname}')

        data = pd.read_csv(ann_fname, names=['clsid', 'x', 'y', 'w', 'h'], sep='\s+')

        # Filter out only shelf class
        shelf_data = data[data['clsid'] == 5].copy()
        shelf_data['clsid'] = 1

        rand_num = random.random()
        if rand_num < 0.2:
            val_image_fname = result_image_fname.replace('train/', 'valid/')
            val_ann_fname = result_ann_fname.replace('train/', 'valid/')
            shutil.copy(image_fname, val_image_fname)
            shelf_data.to_csv(val_ann_fname, index=False, header=False, sep=' ', float_format='%.5f')
        else:
            shutil.copy(image_fname, result_image_fname)
            shelf_data.to_csv(result_ann_fname, index=False, header=False, sep=' ', float_format='%.5f')

    # Write data.yaml file
    LOGGER.info('Writing data.yaml')
    labels = ['background', 'shelf']
    labels_cnt = len(labels)
    labels = json.dumps(labels)
    yaml_data = [
        f'train: {SHELF.name}/train\n',
        f'val: {SHELF.name}/valid\n\n',
        f'nc: {labels_cnt}\n',
        f'names: {labels}\n'
    ]
    with open(SHELF/'data.yaml', 'w') as f:
        f.writelines(yaml_data)

    LOGGER.info(f'✅ Yolov5 Shelf dataset creation complete: {SHELF}')

def _xywhn2xyxy(x, w=1280, h=960, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[0] = w * (x[0] - x[2] / 2) + padw  # top left x
    y[1] = h * (x[1] - x[3] / 2) + padh  # top left y
    y[2] = w * (x[0] + x[2] / 2) + padw  # bottom right x
    y[3] = h * (x[1] + x[3] / 2) + padh  # bottom right y
    return y

def _xyxy2xywh(x, w=1280, h=960):
    # Convert x4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[0] = ((x[0] + x[2]) / 2) / w  # x center
    y[1] = ((x[1] + x[3]) / 2) / h  # y center
    y[2] = (x[2] - x[0]) / w  # width
    y[3] = (x[3] - x[1]) / h  # height
    return y

def _new_obj_coords(obj, sh):
    # Return new shelf image based coordinates if object within shelf.
    # Else return None.
    clsid = obj[0]
    o = _xywhn2xyxy(obj[1:])
    s  = _xywhn2xyxy(sh[1:])
    if (o[0] >= s[0] and
        o[1] >= s[1] and
        o[2] <= s[2] and
        o[3] <= s[3]):
        # translate the bbox
        p = o[0]-s[0], o[1]-s[1], o[2]-s[0], o[3]-s[1]
        w, h = s[2] - s[0], s[3] - s[1]
        q = _xyxy2xywh(p, w, h)
        return np.array([clsid, q[0], q[1], q[2], q[3]])
    else:
        return None

def _process_img_obj_dataset(image_fname):

    ann_fname = image_fname.replace('images/', 'labels/').replace('.jpg', '.txt')
    if (not os.path.exists(image_fname) or
        not os.path.exists(ann_fname)):
        LOGGER.error(f'Image/Annotation does not exist: {image_fname}: {ann_fname}')

    # TXT File annotations:
    # 0 - bottle
    # 1 - can
    # 2 - carton
    # 3 - container
    # 4 - void
    # 5 - shelf
    # 6 - packet
    # 7 - box
    # 8 - cup

    # New labels for objects:
    # 0 - background
    # 1 - bottle
    # 2 - void

    # Read shelf annotations
    data = pd.read_csv(ann_fname, names=['clsid', 'x', 'y', 'w', 'h'], sep='\s+')
    shelves = data[data['clsid'] == 5].copy()
    voids = data[data['clsid'] == 4].copy()
    voids['clsid'] = 2
    objects = data[(data['clsid'] != 4) & (data['clsid'] != 5)].copy()
    objects['clsid'] = 1
    objvoids = pd.concat([objects, voids], ignore_index=True)
    image = Image.open(image_fname)

    # Sort the shelves based on y coordinate
    shelves.sort_values(by='y')
    for idx, shelf in enumerate(shelves.values):
        shelf_ann_fname = ann_fname.replace('.txt', f'-shelf-{idx}.txt')
        shelf_img_fname = image_fname.replace('.jpg', f'-shelf-{idx}.jpg')
        shelf_bb = _xywhn2xyxy(shelf[1:])
        shelf_image = image.crop(shelf_bb)
        shelf_annotations = np.empty((0, 5))
        for obj in objvoids.values:
            new_obj = _new_obj_coords(obj, shelf)
            if new_obj is not None:
                shelf_annotations = np.append(shelf_annotations, np.reshape(new_obj, (1, 5)), axis=0)

        rand_num = random.random()
        if rand_num < 0.2:
            val_image_fname = shelf_img_fname.replace('train/', 'valid/').replace(DOWNLOAD.name, OBJECT.name)
            val_ann_fname = shelf_ann_fname.replace('train/', 'valid/').replace(DOWNLOAD.name, OBJECT.name)
            shelf_image.save(val_image_fname)
            np.savetxt(val_ann_fname, shelf_annotations, fmt='%d %.5f %.5f %.5f %.5f')
        else:
            shelf_img_fname = shelf_img_fname.replace(DOWNLOAD.name, OBJECT.name)
            shelf_ann_fname = shelf_ann_fname.replace(DOWNLOAD.name, OBJECT.name)
            shelf_image.save(shelf_img_fname)
            np.savetxt(shelf_ann_fname, shelf_annotations, fmt='%d %.5f %.5f %.5f %.5f')

def create_object_dataset(opt):
    # Creating object dataset is tricky.
    # First we need to crop the shelf - each shelf will have its own annotations.
    # Then adjust each object to match the new annotations.

    # Create fresh dataset each time.
    if os.path.exists(OBJECT):
        shutil.rmtree(OBJECT, ignore_errors=True)

    os.makedirs(OBJECT)
    os.makedirs(OBJECT/'train')
    os.makedirs(OBJECT/'train/images')
    os.makedirs(OBJECT/'train/labels')
    os.makedirs(OBJECT/'valid')
    os.makedirs(OBJECT/'valid/images')
    os.makedirs(OBJECT/'valid/labels')

    LOGGER.info('Creating object/void dataset')

    with multiprocessing.Pool(processes=4) as pool:
        pool.map(_process_img_obj_dataset, tqdm.tqdm(glob.glob(f'{DOWNLOAD.name}/train/images/*.jpg')))

    # Write data.yaml file
    LOGGER.info('Writing data.yaml')
    labels = ['background', 'bottle', 'void']
    labels_cnt = len(labels)
    labels = json.dumps(labels)
    yaml_data = [
        f'train: {OBJECT.name}/train\n',
        f'val: {OBJECT.name}/valid\n\n',
        f'nc: {labels_cnt}\n',
        f'names: {labels}\n'
    ]
    with open(OBJECT/'data.yaml', 'w') as f:
        f.writelines(yaml_data)

    LOGGER.info(f'✅ Yolov5 Object dataset creation complete: {OBJECT}')

def main(opt):
    if opt.clean is True:
        delete_dataset(opt)
        return

    if opt.json is not None and os.path.exists(opt.json):
        LOGGER.info('Using AML json file: {}'.format(opt.json))
    else:
        LOGGER.error('Unable to find JSON file: {}'.format(opt.json))
        return -1

    # Check if any actions are specified.
    if not (opt.download_images or opt.create_shelf_dataset or opt.create_object_dataset) is True:
        LOGGER.info('Error: no actions specify.')
        return -1

    # Read the data information
    images, annotations, categories = _read_json(opt)

    if opt.download_images is True:
        download_dataset(opt, images, annotations, categories)
        create_complete_dataset(opt, images, annotations, categories)

    if opt.create_shelf_dataset is True:
        create_shelf_dataset(opt)

    if opt.create_object_dataset is True:
        create_object_dataset(opt)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, default=ROOT/'data.json', help='Path to input JSON file')
    parser.add_argument('--clean', action='store_true', help='Clean all past datasets')
    parser.add_argument('--download-images', action='store_true', help='Download images specified in JSON')
    parser.add_argument('--create-shelf-dataset', action='store_true', help='Create shelf-dataset')
    parser.add_argument('--create-object-dataset', action='store_true', help='Create object-dataset')
    parser.add_argument('--threads', type=int, default=100, help='Max threads (only works for --download-images)')
    opt = parser.parse_args()

    # To test for debugging, uncomment one of these & you can step through
    # opt.create_shelf_dataset = True
    # opt.create_object_dataset = True

    return opt

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)


