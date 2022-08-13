import json
import pandas as pd
from utils import *
import json
import os
import tqdm
from urllib.request import urlretrieve
import threading
import yaml
import random

def download_file_per_thread(url, file_name):
    urlretrieve(url, file_name)

def process_aml_json(json_file):
    print('Reading exported Azure data')
    with open(json_file, 'r') as f:
        data = json.load(f)

    print('Create yolov5 folder structure')
    shutil.rmtree('results/', ignore_errors=True)

    os.makedirs('results')
    os.chdir('results')
    os.makedirs('train')
    os.makedirs('valid')
    os.makedirs('train/images')
    os.makedirs('train/labels')
    os.makedirs('valid/images')
    os.makedirs('valid/labels')

    print('Downloading images')
    threads = []
    idx = 0
    for image in tqdm.tqdm(data['images']):
        idx+=1
        image['file_name'] = os.path.basename(image['file_name'])
        # print(image)
        del image['coco_url']
        url = image['absolute_url']
        del image['absolute_url']

        rand_num = random.random()
        if rand_num < 0.8:
            image['file_name'] = 'train/images/' + image['file_name']
        else:
            image['file_name'] = 'valid/images/' + image['file_name']

        t = threading.Thread(target=download_file_per_thread, args=(url, image['file_name']))
        t.daemon = True
        threads.append(t)
        t.start()

        # After 100 threads, wait till all finish.
        if idx % 100 == 0:
            for t in threads:
                t.join()
            threads = []

    # Create image dict
    images = {'%g' % x['id']: x for x in data['images']}

    # Class to label mapping (convert everything to shelf, bottle, void)
    cls_to_label = {
        1: 1, # bottle
        2: 1, # can
        3: 1, # carton
        4: 1, # container
        5: 2, # void
        6: 0, # shelf
        7: 1, # packet
        8: 1, # box
        9: 1, # cup
    }

    # Write labels file
    print('Writing annotations')
    for x in tqdm.tqdm(data['annotations']):
        img = images['%g' % x['image_id']]
        f = img['file_name'].replace('images/', 'labels/')

        # The COCO box format is [top left x, top left y, width, height]
        box = np.array(x['bbox'], dtype=np.float64)

        # Yolov5 format is [center left x, center left y, width, height]
        box[:2] += box[2:] / 2

        # Write
        if box[2] > 0 and box[3] > 0:  # if w > 0 and h > 0
            cls = cls_to_label[x['category_id']] # label

            # Map classes
            line = cls, *(box)  # cls, box or segments
            with open(Path(f).with_suffix('.txt'), 'a') as file:
                file.write(('%g ' * len(line)).rstrip() % line + '\n')

    # Write data.yaml file
    print('Writing data.yaml')
    labels = ['shelf', 'bottle', 'void']
    yaml_data = [
        'train: "results/train"\n',
        'val: "results/valid"\n',
        'nc: 3\n',
        "names: {}".format(str(labels).replace('\'', '"'))
    ]
    with open('data.yaml', 'w') as f:
        f.writelines(yaml_data)

    os.chdir('..')
    print('Done.')

if __name__ == '__main__':
    process_aml_json('azure-export-data.json')


