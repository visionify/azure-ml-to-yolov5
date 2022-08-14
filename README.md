# Azure ML to Yolov5 Pipeline.

## Usage

```bash
$ python3 aml-to-yolov5.py --help
usage: aml-to-yolov5.py [-h] [--aml-json AML_JSON] [--force-download] [--shelf-dataset] [--object-dataset] [--results RESULTS] [--threads THREADS]

optional arguments:
  -h, --help           show this help message and exit
  --aml-json AML_JSON  Path to input JSON file
  --force-download     Force re-download of images based on JSON
  --shelf-dataset      Create shelf-dataset
  --object-dataset     Create object-dataset
  --results RESULTS    Results dataset location.
  --threads THREADS    Download dataset max threads
```

- A starting JSON file is needed to start this pipeline. This can be downloaded via Azure Machine Learning Studio - and export the dataset as a COCO JSON Format. Please make sure the Blob Storage provides anonymous read access to the images.

- Run this command to download all the images.

```bash
python3 aml-to-yolov5.py --aml-json aml-export.json --download-images
```

- Run this command to create Yolov5 annotations

```bash
python3 aml-to-yolov5.py --aml-json aml-export.json --annotate
```

- Run this command to create a shelf dataset

```bash
python3 aml-to-yolov5.py --aml-json aml-export.json --create-shelf-dataset
```

- Run this command to create an object model dataset (shelves are cropped, and object bounding boxes are normalized)

```bash
python3 aml-to-yolov5.py --aml-json aml-export.json --create-object-dataset
```

- The results are alwyas in specified locations (Full dataset: `results`, Shelf dataset: `shelf-dataset`, Object dataset: `object-dataset`)
