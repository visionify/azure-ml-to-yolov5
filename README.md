# Azure ML to Yolov5 Pipeline

## Usage

```bash
python3 az2yolo.py --help

usage: az2yolo.py [-h] [--json JSON] [--clean] [--download-images] [--create-shelf-dataset] [--create-object-dataset] [--threads THREADS]

optional arguments:
  -h, --help            show this help message and exit
  --json JSON           Path to input JSON file
  --clean               Clean all past datasets
  --download-images     Download images specified in JSON
  --create-shelf-dataset
                        Create shelf-dataset
  --create-object-dataset
                        Create object-dataset
  --threads THREADS     Max threads (only works for --download-images)
  ```

- Start by cleaning any prior datasets.

```bash
python3 az2yolo.py --clean
```

- A starting JSON file is needed to start this pipeline. This can be downloaded via Azure Machine Learning Studio - and export the dataset as a COCO JSON Format. Please make sure the Blob Storage provides anonymous read access to the images. Copy over the JSON file in this folder.

- Run this command to download all the images mentioned in the COCO json dataset & create a yolov5 dataset from it.

```bash
python3 az2yolo.py --json data.json --download-images
```

- Create a shelf dataset. (Only use shelf labels)

```bash
python3 az2yolo.py --json data.json --create-shelf-dataset
```

- Run this command to create an object dataset. (Crop shelves and renormalize other classes to match shelf boundaries)

```bash
python3 az2yolo.py --json data.json --create-object-dataset
```

- The results are in these folders (Original download: `download`, Full Yolov5 dataset: `results`, Shelf dataset: `shelf-dataset`, Object dataset: `object-dataset`)

- Questions: [hmurari@visionify.ai](mailto:hmurari@visionify.ai)
