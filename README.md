# Azure ML to Yolov5 Model Creation

## Download Data

- Go to Azure ML studio and export the dataset as a COCO JSON format.

- Once downloaded, rename the file as `azure-export-data.json` and put it in this directory.

- Activate the conda environment where your conda/venv dependencies are installed. For example:

```bash
conda activate harsh
```

- Run this command:

```bash
python3 aml-to-yolov5.py
```

- Yolov5 formatted dataset will be available under `results` folder.

## Train yolov5 model

- Clone yolov5 folder

```bash
cd ..
git clone https://github.com/ultralytics/yolov5.git yolov5-06-29
```

- Move the results dataset into yolov5 folder.

```bash
mv azure-ml-to-yolo5/results yolov5-06-29/beverages-06-29
```

- Rename the training/validation set location names in data.yaml file.

```bash
train: "results/train"
val: "results/valid"

to:

train: "beverages-06-29/train"
val: "beverages-06-29/valid"
```

- Train the model

```bash
cd yolov5-06-29
python3 train.py --data beverages-06-29/data.yaml --cfg models/yolov5s.yaml --weights yolov5s.pt --batch-size 64 --device 0 --epochs 120
```

- Evaluate performance on wandb.
