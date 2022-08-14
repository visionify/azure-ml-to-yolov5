import os
import sys
import logging
from pathlib import Path
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def set_logging(name=None):
    level = logging.INFO
    log = logging.getLogger(name)
    log.setLevel(level)
    handler = logging.StreamHandler()
    FORMAT = logging.Formatter("%(asctime)s - %(levelname).3s - %(name)22s:%(lineno)3d - %(message)s", "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(FORMAT)
    handler.setLevel(level)
    log.addHandler(handler)

set_logging()  # run before defining LOGGER
LOGGER = logging.getLogger("AMLPipeline")  # define globally (used in train.py, val.py, detect.py, etc.)
