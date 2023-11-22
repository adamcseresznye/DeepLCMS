import os
import random
from pathlib import Path

import numpy as np


class Configuration:
    img_path = Path("/content/experiment")
    train_dir = img_path / "train"
    val_dir = img_path / "val"
    test_dir = img_path / "test"
    seed = 2643
