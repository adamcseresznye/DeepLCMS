import os
import random
from pathlib import Path

import numpy as np


class Configuration:
    VER = 1
    RAW_DATA_PATH = Path(__file__).parents[2].joinpath("data/raw")
    INTERIM_DATA_PATH = Path(__file__).parents[2].joinpath("data/interim")
    seed = 2643


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
