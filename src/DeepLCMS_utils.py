import os
import random
from pathlib import Path

import numpy as np


class Configuration:
    VER = 1
    RAW_DATA_PATH = Path(__file__).parents[1].joinpath("data/raw")
    INTERIM_DATA_PATH = Path(__file__).parents[1].joinpath("data/interim")
