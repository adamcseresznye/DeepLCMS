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
    batch_size = 32
    pretained_model_name = "convnext_large_mlp.clip_laion2b_augreg_ft_in1k_384"
    learning_rate = 0.006918309709189364
