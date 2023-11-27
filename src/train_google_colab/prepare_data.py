import os
import random
from pathlib import Path

import colab_utils
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torchvision
from timm.data import create_transform, resolve_data_config
from torchvision.transforms import Compose, RandomRotation


def get_timm_transforms(
    model,
    preprocess_train: Compose = None,
    color_jitter: float = 0.05,
    re_prob: float = 0.05,
) -> tuple[Compose, Compose, Compose]:
    """
    Get preprocessing transforms for training, testing, and validation based on
    the data configuration of the model.

    Args:
        model: The timm model.
        preprocess_train (Compose, optional): Optional existing training transforms.
        color_jitter (float): Intensity of color jittering. Default is 0.05.
        re_prob (float): Probability of applying random erasing. Default is 0.05.

    Returns:
        Tuple[Compose, Compose, Compose]: A tuple containing the preprocessing transforms
        for training, validation, and testing.
    """
    # Resolve data configuration for the model
    data_cfg = resolve_data_config(model.model.default_cfg)

    if preprocess_train is None:
        # Create a list of transforms
        preprocess_train = create_transform(
            input_size=data_cfg["input_size"],
            is_training=True,
            color_jitter=color_jitter,
            hflip=0,
            vflip=0,
            scale=(1.0, 1.0),  # Remove random zoom effect
            mean=data_cfg["mean"],
            std=data_cfg["std"],
            re_prob=re_prob,
        )
        # Add the RandomRotation transform to the list of transforms
        preprocess_train.transforms.insert(0, RandomRotation(1))

    # Create the transform object for testing
    preprocess_test = create_transform(
        **data_cfg,
        is_training=False,
    )

    # Create the transform object for validation
    preprocess_val = create_transform(
        **data_cfg,
        is_training=False,
    )
    return preprocess_train, preprocess_val, preprocess_test


def get_dataloaders(
    train_dir: str = colab_utils.Configuration.train_dir,
    val_dir: str = colab_utils.Configuration.val_dir,
    test_dir: str = colab_utils.Configuration.test_dir,
    batch_size: int = colab_utils.Configuration.batch_size,
    preprocess_train: torchvision.transforms.Compose = None,
    preprocess_val: torchvision.transforms.Compose = None,
    preprocess_test: torchvision.transforms.Compose = None,
) -> tuple:
    """
        Get dataloaders for training, validation, and testing.
    s
        Args:
            train_dir (str): Path to the training data directory.
            val_dir (str): Path to the validation data directory.
            test_dir (str): Path to the test data directory.
            batch_size (int): Number of samples per batch.
            preprocess_train (torchvision.transforms.Compose): Preprocessing
            transforms for training data.
            preprocess_val (torchvision.transforms.Compose): Preprocessing
            transforms for validation data.
            preprocess_test (torchvision.transforms.Compose): Preprocessing
            transforms for test data.

        Returns:
            tuple: A tuple containing the training, validation, and test dataloaders.
    """
    NUM_WORKERS = os.cpu_count()

    train_data = torchvision.datasets.ImageFolder(
        root=train_dir,
        transform=preprocess_train,
        target_transform=None,
    )

    val_data = torchvision.datasets.ImageFolder(
        root=val_dir,
        transform=preprocess_val,
        target_transform=None,
    )

    test_data = torchvision.datasets.ImageFolder(
        root=test_dir,
        transform=preprocess_test,
        target_transform=None,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_dataloader, val_dataloader, test_dataloader


def inspect_dataloader(
    dataloader: torch.utils.data.DataLoader, save: bool = True
) -> None:
    """
    Visualize a batch of images from a dataloader.

    Args:
        dataloader (DataLoader): The dataloader to visualize.
        save (bool): If True, save the visualization as "transformed_grid.png". Default is True.
    """
    images, labels = next(iter(dataloader))
    grid = torchvision.utils.make_grid(images)

    plt.figure(figsize=(15, 25))

    img = plt.imshow(grid.permute(1, 2, 0)).figure
    plt.axis("off")
    plt.tight_layout()

    if save:
        img.savefig("transformed_grid.png", dpi=300)
