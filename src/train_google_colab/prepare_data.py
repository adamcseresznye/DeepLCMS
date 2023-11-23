import os
import random
from pathlib import Path

import colab_utils
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torchvision


def get_timm_transforms(model):
    """
    Get preprocessing transforms for training, testing, and validation based on
    the data configuration of the model.

    Args:
        model: The timm model.

    Returns:
        Tuple: A tuple containing the preprocessing transforms for training,
        validation and testing.
    """
    # Resolve data configuration for the model
    data_cfg = timm.data.resolve_data_config(model.model.default_cfg)

    # Create the transform object for training
    preprocess_train = timm.data.create_transform(
        **data_cfg,
        is_training=False,
        # Add any additional parameters for training transforms if needed
    )

    # Create the transform object for testing
    preprocess_test = timm.data.create_transform(
        **data_cfg,
        is_training=False,
        # Add any additional parameters for testing transforms if needed
    )

    # Create the transform object for validation
    preprocess_val = timm.data.create_transform(
        **data_cfg,
        is_training=False,
        # Add any additional parameters for validation transforms if needed
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
