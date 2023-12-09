import os
from pathlib import Path
from typing import Any, Tuple

import colab_functions
import colab_utils
import matplotlib.pyplot as plt
import prepare_data
import timm
import torch
import torchmetrics
import torchvision
from lightning.pytorch import LightningDataModule


class LCMSDataModule(LightningDataModule):
    def __init__(
        self,
        model: Any,
        data_dir: Path = colab_utils.Configuration.img_path,
        batch_size: int = colab_utils.Configuration.batch_size,
        color_jitter: float = 0.2,
        re_prob: float = 0.2,
    ) -> None:
        """
        LightningDataModule for handling LCMS (Liquid Chromatography-Mass Spectrometry) data.

        Args:
            model (Any): The model to be used with the data.
            data_dir (str): Root directory for all data. Defaults to colab_utils.Configuration.img_path.
            batch_size (int): Size of each batch. Defaults to colab_utils.Configuration.batch_size.
            color_jitter (float): Intensity of color jitter transformation. Defaults to 0.2.
            re_prob (float): Probability of applying random erasing transformation. Defaults to 0.2.
        """
        super().__init__()
        self.model = model
        self.data_dir = data_dir
        self.train_dir = data_dir / "train"
        self.val_dir = data_dir / "val"
        self.test_dir = data_dir / "test"
        self.batch_size = batch_size
        self.color_jitter = color_jitter
        self.re_prob = re_prob

    def setup(self, stage: str) -> None:
        """
        Setup method to be called before training, validation, and testing.
        This method is empty in this implementation.
        """
        pass

    def get_timm_transforms(self) -> Tuple[Any, Any, Any]:
        """
        Get torchvision and timm transforms for training, validation, and testing.

        Returns:
            Tuple of torchvision and timm transforms for training, validation, and testing.
        """
        # Resolve data configuration for the model
        data_cfg = timm.data.resolve_data_config(self.model.model.default_cfg)

        preprocess_train = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomRotation(10),
                timm.data.create_transform(
                    input_size=data_cfg["input_size"],
                    is_training=True,
                    color_jitter=self.color_jitter,
                    hflip=0,
                    vflip=0,
                    scale=(1.0, 1.0),  # Remove random zoom effect
                    mean=data_cfg["mean"],
                    std=data_cfg["std"],
                    re_prob=self.re_prob,
                ),
            ]
        )

        preprocess_test = timm.data.create_transform(
            **data_cfg,
            is_training=False,
        )

        preprocess_val = timm.data.create_transform(
            **data_cfg,
            is_training=False,
        )

        return preprocess_train, preprocess_val, preprocess_test

    def create_imagefolders(self) -> Tuple[Any, Any, Any]:
        """
        Create ImageFolder datasets for training, validation, and testing.

        Returns:
            Tuple of ImageFolder datasets for training, validation, and testing.
        """
        preprocess_train, preprocess_val, preprocess_test = self.get_timm_transforms()

        train_data = torchvision.datasets.ImageFolder(
            root=self.train_dir,
            transform=preprocess_train,
            target_transform=None,
        )

        val_data = torchvision.datasets.ImageFolder(
            root=self.val_dir,
            transform=preprocess_val,
            target_transform=None,
        )

        test_data = torchvision.datasets.ImageFolder(
            root=self.test_dir,
            transform=preprocess_test,
            target_transform=None,
        )

        return train_data, val_data, test_data

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create DataLoader for training data.

        Returns:
            DataLoader for training data.
        """
        train_data, _, _ = self.create_imagefolders()
        num_workers = os.cpu_count()

        train_dataloader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return train_dataloader

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create DataLoader for validation data.

        Returns:
            DataLoader for validation data.
        """
        _, val_data, _ = self.create_imagefolders()
        num_workers = os.cpu_count()

        val_dataloader = torch.utils.data.DataLoader(
            val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        return val_dataloader

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create DataLoader for test data.

        Returns:
            DataLoader for test data.
        """
        _, _, test_data = self.create_imagefolders()
        num_workers = os.cpu_count()

        test_dataloader = torch.utils.data.DataLoader(
            test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        return test_dataloader


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
        plt.savefig("transformed_grid.png", bbox_inches="tight", pad_inches=0, dpi=300)
