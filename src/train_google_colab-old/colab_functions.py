import os
import random
from pathlib import Path
from typing import Tuple

import colab_utils
import matplotlib.pyplot as plt
import torch
import torchvision
from PIL import Image


def open_random_image(
    img_path: Path = colab_utils.Configuration.img_path,
) -> Image.Image:
    """
    Opens a random image from the specified path, prints information about the image,
    resizes it to 300x300, and returns the image object.

    Parameters:
    - img_path (Path): The path where the images are located. Defaults to colab_utils.Configuration.img_path.

    Returns:
    Tuple[str, int, int, Image.Image]: A tuple containing:
        - The class of the random image.
        - Height of the image.
        - Width of the image.
        - The PIL Image object.

    Example:
    >>> open_random_image()
    ('class_name', 300, 300, <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=300x300 at 0x...>)
    """
    image_path_list = list(img_path.glob("*/*/*.jpeg"))
    random_image_path = random.choice(image_path_list)
    random_image_path_class = random_image_path.parent.stem

    img = Image.open(random_image_path).resize((300, 300))

    print(f"Image class: {random_image_path_class}")
    print(f"Image height: {img.height}")
    print(f"Image width: {img.width}")

    return img


def inspect_directory(
    train_dir: Path = colab_utils.Configuration.train_dir,
    test_dir: Path = colab_utils.Configuration.test_dir,
    val_dir: Path = colab_utils.Configuration.val_dir,
):
    """
    Inspects the number of items in the specified directories.

    Parameters:
    - train_dir (Path): The path to the training directory.
    - test_dir (Path): The path to the test directory.
    - val_dir (Path): The path to the validation directory.

    Returns:
    None
    """
    print("Number of items in the train directory:")
    for pth in train_dir.iterdir():
        print(f"{pth}: {len(list(pth.glob('*.jpeg')))}")

    print("\nNumber of items in the test directory:")
    for pth in test_dir.iterdir():
        print(f"{pth}: {len(list(pth.glob('*.jpeg')))}")

    print("\nNumber of items in the validation directory:")
    for pth in val_dir.iterdir():
        print(f"{pth}: {len(list(pth.glob('*.jpeg')))}")


import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision


def inspect_training_images(
    train_dir: Path = colab_utils.Configuration.train_dir, save: bool = True
):
    """
    Inspects and displays random training images.

    Parameters:
    - train_dir (Path): The path to the training directory.
    - save (bool): Whether to save the displayed images to a file. Defaults to True.

    Returns:
    None
    """
    # Load samples from the ImageFolder
    loaded_samples = torchvision.datasets.ImageFolder(
        root=train_dir,
        transform=torchvision.transforms.ToTensor(),
    )

    # Create a 3x3 grid for displaying images
    fig = plt.figure(figsize=(9, 9))
    rows, cols = 3, 3

    # Set a random seed for reproducibility
    random.seed(colab_utils.Configuration.seed)

    for i in range(1, rows * cols + 1):
        fig.add_subplot(rows, cols, i)

        # Choose a random index
        random_index = random.randint(0, len(loaded_samples) - 1)

        # Display the image
        plt.imshow(loaded_samples[random_index][0].permute(1, 2, 0))

        # Get the class label for the image
        class_label = [
            item
            for item, key in loaded_samples.class_to_idx.items()
            if key == loaded_samples[random_index][1]
        ][0]

        plt.title(class_label)
        plt.axis(False)

    # Save the figure if specified
    if save:
        plt.savefig("example_grid.png", dpi=300)

    plt.show()


def get_device() -> str:
    """
    Get the available device (CPU or GPU) and print GPU information if available.

    Returns:
    str: The device string ("cuda" or "cpu").
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    return device
