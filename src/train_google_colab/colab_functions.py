import os
import random
from pathlib import Path
from typing import List, Optional, Tuple, Union

import colab_utils
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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


def get_experiment_results() -> pd.DataFrame:
    """
    Search for CSV files in the "/content/logs" directory and its subdirectories,
    read each CSV file into a Pandas DataFrame, assign an "experiment" column with
    the stem of the parent directory, and concatenate all these DataFrames.

    Returns:
        Optional[pd.DataFrame]: A concatenated DataFrame containing the results
        of the experiments. Returns None if no CSV files are found.

    Example:
    ```python
    import pandas as pd

    # To get experiment results from CSV files in "/content/logs"
    experiment_results = get_experiment_results()

    if experiment_results is not None:
        print("Experiment results obtained successfully.")
        # Further process or analyze the experiment_results DataFrame
    else:
        print("No CSV files found. Unable to retrieve experiment results.")
    ```
    """
    list_of_files = []
    csv_list = list(Path("/content/logs").glob("**/*.csv"))

    if not csv_list:
        print("No CSV files found.")
        return None

    for csv_file in csv_list:
        file_name: str = csv_file.parents[1].name
        print(f"Reading CSV: {csv_file}")

        read_in_file = pd.read_csv(csv_file).assign(experiment=file_name)
        list_of_files.append(read_in_file)

    final_file = pd.concat(list_of_files, ignore_index=True)
    return (
        final_file.melt(id_vars=["epoch", "experiment"])
        .dropna(subset="value")
        .query("variable!='step'")
    )


def plot_experiment_results(
    df: pd.DataFrame, save: Optional[bool] = True
) -> Union[plt.Figure, None]:
    """
    Plot experiment results using line plots for each variable.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing experiment results.
    - save (bool, optional): Whether to save the generated plot. Defaults to True.

    Returns:
    - Union[plt.Figure, None]: The matplotlib Figure object if the plot is created,
      or None if the DataFrame is empty.

    Example:
    ```python
    import pandas as pd
    import matplotlib.pyplot as plt

    # Assuming df is your DataFrame with columns 'variable', 'epoch', 'value', and 'experiment'
    # For example, you can create a DataFrame using:
    # df = pd.DataFrame({'variable': ['A', 'A', 'B', 'B'],
    #                    'epoch': [1, 2, 1, 2],
    #                    'value': [10, 12, 8, 15],
    #                    'experiment': ['Exp1', 'Exp1', 'Exp2', 'Exp2']})

    plot_experiment_results(df, save=True)
    ```
    This will create line plots for each variable, with separate lines for each experiment,
    and save the plot as 'experiment_result.png' in the current working directory.
    """
    if df.empty:
        print("DataFrame is empty. Cannot create the plot.")
        return None

    num_variables = len(df["variable"].unique())
    num_rows = min(2, num_variables)
    num_cols = min(5, (num_variables + 1) // 2)

    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(18, 8))
    axs = axs.flatten()
    unique_variables = sorted(df["variable"].unique().tolist())

    handles, labels = [], []

    for idx, metric in enumerate(unique_variables):
        temp_df = df.query("variable == @metric")

        line_plot = sns.lineplot(
            data=temp_df,
            x="epoch",
            y="value",
            hue="experiment",
            style="experiment",
            markers=True,
            dashes=True,
            ax=axs[idx],
        ).set(title=f'{metric.replace("_", " ").title()}')

        # Collect handles and labels only from the first subplot
        if idx == 0:
            handles, labels = axs[idx].get_legend_handles_labels()

        # Hide the legend in each subplot
        axs[idx].legend().set_visible(False)

    # Create a single legend outside the subplots
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.2, 0.97))
    plt.tight_layout()

    if save:
        fig.savefig("experiment_result.png", dpi=300, bbox_inches="tight")

    plt.close(fig)  # Close the figure to prevent double display

    return fig
