import os
import random
from pathlib import Path
from typing import List, Optional, Tuple, Union

import colab_utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.utils.data
import torchmetrics
import torchvision
from PIL import Image
from torchcam.methods import LayerCAM
from torchcam.utils import overlay_mask
from torchvision.io.image import read_image
from torchvision.transforms.functional import to_pil_image


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
    df: pd.DataFrame, save: Optional[bool] = True, bbox_to_anchor=(1.1, 0.97)
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
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=bbox_to_anchor)
    plt.tight_layout()

    if save:
        fig.savefig("experiment_result.png", dpi=300, bbox_inches="tight")

    plt.close(fig)  # Close the figure to prevent double display

    return fig


def inspect_predictions(
    logits: list, test_dataloader: torch.utils.data.DataLoader, save: bool = True
) -> None:
    """
    Inspect model predictions by displaying a 3x3 grid of images with predicted labels.

    Parameters:
    - logits (list): List of tensors containing model logits for each sample.
    - test_dataloader (torch.utils.data.DataLoader): DataLoader for the test dataset.
    - save (bool, optional): If True, save the prediction matrix plot. Defaults to True.

    Returns:
    - None

    Example:
    ```python
    # Assuming you have a test_dataloader and logits_list defined
    inspect_predictions(logits_list, test_dataloader, save=True)
    ```
    The title color in each subplot indicates the accuracy of the prediction:
    - Green: Prediction matches the actual label.
    - Red: Prediction does not match the actual label.
    """

    images, labels = next(iter(test_dataloader))

    first_batch_of_logits = logits[: test_dataloader.batch_size]
    probabilities = torch.sigmoid(torch.cat(first_batch_of_logits, dim=0)).squeeze()

    threshold = 0.5
    predicted_labels = (probabilities > threshold).float().view(-1)

    # Create a 3x3 grid for displaying images
    fig = plt.figure(figsize=(12, 12))
    rows, cols = 3, 3

    for i in range(1, rows * cols + 1):
        fig.add_subplot(rows, cols, i)

        # Choose a random index
        random_index = np.random.randint(0, probabilities.shape[0] - 1)
        title = f"Predicted: {probabilities[random_index].item():.2%} proba of class 1 | True: {int(labels[random_index])}"
        color = (
            "g"
            if round(probabilities[random_index].item()) == int(labels[random_index])
            else "r"
        )
        plt.title(title, fontsize=8, color=color, weight="bold")
        plt.imshow(images[random_index].permute(1, 2, 0))

        plt.axis(False)

    if save:
        plt.savefig("prediction_matrix.png", bbox_inches="tight", dpi=300)


def evaluate_predictions(
    logits: list,
    test_dataloader: torch.utils.data.DataLoader,
    binary: bool = True,
    save: bool = True,
) -> None:
    """
    Evaluate predictions using binary or multiclass classification metrics.

    Parameters:
    - logits (list): List of tensors containing model logits for each sample.
    - test_dataloader (torch.utils.data.DataLoader): DataLoader for the test dataset.
    - binary (bool, optional): If True, perform binary classification. Defaults to True.
    - save (bool, optional): If True, save the BinaryConfusionMatrix plot. Defaults to True.

    Returns:
    - None

    Example:
    ```python
    # Assuming you have a test_dataloader and logits_list defined
    evaluate_predictions(logits_list, test_dataloader, binary=True, save=True)
    ```
    """

    true_labels = torch.tensor(test_dataloader.dataset.targets)
    if binary:
        probabilities = torch.sigmoid((torch.cat(logits, dim=0)))
        # Threshold probabilities to get binary predictions (0 or 1)
        threshold = 0.5
        predicted_labels = (probabilities > threshold).float().view(-1)
    else:
        # in case of multiclass problems
        probabilities = torch.softmax(logits, dim=1)
        predicted_labels = probabilities.argmax(dim=1)

    acc = (true_labels == predicted_labels).sum().item() / len(true_labels)

    metric_f1 = torchmetrics.classification.BinaryF1Score()
    f1 = metric_f1(true_labels, predicted_labels)

    metric_precision = torchmetrics.classification.BinaryPrecision()
    precision = metric_precision(true_labels, predicted_labels)

    metric_recall = torchmetrics.classification.BinaryRecall()
    recall = metric_recall(true_labels, predicted_labels)

    bcm = torchmetrics.classification.BinaryConfusionMatrix()
    bcm(true_labels, predicted_labels)

    print(
        f"Accuracy: {acc:.2f} | F1: {f1:.2f} | Precision: {precision:.2f} | Recall: {recall:.2f}"
    )
    fig_, ax_ = bcm.plot(add_text=True)

    if save:
        plt.savefig("BinaryConfusionMatrix.png", bbox_inches="tight", dpi=300)


def plot_activation(
    dataloader: torch.utils.data.DataLoader,
    device: str,
    model: torch.nn.Module,
    save: bool = True,
) -> None:
    """
    Generate a 3x3 grid of images with Class Activation Maps (CAM) and optionally save the plot.

    Parameters:
    - dataloader (Any): The DataLoader containing the images and labels.
    - device (Any): The device on which the model should run (e.g., 'cuda' or 'cpu').
    - model (Any): The neural network model.
    - save (bool, optional): Whether to save the plot as 'plot_activation.png'. Default is True.

    Example:
    ```python
    from torchvision import models, transforms
    from torch.utils.data import DataLoader
    from your_dataset_module import YourDataset  # Replace 'your_dataset_module' with the actual module name

    # Assuming you have a DataLoader named 'your_dataloader' and a device 'cuda'
    your_dataloader = DataLoader(YourDataset(...), batch_size=32, shuffle=True)
    your_model = models.resnet50(pretrained=True)
    your_model.to('cuda')

    plot_activation(your_dataloader, 'cuda', your_model, save=True)
    ```
    """
    for param in model.parameters():
        param.requires_grad = True

    images, labels = next(iter(dataloader))
    images, labels = images.to(device), labels.to(device)

    # Create a 3x3 grid for displaying images
    fig = plt.figure(figsize=(12, 12))
    rows, cols = 3, 3

    for i in range(1, rows * cols + 1):
        fig.add_subplot(rows, cols, i)

        # Choose a random index
        random_index = np.random.randint(0, len(dataloader.dataset) - 1)

        # Retrieve the CAM from several layers at the same time
        cam_extractor = LayerCAM(
            model, ["model.layer2", "model.layer3", "model.layer4"]
        )

        # Preprocess your data and feed it to the model
        out = model(images[random_index].unsqueeze(0))
        # Retrieve the CAM by passing the class index and the model output
        cams = cam_extractor(out.squeeze(0).argmax().item(), out)

        result = overlay_mask(
            to_pil_image(images[random_index]), to_pil_image(cams, mode="F"), alpha=0.5
        )
        plt.imshow(result)
        plt.title(f"Class: {labels[random_index]}")
        plt.axis(False)

        cam_extractor.remove_hooks()

    if save:
        plt.savefig("plot_activation.png", bbox_inches="tight", dpi=300)

    plt.show()  # Display the plot
