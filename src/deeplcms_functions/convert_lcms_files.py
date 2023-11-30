import gc
import io
import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyopenms as oms
from PIL import Image, ImageChops
from scipy import ndimage
from sklearn import model_selection
from tqdm import tqdm

from deeplcms_functions import utils


def plot_2D_spectra_overview(
    file: Path, save: bool = True, dpi: int = 300, nx: int = 500, ny: int = 500
) -> None:
    """
    Plot a 2D overview of mass spectrometry spectra.

    This function reads mass spectrometry data from the provided file and generates
    a 2D heatmap. The x-axis represents the retention time (RT), and the y-axis
    represents the m/z values.

    Args:
        file (Path): Path to the mass spectrometry data file.
        save (bool, optional): Whether to save the generated plot. Defaults to True.
        dpi (int, optional): Dots per inch for the saved image. Defaults to 300.
        nx (int, optional): Number of bins along the x-axis (RT). Defaults to 500.
        ny (int, optional): Number of bins along the y-axis (m/z). Defaults to 500.

    Returns:
        Optional[Image.Image]: If 'save' is False, returns a PIL Image of the generated plot. Otherwise, returns None.

    Example:
        To plot a 2D overview of mass spectrometry data:
        ```python
        plot_2D_spectra_overview(Path("example_data.mzML"), save=True)
        ```
        This generates a 2D heatmap, saving it as a JPEG file with default settings.
    """
    # temporarily set the working directory to the folder where file is
    original_cwd = Path.cwd()
    os.chdir(file.parent)

    try:
        # Load mass spectrometry data from mzML file
        exp = oms.MSExperiment()
        loader = oms.MzMLFile()
        loadopts = loader.getOptions()
        loadopts.setMSLevels([1])
        loadopts.setSkipXMLChecks(True)
        loadopts.setIntensity32Bit(True)
        loadopts.setIntensityRange(
            oms.DRange1(oms.DPosition1(5000), oms.DPosition1(sys.maxsize))
        )
        loader.setOptions(loadopts)
        loader.load(file.name, exp)
        exp.updateRanges()

        # Create a DataFrame from the 2D peak data
        spectraarrs2d = exp.get2DPeakDataLong(
            exp.getMinRT(), exp.getMaxRT(), exp.getMinMZ(), exp.getMaxMZ()
        )
        spectradf = pd.DataFrame(
            {"RT": spectraarrs2d[0], "mz": spectraarrs2d[1], "inty": spectraarrs2d[2]}
        )

        # Create a 2D histogram of the data
        hist, xedges, yedges = np.histogram2d(
            spectradf["RT"],
            spectradf["mz"],
            bins=[nx, ny],  # Adjust the number of bins as needed
            weights=spectradf["inty"],
        )
        # Normalize intensity values using the highest peak
        hist = hist / np.max(hist)

        # Apply Gaussian filter for noise reduction
        filter_size = 2  # Set the size of the Gaussian filter
        hist = ndimage.gaussian_filter(hist, sigma=filter_size)

        # Scale the filtered image to the range 0-255
        hist *= 255.0 / hist.max()

        # Take the logarithm of the histogram to better visualize large ranges of data
        hist = np.log1p(hist.T)  # Transpose the histogram for correct orientation

        # Create the colormap
        cmap = plt.get_cmap("jet")  # Use the 'jet' colormap or choose another

        # Create the plot
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(
            hist, interpolation="nearest", cmap=cmap, origin="lower", aspect="auto"
        )
        ax.set_xlabel("RT")
        ax.set_ylabel("mz")
        ax.set_title("")
        ax.axis("off")
        plt.tight_layout(pad=0)

        # Convert the plot to a PIL Image object
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        PIL_image = Image.open(buf).convert("RGB")

        if save:
            fig.savefig(
                f"{file.stem}.jpeg",
                dpi=dpi,
                pad_inches=0,
                bbox_inches="tight",
                transparent=True,
            )
        return PIL_image
    finally:
        try:
            # Close the figure to release resources
            plt.close()

            # Reset the DataFrame to release its memory
            with warnings.catch_warnings():  # Suppress the Pandas warning about resetting the frame
                warnings.simplefilter("ignore", category=UserWarning)
                spectradf = pd.DataFrame()

            # Force garbage collection to release memory
            gc.collect()

        except UnboundLocalError:
            pass  # Ignore the error if spectradf is not defined

        os.chdir(original_cwd)


def plot_2D_spectra_slices(
    file: Path,
    num_slices: int = 10,
    save: bool = True,
    dpi: int = 300,
    show: bool = False,
) -> None:
    """
    Plot 2D slices of mass spectrometry data.

    This function generates and saves 2D slices of mass spectrometry data from an mzML file.
    Each slice corresponds to a specific mass range. The slices are visualized using a 2D histogram.

    Args:
        file (Path): The path to the mzML file.
        num_slices (int, optional): The number of slices to generate. Defaults to 10.
        save (bool, optional): Whether to save the generated plots. Defaults to True.
        show (bool, optional): Whether to display the generated plots. Defaults to False.
        dpi (int, optional): Dots per inch for the saved plots. Defaults to 300.

    Returns:
        None

    Note:
        - The generated plots are saved in the same directory as the input mzML file.
        - The mass range is divided into equal slices, and each slice corresponds to a specific mass range.

    Example:
        To plot 2D slices of mass spectrometry data and save the plots without displaying them:
        >>> plot_2D_spectra_slices(Path("example.mzML"), num_slices=5, save=True, show=False)

    Raises:
        AssertionError: If the number of slices is less than 2. The function requires at least two slices
        to create meaningful visualizations.

    """

    # temporarily set the working directory to the folder where file is
    original_cwd = Path.cwd()
    os.chdir(file.parent)

    # Load mass spectrometry data from mzML file
    exp = oms.MSExperiment()
    loader = oms.MzMLFile()
    loadopts = loader.getOptions()
    loadopts.setMSLevels([1])
    loadopts.setSkipXMLChecks(True)
    loadopts.setIntensity32Bit(True)
    loadopts.setIntensityRange(
        oms.DRange1(oms.DPosition1(5000), oms.DPosition1(sys.maxsize))
    )
    loader.setOptions(loadopts)
    loader.load(file.name, exp)
    exp.updateRanges()

    initial_mass = exp.getMinMZ()
    final_mass = exp.getMaxMZ()
    num_slices = num_slices
    mass_increment = (final_mass - initial_mass) / (num_slices + 1)
    temp_final_mass = initial_mass + mass_increment

    assert num_slices >= 2, (
        f"num_slices must be 2 or greater, got {num_slices}. "
        "For visualizing the whole mass range use the plot_2D_spectra_overview function instead."
    )
    for num in range(num_slices):
        # Create a DataFrame from the 2D peak data
        spectraarrs2d = exp.get2DPeakDataLong(
            exp.getMinRT(), exp.getMaxRT(), initial_mass, temp_final_mass
        )
        spectradf = pd.DataFrame(
            {
                "RT": spectraarrs2d[0],
                "mz": spectraarrs2d[1],
                "inty": spectraarrs2d[2],
            }
        )

        # Create a 2D histogram of the data
        hist, xedges, yedges = np.histogram2d(
            spectradf["RT"],
            spectradf["mz"],
            bins=[500, 500],  # Adjust the number of bins as needed
            weights=spectradf["inty"],
        )
        # Normalize intensity values using the highest peak
        hist = hist / np.max(hist)

        # Apply Gaussian filter for noise reduction
        filter_size = 2  # Set the size of the Gaussian filter
        hist = ndimage.gaussian_filter(hist, sigma=filter_size)

        # Scale the filtered image to the range 0-255
        hist *= 255.0 / hist.max()

        # Take the logarithm of the histogram to better visualize large ranges of data
        hist = np.log1p(hist.T)  # Transpose the histogram for correct orientation

        # Create the colormap
        cmap = plt.get_cmap("jet")  # Use the 'jet' colormap or choose another

        # Create the plot
        plt.figure(figsize=(4, 4))
        plt.imshow(hist, interpolation="nearest", cmap=cmap, origin="lower")
        plt.xlabel("RT")
        plt.ylabel("mz")
        plt.title("")
        plt.axis("off")
        plt.tight_layout()

        if save:
            plt.savefig(f"{file.stem}_slice{num + 1}.jpeg", dpi=dpi)
        if show:
            plt.show()
        else:
            plt.close()

        initial_mass += mass_increment
        temp_final_mass += mass_increment

    os.chdir(original_cwd)


def create_train_val_test_directories(
    study_name: str, path: Union[str, Path], group_1: str, group_2: str
) -> None:
    """
    Create directory structure for a study with train and val folders for two groups.

    Args:
        study_name (str): The name of the study.
        path (str or Path): The base path where the study folders will be created.
        group_1 (str): The name of the first group (e.g., control group).
        group_2 (str): The name of the second group (e.g., experimental group).

    Returns:
        None

    Example:
        To create a directory structure for a study named "example_study" with train and val folders
        for a control group ("control_group") and an experimental group ("experimental_group"):
        >>> create_train_test_directories("example_study", "/path/to/studies", "control_group", "experimental_group")
    """
    assert (
        group_1 != group_2
    ), "Group names must be unique and different for proper folder creation."
    if isinstance(path, str):
        path = Path(path)

    study_path = path.joinpath(study_name)
    parent_folders = ["train", "val", "test"]
    daughter_folders = [group_1, group_2]

    for parent in parent_folders:
        for daughter in daughter_folders:
            appended_path = study_path.joinpath(parent).joinpath(daughter)
            print(appended_path)
            appended_path.mkdir(parents=True, exist_ok=False)


def get_train_val_test_split(
    path: Union[str, Path],
    test_portion: float,
    val_portion: float,
    random_state: int = utils.Configuration.seed,
) -> pd.DataFrame:
    """
    Split the data into training, validation, and test sets.

    Args:
        path (str or Path): The path to the dataset list containing what sample belongs to what group in Parquet format.
        test_portion (float): The proportion of the data to include in the test split.
        val_portion (float): The proportion of the remaining data to include in the validation split.
        random_state (int, optional): Seed for the random number generator. Defaults to utils.Configuration.seed.

    Returns:
        pd.DataFrame: A DataFrame with a 'split' column indicating whether each sample is in the training, validation, or test set.

    Example:
        To split a dataset at "/path/to/dataset.parquet" into training, validation, and test sets:
        >>> split_df = get_train_val_test_split("/path/to/dataset.parquet", 0.2, 0.1)
    """
    if isinstance(path, str):
        path = Path(path)

    file = pd.read_parquet(path)

    # Check that the sum of test and validation portions does not exceed 1
    assert (
        test_portion + val_portion <= 1
    ), "Sum of test_portion and val_portion cannot exceed 1"

    # First split to get train and test set
    remaining, test = model_selection.train_test_split(
        file,
        test_size=test_portion,
        stratify=file.phenotype,
        random_state=random_state,
    )

    # Second split to get train and val set
    train, val = model_selection.train_test_split(
        remaining,
        test_size=val_portion,
        stratify=remaining.phenotype,
        random_state=random_state,
    )

    # Create a DataFrame with a 'split' column
    train_test_val_split_df = file.assign(
        split=lambda df: np.select(
            condlist=[
                df.sample_name.isin(train.sample_name),
                df.sample_name.isin(val.sample_name),
            ],
            choicelist=["train", "val"],
            default="test",
        )
    )

    return train_test_val_split_df


def copy_LCMS_files_SUPERSEEDED(
    df: pd.DataFrame,
    source_folder: Path,
    destination_folder: Path,
    target_col: str,
) -> None:
    """
    Copy LCMS files based on conditions specified in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing information about samples.
        source_folder (Path): The folder containing the source LCMS files.
        destination_folder (Path): The base destination folder.
        target_col (str): The column in the DataFrame specifying the target treatment.

    Returns:
        None

    Example:
        To copy LCMS files based on the 'split' and 'treatment' columns in the DataFrame:
        >>> df = pd.DataFrame({
        ...     'sample_name': ['sample1', 'sample2', 'sample3'],
        ...     'split': ['train', 'val', 'train'],
        ...     'treatment': ['control', 'experimental', 'control']
        ... })
        >>> source_folder = Path("/path/to/source")
        >>> destination_folder = Path("/path/to/destination")
        >>> target_col = 'treatment'
        >>> copy_LCMS_files(df, source_folder, destination_folder, target_col)
    """
    # Get a list of all mzML files in the source folder
    all_mzML_files = list(source_folder.glob("*.mzML"))

    # Iterate over unique values in the 'split' column
    for group_split in tqdm(df.split.unique()):
        # Iterate over unique values in the 'treatment' column
        for group_treatment in df.treatment.unique():
            # Filter the DataFrame based on split and treatment conditions
            filtered_sample_list = df.query(
                "split == @group_split and treatment == @group_treatment"
            ).sample_name.to_list()

            # Iterate over all mzML files
            for source_file in all_mzML_files:
                # Check if the stem of the source file is in the filtered sample list
                if source_file.stem in filtered_sample_list:
                    # Copy the source file to the destination folder
                    shutil.copy(
                        source_file,
                        destination_folder.joinpath(group_split).joinpath(
                            group_treatment
                        ),
                    )


def convert_LCMS_files_and_move_images(
    source_folder: Path,
    df: pd.DataFrame,
    destination_folder: Path,
    target_col: str = "treatment",
) -> None:
    """
    Convert LCMS files to JPEG format based on conditions specified in the DataFrame.

    Args:
        source_folder (Path): The folder containing the source LCMS files in mzML format.
        df (pd.DataFrame): The DataFrame containing information about samples.
        destination_folder (Path): The base destination folder for the converted JPEG files.
        target_col (str, optional): The column in the DataFrame specifying the target treatment. Defaults to "treatment".

    Returns:
        None

    Example:
        To convert LCMS files to JPEG format based on the 'split' and 'treatment' columns in the DataFrame:
        >>> df = pd.DataFrame({
        ...     'sample_name': ['sample1', 'sample2', 'sample3'],
        ...     'split': ['train', 'val', 'train'],
        ...     'treatment': ['control', 'experimental', 'control']
        ... })
        >>> source_folder = Path("/path/to/source")
        >>> destination_folder = Path("/path/to/destination")
        >>> target_col = 'treatment'
        >>> convert_files(source_folder, df, destination_folder, target_col)
    """
    # Convert mzML files to JPEG format
    for file_ in tqdm(list(source_folder.glob("*.mzML"))):
        plot_2D_spectra_overview(file_, save=True, show=False)

        del file_
        gc.collect()

    # Get a list of all JPEG files in the source folder
    all_jpeg_files = list(source_folder.glob("*.jpeg"))

    # Iterate over unique values in the 'split' column
    for group_split in tqdm(df.split.unique()):
        # Iterate over unique values in the 'treatment' column
        for group_treatment in df.phenotype.unique():
            # Filter the DataFrame based on split and treatment conditions
            filtered_sample_list = df.query(
                "split == @group_split and phenotype == @group_treatment"
            ).sample_name.to_list()

            # Iterate over all JPEG files
            for source_file in all_jpeg_files:
                # Check if the stem of the source file is in the filtered sample list
                if source_file.stem in filtered_sample_list:
                    new_file_path = (
                        destination_folder.joinpath(group_split)
                        .joinpath(group_treatment)
                        .joinpath(source_file.name)
                    )
                    source_file.rename(new_file_path)


def augment_images(
    location: str,
    xoffs: int = 5,
    yoffs: int = 5,
    n: int = 10,
    save: bool = False,
    seed: int = utils.Configuration.seed,
) -> None:
    """
    Apply random offsets to an image.

    Parameters:
    - location (str): Path to the input image.
    - xoffs (int): Maximum horizontal offset (positive or negative).
    - yoffs (int): Maximum vertical offset (positive or negative).
    - n (int): Number of augmented images to generate.
    - save (bool): If True, save augmented images.
    - seed (int): Seed for random number generation.

    Returns:
    - None

    Usage Example:
    ```
    augment_images('path/to/your/image.jpg', xoffs=3, yoffs=6, n=5, save=True)
    ```
    This will apply random offsets to the specified image, generating 5 augmented
    images with a maximum horizontal offset of 3 and a maximum vertical offset of 6.
    The augmented images will be saved if `save` is set to True.
    """
    # Validate inputs
    if (
        not isinstance(xoffs, int)
        or not isinstance(yoffs, int)
        or not isinstance(n, int)
        or n <= 0
    ):
        raise ValueError(
            "Invalid input for xoffs, yoffs, or n. They should be positive integers."
        )
    np.random.seed(seed)
    # Convert location to Path object
    location = Path(location)

    with Image.open(location) as img:
        # Open the image
        # img = Image.open(location)
        width, height = img.size

        # Generate random offsets
        x_range_random = np.random.randint(-xoffs, xoffs, size=n, dtype=int)
        y_range_random = np.random.randint(-yoffs, yoffs, size=n, dtype=int)

        for x_offset, y_offset in zip(x_range_random, y_range_random):
            offset_image = ImageChops.offset(
                img, xoffset=int(x_offset), yoffset=int(y_offset)
            )
            # pad the image with blue background
            offset_image.paste((0, 1, 127), (0, 0, int(x_offset), height))
            offset_image.paste((0, 1, 127), (0, 0, width, int(y_offset)))

            if save:
                # Generate the save path
                save_path = (
                    location.parent
                    / f"{location.stem}_xoffset{x_offset}_yoffset{y_offset}.jpeg"
                )

                print(f"Augmented image saved: {save_path}")
                offset_image.save(save_path)

        return None
