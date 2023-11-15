import os
import sys
from pathlib import Path
from typing import Union

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import pyopenms as oms
from matplotlib import pyplot as plt
from scipy import ndimage
from tqdm import tqdm

from deeplcms_functions import convert_lcms_files, inspect_database, utils


def plot_2D_spectra_overview(file: Path, save: bool = True, dpi: int = 300) -> None:
    """
    Plot a 2D overview of mass spectrometry spectra.

    This function reads mass spectrometry data from the provided file and generates
    a 2D heatmap, where the x-axis represents the retention time (RT) and the y-axis
    represents the m/z values.

    Args:
        file (Path): The path to the mass spectrometry data file.
        save (bool, optional): Whether to save the generated plot. Defaults to True.
        dpi (int, optional): Dots per inch for the saved image. Defaults to 300.

    Returns:
        None

    Note:
        - The function uses a 2D histogram of the data to create the heatmap.
        - Intensity values are normalized and a Gaussian filter is applied for noise reduction.
        - The resulting plot provides an overview of the distribution of mass spectrometry data.

    Example:
        To plot a 2D overview of mass spectrometry data:
        >>> plot_spectra_2D_overview(Path("example_data.mzML"), save=True)

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
        plt.savefig(f"{file.stem}.jpeg", dpi=dpi)

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


def create_train_test_directories(
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
    parent_folders = ["train", "val"]
    daughter_folders = [group_1, group_2]

    for parent in parent_folders:
        for daughter in daughter_folders:
            appended_path = study_path.joinpath(parent).joinpath(daughter)
            print(appended_path)
            appended_path.mkdir(parents=True, exist_ok=True)
