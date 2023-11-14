import glob
import os
import shutil

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import pyopenms
from matplotlib import colors
from matplotlib import pyplot as plt
from pyopenms import *
from scipy import ndimage
from skimage import exposure, filters, io


def plot_spectra_2D_overview(file_location, gamma=0.5, num_colors=256, filter_size=2):
    """
    Generates a 2D overview plot of the mass spectra contained in a mzML file. Applies a logarithmic transformation
    and a median filter to the data, subtracts the background, and saves the resulting image as a JPEG file. The x-axis of the plot
    represents the retention time (RT) in seconds and the y-axis represents the mass-to-charge (m/z) values.

    Parameters:
    - file_location (str): the path to the mzML file to be analyzed.
    - gamma (float): gamma correction parameter for image contrast adjustment. Default is 0.5.
    - num_colors (int): the number of colors used to represent the intensity values. Default is 256.
    - filter_size (float): size of Gaussian filter for noise reduction. Default is 2.

    Returns:
    - None. The function generates a plot and saves it as a JPEG file.

    Note: This function requires the following external libraries: numpy, matplotlib, and scipy.

    Example usage:
    plot_spectra_2D_overview('/content/test_file.mzML')

    """

    exp = MSExperiment()
    MzMLFile().load(file_location, exp)

    rows = 500.0
    cols = 500.0
    exp.updateRanges()

    bilip = BilinearInterpolation()
    tmp = bilip.getData()
    tmp.resize(int(rows), int(cols), float())
    bilip.setData(tmp)
    bilip.setMapping_0(0.0, exp.getMinRT(), rows - 1, exp.getMaxRT())
    bilip.setMapping_1(0.0, exp.getMinMZ(), cols - 1, exp.getMaxMZ())
    for spec in exp:
        if spec.getMSLevel() == 1:
            mzs, ints = spec.get_peaks()
            rt = spec.getRT()
            for i in range(0, len(mzs)):
                bilip.addValue(rt, mzs[i], ints[i])

    data = np.ndarray(shape=(int(cols), int(rows)), dtype=np.float64)
    for i in range(int(rows)):
        for j in range(int(cols)):
            data[i][j] = bilip.getData().getValue(i, j)

    ms_map = np.power(data, gamma)

    # Represent intensity values with specified number of colors
    bins = np.linspace(0, np.max(ms_map), num_colors)
    ms_map = np.digitize(ms_map, bins)

    # Normalize intensity values using highest peak
    ms_map = ms_map / np.max(ms_map)

    # Apply Gaussian filter for noise reduction
    ms_map = ndimage.gaussian_filter(ms_map, sigma=filter_size)
    ms_map *= 255.0 / ms_map.max()

    # Display the normalized map using Matplotlib
    plt.imshow(np.rot90(ms_map), cmap="jet")
    plt.title("")
    plt.axis("off")
    plt.tight_layout()
    basename = os.path.splitext(file_location)[0]

    plt.savefig(f"{basename}.jpeg", bbox_inches="tight", pad_inches=0, dpi=300)
