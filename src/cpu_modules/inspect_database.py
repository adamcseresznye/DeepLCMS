from typing import Optional

import pandas as pd


def return_metabolomics_workbench_studies() -> pd.DataFrame:
    """
    Fetches the available metabolomics study dataset from the Metabolomics Workbench website.

    Returns:
        pd.DataFrame: Metabolomics dataset.

    Example:
    >>> metabolomics_data = return_metabolomics_workbench_dataset()
    """
    return pd.read_html(
        "https://www.metabolomicsworkbench.org/data/DRCCStudySummary.php?Mode=StudySummary&SortBy=Study%20ID&AscDesc=desc&ResultsPerPage=2000"
    )[2]


def filter_and_sort_datasets(
    df: pd.DataFrame, min_samples: int = 1, max_samples: Optional[int] = None
) -> pd.DataFrame:
    """
    Filters and sorts a DataFrame based on specific conditions related to samples, analysis, and file size metrics.

    This function performs multiple data processing steps on the input DataFrame:

    1. Column Renaming: Renames a specific column to 'file_size'.
    2. Feature Engineering: Derives new columns from 'file_size' to extract data format, size, and metric.
    3. Data Manipulation: Drops the original 'file_size' column.
    4. Sorting: Sorts the DataFrame based on the 'Samples' column in descending order.
    5. Missing Value Handling: Removes rows with missing values in the 'format' column.
    6. Conditional Filtering: Filters the DataFrame based on specific criteria including 'Samples', 'Analysis', and 'file_size_metric'.
        - If 'max_samples' is None, the function filters for 'Samples' greater than 'min_samples'.
        - Else, it filters for 'Samples' between 'min_samples' and 'max_samples'.

    Parameters:
        df (pd.DataFrame): The DataFrame to be filtered and sorted.
        min_samples (int): Minimum number of samples (default: 1).
        max_samples (int, optional): Maximum number of samples (default: None).

    Returns:
        pd.DataFrame: Filtered and sorted DataFrame.

    Example:
    >>> filtered_data = filter_and_sort_datasets(df, min_samples=50, max_samples=100)
    """
    df = (
        df.rename(columns={"Download(* : Contains raw data)": "file_size"})
        .assign(
            format=lambda df: df.file_size.str.extract(r"Data format:(\w+)"),
            file_size=lambda df: df.file_size.str.extract("(\d+\.*\d+[a-zA-Z]+)"),
            file_size_number=lambda df: df.file_size.str.extract("(\d+\.*\d+)").astype(
                "float"
            ),
            file_size_metric=lambda df: df.file_size.str.extract("([a-zA-Z])"),
        )
        .drop(columns="file_size")
        .sort_values(by="Samples", ascending=False)
        .dropna(subset="format")
    )

    if max_samples is None:
        return df.query(
            "(Samples > @min_samples) and (Analysis == 'LC-MS#') and (file_size_metric != 'T') and (~format.isin(['d', 'wiff']))"
        ).sort_values(by=["file_size_number", "Samples"], ascending=[True, False])
    else:
        return df.query(
            "(@min_samples < Samples < @max_samples) and (Analysis == 'LC-MS#') and (file_size_metric != 'T') and (~format.isin(['d', 'wiff']))"
        ).sort_values(by=["file_size_number", "Samples"], ascending=[True, False])
