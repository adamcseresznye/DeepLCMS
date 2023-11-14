from pathlib import Path


class Configuration:
    VER = 1
    RAW_DATA_PATH = Path(__file__).parents[2].joinpath("data/raw")
    INTERIM_DATA_PATH = Path(__file__).parents[2].joinpath("data/interim")
