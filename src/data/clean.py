from rich import print

import os
import pandas as pd
import numpy as np

CLEAN_DATA_PATH = "../../data/clean/"
PREPROCESSED_DATA_PATH = "../../data/preprocessed/"
PREPROCESSED_DATA_FILES = os.listdir(PREPROCESSED_DATA_PATH)

if __name__ == "__main__":
    print()