from rich import print

import os

DATA_PATH = "../../data"                            # Path to root data directory
CACHE_PATH = "../../data/cache"                     # Path to the cache in the data directory
RAW_DATA_PATH = "../../data/raw"                    # Path to the raw data
PREPROCESSED_DATA_PATH = "../../data/preprocessed"  # Path to the preprocessed data
CLEAN_DATA_PATH = "../../data/clean"                # Path to the clean training data

if __name__ == "__main__":
    print()
    directories = [DATA_PATH, CACHE_PATH, RAW_DATA_PATH, PREPROCESSED_DATA_PATH, CLEAN_DATA_PATH]

    for directory in directories:
        if not os.path.exists(directory):
            print(f"[cyan]{directory}[/cyan] does not exist, creating now...", end="")
            os.mkdir(directory)
            print("[green]done[/green]!")
        else:
            print(f"[cyan]{directory}[/cyan] already exists!")