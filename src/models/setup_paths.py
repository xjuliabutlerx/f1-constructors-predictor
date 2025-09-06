from rich import print

import os

CHECKPOINTS_PATH = "checkpoints"          # Path to saved checkpoints
HEATMAPS_PATH = "heatmaps"                # Path to the heatmap visuals
MODELS_PATH = "torch_models"              # Path to the saved models
TRAINING_DATA_PATH = "training_data"      # Path to the training data excels

if __name__ == "__main__":
    print()
    directories = [CHECKPOINTS_PATH, HEATMAPS_PATH, MODELS_PATH, TRAINING_DATA_PATH]

    for directory in directories:
        if not os.path.exists(directory):
            print(f"[cyan]{directory}[/cyan] does not exist, creating now...", end="")
            os.mkdir(directory)
            print("[green]done[/green]!")
        else:
            print(f"[cyan]{directory}[/cyan] already exists!")
    print()