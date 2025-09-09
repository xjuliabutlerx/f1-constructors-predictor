from rich import print

import os

V1 = "v1"
V2 = "v2"

CHECKPOINTS_PATH = "checkpoints"          # Path to saved checkpoints
HEATMAPS_PATH = "heatmaps"                # Path to the heatmap visuals
MODELS_PATH = "pretrained_models"         # Path to the saved models
TRAINING_DATA_PATH = "training_data"      # Path to the training data excels

if __name__ == "__main__":
    print()
    base_dirs = [V1, V2]
    sub_dirs = [CHECKPOINTS_PATH, HEATMAPS_PATH, MODELS_PATH, TRAINING_DATA_PATH]

    for base in base_dirs:
        for sub in sub_dirs:
            new_dir_path = os.path.join(base, sub)
            if not os.path.exists(new_dir_path):
                print(f"[cyan]{new_dir_path}[/cyan] does not exist, creating now...", end="")
                os.mkdir(new_dir_path)
                print("[green]done[/green]!")
            else:
                print(f"[cyan]{new_dir_path}[/cyan] already exists!")
        
    print()