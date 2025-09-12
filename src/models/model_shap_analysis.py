from predict_ensemble import get_device
from rich import print

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shap
import torch

def callable_model(model, x:np.ndarray):
    X_values = torch.tensor(x, dtype=torch.float32).to(device)
    with torch.no_grad():
        scores = model(X_values).detach().cpu().numpy().flatten()
        order = np.argsort(-scores)                # descending order
        ranks = np.empty_like(order, dtype=int)
        ranks[order] = np.arange(1, len(scores) + 1)
        return ranks

def run_shap_analysis(model_name:str, model_path:str, dataset_df:pd.DataFrame, feature_cols:list, device:torch.device, sample_size:int=50, analysis_dir:str="shap_analysis"):
    X_values = torch.tensor(dataset_df[feature_cols].values, dtype=torch.float32).to(device)
    X_numpy = dataset_df[feature_cols].values.astype(np.float32)

    model = F1ConstructorsClassifier(input_dim=X_values.shape[1], output_dim=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    shap_explainer = shap.Explainer(lambda x: callable_model(model, x), X_numpy[:sample_size])
    shap_values = shap_explainer(X_numpy)
    shap_values = shap_values[0] if isinstance(shap_values, list) else shap_values

    rng = np.random.default_rng(42)  # fixed seed for reproducibility

    plt.title(f"SHAP Summary for {model_name}")
    shap.summary_plot(shap_values, X_values, feature_names=feature_cols, show=False, max_display=len(feature_cols), rng=rng)
    plt.savefig(os.path.join(analysis_dir, f"shap_summary_{model_name}.png"), bbox_inches="tight")
    plt.close()

    plt.title(f"SHAP Bar Plot for {model_name}")
    shap.summary_plot(shap_values, X_values, feature_names=feature_cols, plot_type="bar", show=False, max_display=len(feature_cols), rng=rng)
    plt.savefig(os.path.join(analysis_dir, f"shap_bar_{model_name}.png"), bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--models_dir_path", "-m", type=str, required=True, default=None)
    PARSER.add_argument("--training_data_path", "-d", type=str, required=False, default=None)
    PARSER.add_argument("--version", "-v", type=int, required=False, default=1)
    PARSER.add_argument("--sample_size", "-s", type=int, default=50)
    PARSER.add_argument("--analysis_path", "-a", type=str, default="shap_analysis", help="Directory to save SHAP analysis results")

    ARGS = PARSER.parse_args()

    print()
    print("[yellow]*** F1 CONSTRUCTOR CLASSIFIER MODEL SHAP ANALYSIS ***[/yellow]")
    print()

    models_dir_path = ARGS.models_dir_path
    training_data_path = ARGS.training_data_path if ARGS.training_data_path is not None else os.path.join("../../data/clean/", "f1_clean_data.csv")
    version = ARGS.version
    sample_size = ARGS.sample_size
    analysis_path = ARGS.analysis_path

    if models_dir_path is None or not os.path.isdir(models_dir_path):
        print(f"[red]ERROR[/red]: You must provide a valid directory path for the pretrained models.\n")
        exit(0)
    elif len(os.listdir(models_dir_path)) == 0:
        print(f"[red]ERROR[/red]: No files found in the directory {models_dir_path}\n")
        exit(0)
    
    model_files_list = os.listdir(models_dir_path)

    for file in model_files_list:
        if not file.endswith(".pt"):
            print(f"[red]ERROR[/red]: The file {file} is not a valid pretrained model file.\n")
            exit(0)
    
    if training_data_path is None or not os.path.exists(training_data_path):
        print(f"[red]ERROR[/red]: You must provide a valid path for the prediction data.\n")
        exit(0)

    if version not in [1, 2]:
        print(f"[red]ERROR[/red]: Invalid model version {version}.\n")
        exit(0)

    print(f"Parameters:")
    print(f" > Model State Path: {models_dir_path}")
    for model_file in model_files_list:
        print(f"   - {model_file}")
    print(f" > Prediction Data Path: {training_data_path}")
    print(f" > Evaluation Device: ", end="")
    device = get_device()
    print()

    print("Data:")
    if version == 1:
        print(f" > Loading v1 F1 Dataset...", end="")
        from v1.f1_dataset import F1Dataset
        print("[green]done[/green]")

        print(f" > Loading v1 F1 Constructors Classifier model...", end="")
        from v1.f1_constructors_rank_classifier import F1ConstructorsClassifier
        print("[green]done[/green]")
    elif version == 2:
        print(f" > Loading v2 F1 Dataset...", end="")
        from v2.f1_dataset import F1Dataset
        print("[green]done[/green]")

        print(f" > Loading v2 F1 Constructors Classifier model...", end="")
        from v2.f1_constructors_rank_classifier import F1ConstructorsClassifier
        print("[green]done[/green]")
    
    print(f" > Loading dataset...", end="")
    dataset = F1Dataset(training_data_path)
    print(f"[green]done[/green]")

    print(f" > Retrieving feature column names...", end="")
    feature_cols = dataset.get_feature_columns()
    print(f"[green]done[/green]")
    print()

    # Make a directory to save the model analysis diagrams if it doesn't already exist
    if not os.path.exists(analysis_path):
        os.mkdir(analysis_path)

    analysis_sub_folder = os.path.join(analysis_path), f"v{version}"
    if not os.path.exists(analysis_sub_folder):
        os.mkdir(analysis_sub_folder)

    print("Analysis:")
    for model_file in model_files_list:
        model_name = model_file[:-3].replace("_", " ")
        model_name = model_name.title()
        model_name = model_name.replace(f"V{version}", f"v{version}")       # make the "v" for version lowercase

        print(f" > Analyzing [magenta]{model_name}[/magenta]...", end="")
        run_shap_analysis(model_name, os.path.join(models_dir_path, model_file), dataset.df, feature_cols, device, analysis_dir=analysis_sub_folder)
        print(f"[green]done[/green]")