from datetime import datetime
from f1_constructors_rank_classifier import F1ConstructorsClassifier
from f1_dataset import F1Dataset
from rich import print
from train import get_device

import argparse
import os
import pandas as pd
import torch

def predict(model_name:str, model_path:str, dataset_df:pd.DataFrame, device:torch.device):
    idx_final_rank = dataset_df.sort_values(["TeamId", "RoundsCompleted"]).groupby(["TeamId"])["RoundsCompleted"].idxmax()
    prediction_df = dataset_df.loc[idx_final_rank].copy()
    prediction_df.sort_values(["TeamId"])

    X_pred = torch.tensor(prediction_df[feature_cols].values, dtype=torch.float32)

    print(f"Model - {model_name}")
    print(f" > Instantiating model and loading state...", end="")
    model = F1ConstructorsClassifier(input_dim=X_pred.shape[1], output_dim=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("[green]done[/green]")

    print(f" > Running prediction...", end="")
    scores = model(X_pred.to(device)).detach().cpu().numpy()
    prediction_df["PredictedFinalRank"] = scores
    prediction_df["PredictedFinalRank"] = prediction_df["PredictedFinalRank"].rank(method="first", ascending=False).astype(int)
    print("[green]done[/green]")

    predicted_team_rankings = prediction_df.sort_values('PredictedFinalRank', ascending=True)['TeamName'].to_list()
    
    print(f" > Prediction: {predicted_team_rankings}")

    print()

    return predicted_team_rankings

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--models_dir_path", "-m", type=str, required=True, default=None)
    PARSER.add_argument("--prediction_data_path", "-d", type=str, required=True, default=None)

    ARGS = PARSER.parse_args()

    print()
    print("[yellow]*** F1 CONSTRUCTOR CLASSIFIER MODEL ENSEMBLE PREDICTOR ***[/yellow]")
    print()

    models_dir_path = ARGS.models_dir_path
    pred_data_path = ARGS.prediction_data_path
    model_files_list = []

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
    
    if pred_data_path is None or not os.path.exists(pred_data_path):
        print(f"[red]ERROR[/red]: You must provide a valid path for the prediction data.\n")
        exit(0)

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M")

    print(f"Parameters:")
    print(f" > Model State Path: {models_dir_path}")
    for model_file in model_files_list:
        print(f"   - {model_file}")
    print(f" > Prediction Data Path: {pred_data_path}")
    print(f" > Evaluation Device: ", end="")
    device = get_device()
    print()

    print("Data:")
    print(f" > Loading prediction dataset...", end="")
    dataset = F1Dataset(os.path.join("../../data/clean/", "f1_clean_prediction_data.csv"))
    print(f"[green]done[/green]")

    print(f" > Retrieving feature column names...", end="")
    feature_cols = dataset.get_feature_columns()
    print(f"[green]done[/green]")

    year = dataset.df["Year"].unique().tolist()
    if len(year) > 1:
        print(f"[red]ERROR[/red]: There is data from more than 1 F1 seasons; found {len(year)}. This data is not suitable for prediction.\n")
        exit(0)
    year = year[0]
    print(f"  - Running F1 constructor's championship prediction for the year {year}")

    teams = dataset.df["TeamName"].unique().tolist()
    print(f"  - Found {len(teams)} number of teams: {', '.join(teams)}")
    print()

    ensemble_results_df = pd.DataFrame()
    ensemble_results_df["Current Rank"] = dataset.df.sort_values("CurrentRank", ascending=True)["TeamName"].unique().tolist()

    orig_pred_df = pd.read_csv(os.path.join("../../data/clean/", "f1_clean_prediction_data.csv"))
    last_round = orig_pred_df["Round"].max()
    last_race_df = orig_pred_df[orig_pred_df["Round"] == last_round]
    last_race_df = last_race_df.sort_values("TotalPoints", ascending=False)["TotalPoints"]
    ensemble_results_df["Current Points"] = last_race_df.to_list()

    for model_file in model_files_list:
        model_name = model_file[:-3].replace("_", " ")
        model_name = model_name.title()

        prediction = predict(model_name, os.path.join(models_dir_path, model_file), dataset.df, device)

        ensemble_results_df[model_name] = prediction

    print(f"Predicted Results for the {year} F1 Constructor's Championship:")
    print(ensemble_results_df.head(len(teams)), end="\n\n")
    
    results_filename = f"{current_datetime}_ensemble_predictions.xlsx"
    ensemble_results_df.to_excel(f"../../data/ensemble-predictions/{results_filename}", index=False)
    print(f"Saved results to {results_filename}.", end="\n\n")