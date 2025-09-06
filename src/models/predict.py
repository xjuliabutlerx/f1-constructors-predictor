from f1_constructors_rank_classifier import F1ConstructorsClassifier
from f1_dataset import F1Dataset
from rich import print
from train import get_device

import argparse
import os
import numpy as np
import pandas as pd
import torch

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--model_state_path", "-m", type=str, required=True, default=None)
    PARSER.add_argument("--prediction_data_path", "-d", type=str, required=True, default=None)

    ARGS = PARSER.parse_args()

    print()
    print("[yellow]*** TRAINING F1 CONSTRUCTOR CLASSIFIER MODEL ***[/yellow]")
    print()

    model_path = ARGS.model_state_path
    pred_data_path = ARGS.prediction_data_path

    if model_path is None or not os.path.exists(model_path):
        print(f"[red]ERROR[/red]: You must provide a valid path for the saved model state.\n")
        exit(0)
    
    if pred_data_path is None or not os.path.exists(pred_data_path):
        print(f"[red]ERROR[/red]: You must provide a valid path for the prediction data.\n")
        exit(0)

    print(f"Parameters:")
    print(f" > Model State Path: {model_path}")
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

    idx_final_rank = dataset.df.sort_values(["TeamId", "RoundsCompleted"]).groupby(["TeamId"])["RoundsCompleted"].idxmax()
    prediction_df = dataset.df.loc[idx_final_rank].copy()
    prediction_df.sort_values(["TeamId"])

    X_pred = torch.tensor(prediction_df[feature_cols].values, dtype=torch.float32)

    print("Model:")
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
    print()

    print("Results:")
    print(f"Current Standings for the {year} F1 Constructor's Championship:")
    current_ranks = prediction_df.sort_values("CurrentRank", ascending=True)["TeamName"].to_list()
    for i, team in enumerate(current_ranks, start=1):
        print(f"{i}. {team}")
    print()

    print(f"Predicted Results for the {year} F1 Constructor's Championship:")
    ranking_results = prediction_df.sort_values("PredictedFinalRank", ascending=True)["TeamName"].to_list()
    for i, team in enumerate(ranking_results, start=1):
        print(f"{i}. {team}")
    print()

    orig_pred_df = pd.read_csv(os.path.join("../../data/clean/", "f1_clean_prediction_data.csv"))
    last_round = orig_pred_df["Round"].max()
    last_race_df = orig_pred_df[orig_pred_df["Round"] == last_round]
    last_race_df = last_race_df.sort_values("TotalPoints", ascending=False)["TotalPoints"]
    current_points = last_race_df.to_list()

    results_data = {
        "Current Ranks": current_ranks,
        "Current Points": current_points,
        "Predicted Ranks": ranking_results
    }

    results_df = pd.DataFrame(results_data)
    print(results_df, end="\n\n")