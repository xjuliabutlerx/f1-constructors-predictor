from .f1_constructors_rank_classifier import F1ConstructorsClassifier
from .f1_dataset import F1Dataset
from rich import print

import os
import pandas as pd
import torch

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda", 0)
    else:
        return torch.device("cpu")

def predict(model_path:str, dataset_df:pd.DataFrame, feature_cols, device:torch.device):
    idx_final_rank = dataset_df.sort_values(["TeamName", "RoundsCompleted"]).groupby(["TeamName"])["RoundsCompleted"].idxmax()
    prediction_df = dataset_df.loc[idx_final_rank].copy()
    prediction_df.sort_values(["TeamName"])

    X_pred = torch.tensor(prediction_df[feature_cols].values, dtype=torch.float32)

    print(model_path)
    model = F1ConstructorsClassifier(input_dim=X_pred.shape[1], output_dim=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    scores = model(X_pred.to(device)).detach().cpu().numpy()
    prediction_df["PredictedFinalRank"] = scores
    prediction_df["PredictedFinalRank"] = prediction_df["PredictedFinalRank"].rank(method="first", ascending=False).astype(int)

    predicted_team_rankings = prediction_df.sort_values('PredictedFinalRank', ascending=True)['TeamName'].to_list()

    return predicted_team_rankings

def run_scenario(input_df:pd.DataFrame):
    print(input_df)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    device = get_device()

    input_df.to_csv("temp_dataset.csv", index=False)
    dataset = F1Dataset("temp_dataset.csv")

    v3_models_path = os.path.join(current_dir, "..", "..", "..", "models", "v3", "pretrained_models")
    v3_models = os.listdir(v3_models_path)
    scenario_results_df = pd.DataFrame()

    for model_file in v3_models:
        model_name = model_file[:-3].replace("_", " ")
        model_name = model_name.title()
        model_name = model_name.replace("V3", "v3")       # make the "v" for version lowercase
        prediction = predict(os.path.join(v3_models_path, model_file), dataset.df, dataset.get_feature_columns(), device)
        scenario_results_df[model_name] = prediction

    os.remove("temp_dataset.csv")

    return scenario_results_df