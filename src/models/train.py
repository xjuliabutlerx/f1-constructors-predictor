from datetime import datetime
from f1_constructors_rank_classifier import F1ConstructorsClassifier
from f1_dataset import F1Dataset
from rich import print
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import mean_absolute_error, median_absolute_error, max_error

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--num_epochs", "-e", type=int, default=50)
PARSER.add_argument("--learning_rate", "-lr", type=float, default=0.001)
PARSER.add_argument("--decay_rate", "-d", type=float, default=0.95)
PARSER.add_argument("--decay_every", "-f", type=int, default=5)
PARSER.add_argument("--test_size", "-s", type=float, default=0.2)
PARSER.add_argument("--margin", "-m", type=float, default=1.0)
PARSER.add_argument("--patience", "-p", type=int, default=15)
PARSER.add_argument("--random_state", "-r", type=int, default=24)
PARSER.add_argument("--checkpoint", "-c", required=False, type=str, default=None)

ARGS = PARSER.parse_args()

def get_device():
    if torch.backends.mps.is_available():
        print(f"GPU detected - Apple [magenta]Metal Performance Shaders[/magenta]")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print(f"GPU detected - NVIDIA [magenta]Compute Unified Device Architecture[/magenta]")
        return torch.device("cuda", 0)
    else:
        print("No GPU detected - defaulting to CPU")
        return torch.device("cpu")

def load_checkpoint(model:F1ConstructorsClassifier, optimizer:optim.Adam, checkpoint_path:str, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    best_epoch = -1
    best_rho = -1.0

    if "model_state_dict" not in checkpoint:
        model.load_state_dict(checkpoint)         # For early training, I didn't save a full checkpoint dictionary
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if "epoch" in checkpoint:
        best_epoch = checkpoint["epoch"] if checkpoint["epoch"] else -1

    if "best_rho" in checkpoint:
        best_rho = checkpoint["best_rho"] if checkpoint["best_rho"] else -1.0

    return model, optimizer, best_epoch, best_rho

def adjust_learning_rate(learning_rate, optimizer:optim.Adam, epoch, decay_rate, decay_every):
    lr = learning_rate * (decay_rate ** (epoch // decay_every))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr

def rank_misalignment_heatmap(final_test_df:pd.DataFrame, all_y_true, all_y_pred, mean_abs_error, med_abs_error, max_error, best_rho):
    team_count = final_test_df.groupby("Year")["TeamId"].nunique().max()
    heatmap = np.zeros((team_count, team_count), dtype=int)

    # Build misalignment matrix
    for t, p in zip(all_y_true, all_y_pred):
        heatmap[t-1, p-1] += 1

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=[f"Pred {i}" for i in range(1, team_count+1)],
                yticklabels=[f"True {i}" for i in range(1, team_count+1)])
    plt.xlabel("Predicted Rank")
    plt.ylabel("True Rank")
    plt.title(f"F1 Constructor Rank Misalignment Heatmap\nBest Rho={best_rho:.4f}, Mean Absolute Error={mean_abs_error:.2f}, Median={med_abs_error:.2f}, Max={max_error}")
    plt.savefig(os.path.join("heatmaps", f"model_heatmap_{datetime.now().strftime('%Y-%m-%d_%H:%M')}.png"))
    plt.show()

if __name__ == "__main__":
    print()

    test_size = ARGS.test_size
    num_epochs = ARGS.num_epochs
    learning_rate = ARGS.learning_rate
    decay = ARGS.decay_rate
    decay_every = ARGS.decay_every
    margin = ARGS.margin
    patience = ARGS.patience
    random_state = ARGS.random_state
    checkpoint_path = ARGS.checkpoint

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M")

    if test_size > 1:
        print(f"[red]ERROR[/red]: Test size cannot be more than 1; received {test_size}")
        exit(0)
    elif test_size <= 0:
        print(f"[red]ERROR[/red]: Test size cannot be less than or equal to 0; received {test_size}")
        exit(0)

    model_param_data = {
        "Test Timestamp": [current_datetime],
        "Number of Epochs": [num_epochs],
        "Training Size": [(1 - test_size) * 100],
        "Test Size": [test_size * 100],
        "Decay Rate": [decay],
        "Decay Every n Epochs": [decay_every],
        "Margin": [margin],
        "Patience": [patience],
        "Random State": [random_state]
    }
    model_training_params_df = pd.DataFrame(model_param_data)

    print("[yellow]*** TRAINING F1 CONSTRUCTOR CLASSIFIER MODEL ***[/yellow]")
    print()

    print("Training Parameters:")
    print(f" > Test Size: {test_size * 100}%")
    print(f" > Number of Epochs: {num_epochs}")
    print(f" > Intial Learning Rate: {learning_rate}")
    print(f" > Decay: {decay}")
    print(f" > Training Device: ", end="")
    device = get_device()
    print(f" > Loss Function: Margin Ranking Loss")
    print(f" > Loss Function Margin: {margin}")
    print(f" > Patience: {patience}")
    print(f" > Random State: {random_state}")
    print(f" > Load from Checkpoint: {True if checkpoint_path is not None else False}")
    print()

    print("Data:")
    print(f" > Loading dataset...", end="")
    dataset = F1Dataset(os.path.join("../../data/clean/", "f1_clean_data.csv"))
    print(f"[green]done[/green]")

    print(f" > Retrieving feature column names...", end="")
    feature_cols = dataset.get_feature_columns()
    print(f"[green]done[/green]")

    print(f" > Splitting dataset into training and testing...", end="")
    X_train, y_train, test_df = dataset.get_random_split(test_size=test_size, random_state=random_state)
    print(f"[green]done[/green]")

    print(f" > Converting the training dataset into tensors...", end="")
    X_train = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train.to_numpy(), dtype=torch.long).to(device)
    print(f"[green]done[/green]")
    print()

    print("Model and Model Parameters:")
    print(f" > Loading model into training device...", end="")
    model = F1ConstructorsClassifier(X_train.shape[1], 1).to(device)
    print(f"[green]done[/green]")

    print(f" > Instantiating optimizer with learning rate...", end="")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print(f"[green]done[/green]")

    print(f" > Instantiating loss function...", end="")
    loss_func = nn.MarginRankingLoss(margin=margin)
    print(f"[green]done[/green]")

    best_rho = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    best_checkpoint_path = ""

    if checkpoint_path is not None:
        print(f" > Loading saved checkpoint...", end="")
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            model, optimizer, best_epoch, best_rho = load_checkpoint(model, optimizer, checkpoint_path, device)
            print(f"[green]done[/green]")
        else:
            print(f"[red]failed[/red] (moving forward with new model and weights)")

    print()
    print("Training model...")
    training_df = pd.DataFrame(columns=["Epoch", "Learning Rate", "Training Loss", "Spearman's Rho", "Kendall's Tau"])

    for epoch in range(num_epochs):
        model.train()
        learning_rate = adjust_learning_rate(learning_rate, optimizer, epoch, decay_rate=decay, decay_every=decay_every)

        scores = model(X_train).to(device)

        years_train = pd.DataFrame(X_train.cpu().numpy()).iloc[:, 0]
        years_train = years_train.astype(int)

        X_i_pairs, X_j_pairs, pairs_result = [], [], []

        for year in np.unique(years_train):
            idx = np.where(years_train == year)[0]
            n_pairs = min(500, len(idx) * (len(idx) - 1))   # Sample at most 500 pairs from each year

            i_idx = np.random.choice(idx, n_pairs)
            j_idx = np.random.choice(idx, n_pairs)

            X_i_pairs.extend(X_train[i_idx])
            X_j_pairs.extend(X_train[j_idx])

            pair_targets = np.where(y_train[i_idx].cpu().numpy() < y_train[j_idx].cpu().numpy(), 1.0, -1.0)
            pairs_result.extend(pair_targets)

        X_i_pairs = torch.stack(X_i_pairs).to(device)
        X_j_pairs = torch.stack(X_j_pairs).to(device)
        pairs_result = torch.tensor(pairs_result, dtype=torch.float32).to(device)

        s_i = model(X_i_pairs)
        s_j = model(X_j_pairs)

        loss = loss_func(s_i, s_j, pairs_result)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            years_test = test_df["Year"].unique().tolist()
            idx_final_rank = test_df.sort_values(["Year", "TeamId", "RoundsCompleted"]).groupby(["Year", "TeamId"])["RoundsCompleted"].idxmax()

            final_test_df = test_df.loc[idx_final_rank].copy()
            final_test_df.sort_values(["Year", "TeamId"])

            X_test = torch.tensor(final_test_df[feature_cols].values, dtype=torch.float32, device=device)
            final_test_df["PredictedFinalRank"] = model(X_test).detach().cpu().numpy()
            final_test_df["PredictedFinalRank"] = final_test_df.groupby("Year")["PredictedFinalRank"].rank(method="first", ascending=False).astype(int)

            all_y_true = []
            all_y_pred = []
            spearman_scores_by_year = {}
            kendall_scores_by_year = {}
            predicted_rankings_by_year = {}

            for year, group in final_test_df.groupby("Year"):
                true_rank = group["FinalRank"].to_numpy()
                pred_rank = group["PredictedFinalRank"].to_numpy()

                # Create a more readable result for the user to see the rankings by team name
                pred_teams = group.sort_values("PredictedFinalRank", ascending=True)["TeamName"].to_list()
                true_teams = group.sort_values("FinalRank", ascending=True)["TeamName"].to_list()
                results = {
                    "Predicted": pred_teams,
                    "Actual": true_teams
                }
                predicted_rankings_by_year[year] = results

                all_y_true.extend(true_rank)
                all_y_pred.extend(pred_rank)

                rho, _ = spearmanr(true_rank, pred_rank)
                tau, _ = kendalltau(true_rank, pred_rank)

                spearman_scores_by_year[year] = float(rho) if not np.isnan(rho) else None
                kendall_scores_by_year[year] = float(tau) if not np.isnan(tau) else None

            valid_rhos = [r for r in spearman_scores_by_year.values() if r is not None]
            valid_taus = [t for t in kendall_scores_by_year.values() if t is not None]

            avg_rho = float(np.mean(valid_rhos)) if valid_rhos else float("nan")
            avg_tau = float(np.mean(valid_taus)) if valid_taus else float("nan")

        # Early stopping & checkpoint
        improved = avg_rho > best_rho + 1e-4  # tiny tolerance
        if improved:
            best_rho = avg_rho
            best_epoch = epoch + 1
            epochs_no_improve = 0
            best_checkpoint_path = f"{current_datetime}_checkpoint_epoch_{best_epoch}_rho_{best_rho:.4f}.pth"
            torch.save({"epoch": best_epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "best_rho": best_rho}, \
                       os.path.join("checkpoints", best_checkpoint_path))
        else:
            epochs_no_improve += 1

        print(f" > Epoch {epoch+1}/{num_epochs} | Learning Rate: {learning_rate:.6f} | Training Loss: {loss.item():.4f} | Average Spearman's rho: {avg_rho:.4f} | Average Kendall's tau: {avg_tau:.4f}")
        print(f"   Spearman's rho by Year: \n{json.dumps(spearman_scores_by_year, indent=4)}")
        print(f"   Kendall's tau by Year: \n{json.dumps(kendall_scores_by_year, indent=4)}")
        print(f"   Predicted vs. Actual Rankings by Year: \n{json.dumps(predicted_rankings_by_year, indent=4)}")
        training_df.loc[len(training_df)] = [epoch + 1, learning_rate, loss.item(), avg_rho, avg_tau]

        if epochs_no_improve >= patience:
            print()
            print(f"[yellow]Early Stopping[/yellow]: no improvement for {patience} epochs. Best rho={best_rho:.4f} @ epoch {best_epoch}.")
            break

    print()

    print("Evaluating model...")
    mean_abs_error = mean_absolute_error(all_y_true, all_y_pred)
    med_abs_error = median_absolute_error(all_y_true, all_y_pred)
    maximum_error = max_error(all_y_true, all_y_pred)

    print(f" > Mean Absolute Error: {mean_abs_error:.4f}")
    print(f" > Median Absolute Error: {med_abs_error:.4f}")
    print(f" > Maximum Error: {maximum_error:.4f}")
    rank_misalignment_heatmap(final_test_df, all_y_true, all_y_pred, mean_abs_error, med_abs_error, maximum_error, best_rho)
    print()

    if best_checkpoint_path != "" and os.path.exists(os.path.join("checkpoints", best_checkpoint_path)):
        print("Loading best checkpoint...", end="")
        best_checkpoint = torch.load(os.path.join("checkpoints", best_checkpoint_path), map_location=device)
        model.load_state_dict(best_checkpoint["model_state_dict"])
        print("[green]done[/green]")

    print("Saving model and training results...", end="")
    model_file_path = os.path.join("torch_models", f"f1_constructors_ranking_model_{current_datetime}.pt")
    torch.save(model.state_dict(), model_file_path)
    
    training_data_file_path = os.path.join("training_data", f"{current_datetime}_training_data.xlsx")
    with pd.ExcelWriter(training_data_file_path) as writer:
        model_training_params_df.to_excel(writer, sheet_name="Model Training Parameters", index=False, float_format="%.4f")
        training_df.to_excel(writer, sheet_name="Model Training by Epoch", index=False, float_format="%.4f")
    print(f"[green]done[/green]")
    print(f" > Model saved to: [magenta]{model_file_path}[/magenta]")
    print(f" > Model training parameters and data saved to: [magenta]{training_data_file_path}[/magenta]", end="\n\n")