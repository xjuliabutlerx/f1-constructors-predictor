from datetime import datetime
from f1_constructors_rank_classifier import F1ConstructorsClassifier
from f1_dataset import F1Dataset
from rich import print
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import accuracy_score, confusion_matrix

import argparse
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
PARSER.add_argument("--test_size", "-s", type=float, default=0.2)
PARSER.add_argument("--use_weighted_loss_func", "-w", type=bool, default=False)
PARSER.add_argument("--margin", "-m", type=float, default=1.0)
PARSER.add_argument("--patience", "-p", type=int, default=15)

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

def adjust_learning_rate(learning_rate, epoch):
    lr = learning_rate

    if epoch > 5:
        lr = 0.001
    elif epoch >= 10:
        lr = 0.0001
    elif epoch >= 20:
        lr = 0.00001

    for param_group in optimizer.param_groups():
        param_group["lr"] = lr

import numpy as np
from scipy.stats import spearmanr

def predicted_ranks_from_scores_by_season(scores, seasons):
    """
    Convert an array of model scores into 1-based ranks per season.
    - scores: 1D numpy array of shape [n_test_rows], higher = better
    - seasons: 1D array-like same length with season id (e.g. year) for each row
    Returns: pred_ranks: 1D numpy int array (1..k) aligned with input order.
    """
    scores = np.asarray(scores)
    seasons = np.asarray(seasons)
    assert scores.shape[0] == seasons.shape[0]

    pred_ranks = np.empty_like(scores, dtype=int)
    for season in np.unique(seasons):
        idx = np.where(seasons == season)[0]
        # argsort descending so highest score gets rank 1
        order_desc = np.argsort(-scores[idx])
        ranks = np.empty_like(order_desc)
        ranks[order_desc] = np.arange(1, len(idx) + 1)  # 1-based ranks
        pred_ranks[idx] = ranks

    return pred_ranks


def per_season_spearman_and_top1(val_ranks, true_ranks, seasons):
    seasons = np.asarray(seasons)
    val_ranks = np.asarray(val_ranks)
    true_ranks = np.asarray(true_ranks)

    season_rhos = {}
    top1_hits = 0
    season_count = 0

    for season in np.unique(seasons):
        idx = np.where(seasons == season)[0]
        if len(idx) < 2:
            continue
        pred = val_ranks[idx]
        true = true_ranks[idx]

        # Spearman's rho
        rho, _ = spearmanr(pred, true)
        season_rhos[season] = float(rho) if not np.isnan(rho) else None

        # Top-1 check (predicted champion is rank 1)
        pred_champion_idx = idx[np.argmin(pred)]  # index of predicted rank==1 in this season
        true_champion_idx = idx[np.argmin(true)]
        if pred_champion_idx == true_champion_idx:
            top1_hits += 1
        season_count += 1

    avg_rho = np.nanmean([v for v in season_rhos.values() if v is not None]) if season_rhos else np.nan
    top1_acc = top1_hits / season_count if season_count > 0 else np.nan

    return avg_rho, season_rhos, top1_acc

def rank_misalignment_heatmap(y_true, y_pred_ranks, title="Rank Misalignment Heatmap"):
    n = len(np.unique(y_true))  # number of teams
    print(f"n = {n}")
    mat = np.zeros((n, n), dtype=int)

    # Build misalignment matrix
    for t, p in zip(y_true, y_pred_ranks):
        mat[int(t)-1, int(p)-1] += 1  # shift to 0-index

    # Misplacement metric
    misplacements = np.abs(np.array(y_true) - np.array(y_pred_ranks))
    mean_abs_error = np.mean(misplacements)
    median_abs_error = np.median(misplacements)
    max_error = np.max(misplacements)

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(mat, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=[f"P{r}" for r in range(1, n+1)],
                yticklabels=[f"P{r}" for r in range(1, n+1)])
    plt.xlabel("Predicted Rank")
    plt.ylabel("True Rank")
    plt.title(f"{title}\nMean Absolute Error={mean_abs_error:.2f}, Median={median_abs_error:.2f}, Max={max_error}")
    plt.savefig(os.path.join("./heatmaps", f"model_heatmap_{datetime.now().strftime('%Y-%m-%d_%H:%M')}.png"))
    plt.show()

    return {
        "matrix": mat,
        "mean_absolute_error": mean_abs_error,
        "median_absolute_error": median_abs_error,
        "max_error": max_error
    }

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        scores = model(X_test).squeeze().cpu().numpy()
    
    # Convert scores into predicted ranks
    pred_order = np.argsort(-scores)  # highest score = best rank
    true_order = np.argsort(y_test.cpu().numpy())
    
    # Map each team index -> predicted rank
    pred_ranks = np.empty_like(pred_order)
    pred_ranks[pred_order] = np.arange(len(pred_order))
    true_ranks = np.empty_like(true_order)
    true_ranks[true_order] = np.arange(len(true_order))
    
    # Confusion matrix
    return rank_misalignment_heatmap(true_ranks, pred_ranks)

if __name__ == "__main__":
    print()

    test_size = ARGS.test_size
    num_epochs = ARGS.num_epochs
    learning_rate = ARGS.learning_rate
    use_weighted_loss_func = ARGS.use_weighted_loss_func
    margin = ARGS.margin
    patience = ARGS.patience

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M")

    if test_size > 1:
        print(f"[red]ERROR[/red]: Test size cannot be more than 1; received {test_size}")
        exit(0)
    elif test_size <= 0:
        print(f"[red]ERROR[/red]: Test size cannot be less than or equal to 0; received {test_size}")
        exit(0)

    print("[yellow]*** TRAINING F1 CONSTRUCTOR CLASSIFIER MODEL ***[/yellow]")
    print()

    print("Training Parameters:")
    print(f" > Test Size: {test_size * 100}%")
    print(f" > Number of Epochs: {num_epochs}")
    print(f" > Intial Learning Rate: {learning_rate}")
    print(f" > Training Device: ", end="")
    device = get_device()
    print(f" > Loss Function: Margin Ranking Loss")
    print(f" > Loss Function Margin: {margin}")
    print(f" > Patience: {patience}")
    print()

    print("Data:")
    print(f" > Loading dataset...", end="")
    dataset = F1Dataset(os.path.join("../../data/clean/", "f1_clean_data.csv"))
    print(f"[green]done[/green]")

    print(f" > Splitting dataset into training and testing...", end="")
    X_train, X_test, y_train, y_test = dataset.get_random_split(test_size=0.2, random_state=24)
    print(f"[green]done[/green]")
    print()

    print("Model and Model Parameters:")
    print(f" > Loading model into training device...", end="")
    model = F1ConstructorsClassifier(X_train.shape[1], 1).to(device)
    print(f"[green]done[/green]")

    print(f" > Instantiating optimizer with learning rate...", end="")
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    print(f"[green]done[/green]")

    print(f" > Instantiating loss function...", end="")
    loss_func = nn.MarginRankingLoss(margin=margin)
    print(f"[green]done[/green]")
    print()

    print("Training model...")
    training_df = pd.DataFrame(columns=["Epoch", "Training Loss", "Spearman's Rho", "Kendall's Tau"])
    best_rho = -1.0
    best_epoch = -1
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        scores = model(X_train)

        i_idx = torch.randint(0, len(y_train), (len(y_train),), device=device)
        j_idx = torch.randint(0, len(y_train), (len(y_train),), device=device)

        s_i, s_j = scores[i_idx], scores[j_idx]

        target = torch.where(y_train[i_idx] < y_train[j_idx], 1, -1).float()

        loss = loss_func(s_i, s_j, target)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_scores = model(X_test).cpu().numpy()
            years_test = pd.DataFrame(X_test.to_numpy())["Year"]
            y_true = y_test.cpu().numpy()

            val_rank = np.empty_like(val_scores, dtype=int)

            spearman_scores = {}
            kendall_scores = {}

            for year in np.unique(years_test):
                idx = np.where(years_test == year)[0]           # rows for that year

                true_ranks = y_true[idx]
                pred_ranks = val_rank[idx]

                # Compute correlation
                rho, _ = spearmanr(true_ranks, pred_ranks)
                tau, _ = kendalltau(true_ranks, pred_ranks)

                spearman_scores[year] = rho
                kendall_scores[year] = tau

                scores_year = val_scores[idx]
                order_desc = np.argsort(-scores_year)           # sort high→low
                ranks = np.empty_like(order_desc)
                ranks[order_desc] = np.arange(1, len(idx) + 1)  # assign 1..k
                val_rank[idx] = ranks

            avg_rho, per_season_rho, top1_accuracy = per_season_spearman_and_top1(val_rank, y_true)

        # Early stopping & checkpoint
        improved = rho > best_rho + 1e-4  # tiny tolerance
        if improved:
            best_rho = rho
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join("./checkpoints", f"best_ranking_model_{current_datetime}.pth"))
        else:
            epochs_no_improve += 1

        print(f" > Epoch {epoch+1}/{num_epochs} | Training Loss: {loss.item():.4f} | Spearman's rho: {rho:.4f} | Kendall's tau: {tau:.4f}")
        training_df.loc[len(training_df)] = [epoch + 1, loss.item(), rho, tau]

        if epochs_no_improve >= patience:
            print()
            print(f"[yellow]Early Stopping[/yellow]: no improvement for {patience} epochs. Best rho={best_rho:.4f} @ epoch {best_epoch}.")
            break

    print()

    # print("Evaluating model...")    
    # final_heatmap_stats = evaluate_model(model, X_test, y_test)
    # print("Matrix:")
    # print(final_heatmap_stats["matrix"])
    # print(f"Mean Absolute Error: {final_heatmap_stats['mean_absolute_error']}")
    # print(f"Median Absolute Error: {final_heatmap_stats['median_absolute_error']}")
    # print(f"Maximum Error: {final_heatmap_stats['max_error']}")
    # print()

    print("Saving model and training results...", end="")
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M")
    torch.save(model.state_dict(), 
               os.path.join("torch_models", f"f1_constructors_ranking_model_{current_datetime}.pt"))
    training_df.to_csv(os.path.join("./training_data", f"training_data_{current_datetime}.csv"), index=False)
    print(f"[green]done[/green]", end="\n\n")