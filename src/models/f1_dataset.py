import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

class F1Dataset(Dataset):

    def __init__(self, data_file_path):
        feature_columns = ["Round", "RoundsCompleted", "RoundsRemaining", "AvgGridPosition", "AvgPosition", "DNFRate", "AvgPointsPerRace", \
                        "TotalPointFinishes", "TotalPodiums", "TotalPoints", "hadPenaltyThisYear"]
        skewed_feature_columns = ["DNFRate", "TotalPointFinishes", "TotalPodiums", "TotalPoints", "hadPenaltyThisYear"]

        self.df = pd.read_csv(data_file_path)

        for col in skewed_feature_columns:
            self.df[col] = np.log1p(self.df[col])

        self.data_features = torch.tensor(self.df[feature_columns].values, dtype=torch.float32)
        self.final_rankings = torch.tensor(self.df["FinalRank"].values, dtype=torch.long)

    def __len__(self):
        return len(self.data_features)

    def __getitem__(self, idx):
        return self.data_features[idx], self.final_rankings[idx]
    
if __name__ == "__main__":
    import os
    dataset = F1Dataset(os.path.join("../../data/clean/", "training_data.csv"))