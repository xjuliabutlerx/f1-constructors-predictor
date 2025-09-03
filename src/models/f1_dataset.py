from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import torch

class F1Dataset(Dataset):

    def __init__(self, data_file_path):
        feature_columns = ["Round", "RoundsCompleted", "RoundsRemaining", "AvgGridPosition", "AvgPosition", "DNFRate", "AvgPointsPerRace", \
                        "TotalPointFinishes", "TotalPodiums", "TotalPoints", "hadPenaltyThisYear"]
        skewed_feature_columns = ["DNFRate", "TotalPointFinishes", "TotalPodiums", "TotalPoints", "hadPenaltyThisYear"]

        self.df = pd.read_csv(data_file_path)

        for col in skewed_feature_columns:
            self.df[col] = np.log1p(self.df[col])

        self.X = self.df[feature_columns].copy()
        self.y = self.df["FinalRank"].copy()

        self.data_features = torch.tensor(self.df[feature_columns].values, dtype=torch.float32)
        self.final_rankings = torch.tensor(self.df["FinalRank"].values, dtype=torch.long)

    def __len__(self):
        return len(self.data_features)

    def __getitem__(self, idx):
        return self.data_features[idx], self.final_rankings[idx]
    
    def get_random_split(self, test_size=0.2, random_state=24):
        # Randomly split the data into training and testing datasets
        # X_train and X_test will be DataFrame objects
        # y_train and y_test will be Series objects
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state, shuffle=True)
        
        # Convert the output into tensors
        X_train = torch.tensor(X_train.values, dtype=torch.float32)
        X_test = torch.tensor(X_test.values, dtype=torch.float32)
        y_train = torch.tensor(y_train.to_numpy(), dtype=torch.long)
        y_test = torch.tensor(y_test.to_numpy(), dtype=torch.long)
        
        # Return the split data as tensors
        return X_train, X_test, y_train, y_test
    
if __name__ == "__main__":
    import os
    dataset = F1Dataset(os.path.join("../../data/clean/", "f1_clean_data.csv"))