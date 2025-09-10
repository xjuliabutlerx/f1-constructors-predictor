from torch.utils.data import Dataset

import numpy as np
import pandas as pd

class F1Dataset(Dataset):

    def __init__(self, data_file_path):
        self.feature_columns = ["Year", "Round", "RoundsRemaining", "PointsEarnedThisRound", "DNFsThisRound", \
                                "PointsLast3Rounds", "DNFsLast3Rounds", "DNFRate", "AvgGridPosition", "AvgPosition", \
                                "AvgPointsPerRace", "TotalPointFinishes", "TotalPodiums", "TotalPoints", "hadPenaltyThisYear", \
                                "PercentileRankAfterRound"]

        self.df = pd.read_csv(data_file_path)

    def __len__(self):
        return len(self.data_features)

    def __getitem__(self, idx):
        return self.data_features[idx], self.final_rankings[idx]
    
    def get_years(self):
        return self.df["Year"].unique().tolist()
    
    def get_total_rounds_for_year(self, year:int):
        if year not in self.get_years():
            return 0
        return int(self.df.loc[self.df["Year"] == year, "Round"].max())

    def get_random_split(self, test_size=0.2, random_state=24):
        # Randomly split the data into training and testing datasets using the pandas random sample method
        test_df = self.df.sample(frac=test_size, random_state=random_state)
        train_df = self.df.drop(test_df.index).sample(frac=1.0)                 # frac specifies the fraction of rows to return
                                                                                # frac = 1 means return all rows in a random order

        # Split the features from the metadata and target variable columns for the training dataset to maintain "row alignment"
        X_train = train_df[self.feature_columns]
        y_train = train_df["FinalRank"]

        return X_train, y_train, test_df
    
    def get_feature_columns(self):
        return self.feature_columns
    
if __name__ == "__main__":
    import os
    dataset = F1Dataset(os.path.join("../../data/clean/", "f1_clean_data.csv"))