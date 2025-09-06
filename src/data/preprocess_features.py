from rich import print
from setup_paths import PREPROCESSED_DATA_PATH, CLEAN_DATA_PATH

import os
import pandas as pd

PREPROCESSED_DATA_FILES = os.listdir(PREPROCESSED_DATA_PATH)

def normalize_teamids(df:pd.DataFrame):
    df["TeamId"] = df["TeamId"].replace("alfa", "sauber")
    df["TeamId"] = df["TeamId"].replace("renault", "alpine")
    df["TeamId"] = df["TeamId"].replace(["toro_rosso", "alphatauri"], ["rb", "rb"])
    df["TeamId"] = df["TeamId"].replace(["force_india", "racing_point"], ["aston_martin", "aston_martin"])

    df["TeamName"] = df["TeamName"].replace("Alfa Romeo Racing", "Alfa Romeo")
    df["TeamName"] = df["TeamName"].replace("Sauber", "Kick Sauber")

    return df

if __name__ == "__main__":
    print()
    print("Starting data cleaning and feature engineering...")

    years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    penalties_dict = {
        2018: [{"aston_martin": 0}], # In 2018, Force India (TeamId aston_martin) received a 59 point penalty for rebranding mid-season
                                     # (this deduction is done by dropping the points associated with the Force India TeamName value) below
        2020: [{"aston_martin": 15}] # In 2020, Racing Point (TeamId aston_martin) received a 15 point penalty for copying the brake ducts from the 2019 Mercedes car
    }

    # Read in all preprocessed data for all seasons but only keep the relevant columns
    all_seasons_data_df = pd.DataFrame()
    for file in PREPROCESSED_DATA_FILES:
        if file.endswith("_season_results.csv"):
            current_year_df = pd.read_csv(os.path.join(PREPROCESSED_DATA_PATH, file))
            all_seasons_data_df = pd.concat([all_seasons_data_df, current_year_df])
    all_seasons_data_df = all_seasons_data_df.sort_values(by=["Year", "Round", "Points"], ascending=[True, True, False]).reset_index(drop=True)

    # Normalize the TeamId values to account for name changes over the years; unlike in the eda.ipynb, we will not rename the TeamName values to
    # make comparisons with official F1 data easier
    all_seasons_data_df = normalize_teamids(all_seasons_data_df)

    # Create new columns for whether the driver did not finish (DNF), finished in a position earning points, or finished on the podium
    all_seasons_data_df["isDNF"] = all_seasons_data_df["ClassifiedPosition"].apply(lambda x: 1 if not x.isnumeric() else 0)
    all_seasons_data_df["isPointsFinish"] = all_seasons_data_df["Points"].apply(lambda x: 1 if x > 0 else 0)
    all_seasons_data_df["isPodiumFinish"] = all_seasons_data_df["Points"].apply(lambda x: 1 if x >= 15 else 0)

    # A DataFrame containing all the teams and their total points at the end of each season
    # Note: In 2022, the official F1 website has Alfa Romeo (TeamId sauber) in place 6 and Aston Martin (TeamId aston_martin) in place 7 despite the tied points
    #       By sorting by ascending Year, descending Points, and ascending TeamName, we ensure that Alfa Romeo is ranked higher than Aston Martin in the final standings
    final_standings_df = all_seasons_data_df[["Year", "TeamId", "TeamName", "Points"]].groupby(["Year", "TeamId", "TeamName"])["Points"].sum().reset_index()
    final_standings_df = final_standings_df.sort_values(by=["Year", "Points", "TeamName"], ascending=[True, False, True]).reset_index(drop=True)

    # The DataFrame with the features and target variable (FinalRank) for training the model
    training_data_df = pd.DataFrame()

    print()
    for year in years:
        print(f" > Processing data for the {year} season")
        print("   - Calculating final standings...")
        current_year_standings = final_standings_df[final_standings_df["Year"] == year]
        current_teams_with_penalties = [list(penalty.keys())[0] for penalty in penalties_dict[year]] if year in penalties_dict.keys() else []

        # In 2018, Force India (TeamId aston_martin) received a 59 point penalty for rebranding mid-season; deleting any rows with the TeamName "Force India"
        if year == 2018:
            current_year_standings = current_year_standings.drop(current_year_standings[current_year_standings["TeamName"] == "Force India"].index)

        # This is generalized logic for applying any point penalties to teams in any year (not for re-branding mid-season)
        if year in penalties_dict.keys():
            for penalty in penalties_dict[year]:
                for team_id, points in penalty.items():
                    current_year_standings.loc[current_year_standings["TeamId"] == team_id, "Points"] -= points
        
        current_year_standings = current_year_standings.sort_values(by=["Points", "TeamName"], ascending=[False, True]).reset_index(drop=True)
        current_year_standings["FinalRanking"] = [i + 1 for i in range(len(current_year_standings))]
        print(current_year_standings, end="\n\n")

        current_teamids_list = current_year_standings["TeamId"].unique().tolist()
        current_race_locations = all_seasons_data_df[all_seasons_data_df["Year"] == year]["Location"].unique().tolist()

        for team_id in current_teamids_list:
            print(f"   - Processing the {year} season data for [magenta]{current_year_standings.loc[current_year_standings['TeamId'] == team_id, 'TeamName'].iloc[0]}[/magenta] ({team_id})...")
            current_team_results_df = pd.DataFrame()
            for location in current_race_locations:
                temp_df = pd.DataFrame()
                # Pull the race results for each of the drivers of the current team; there will be a row for each driver for each race and/or sprint
                current_race_results = all_seasons_data_df.loc[(all_seasons_data_df["TeamId"] == team_id) & (all_seasons_data_df["Year"] == year) & \
                                                                  (all_seasons_data_df["Location"] == location)]
                
                # Calculate the data features for the team
                temp_df["Year"] = current_race_results["Year"]
                temp_df["TeamId"] = current_race_results["TeamId"]
                temp_df["TeamName"] = current_race_results["TeamName"]
                temp_df["Location"] = current_race_results["Location"]
                temp_df["Round"] = current_race_results["Round"]
                temp_df["RoundsCompleted"] = current_race_results["Round"] - 1
                temp_df["RoundsRemaining"] = len(current_race_locations) - current_race_results["Round"]
                temp_df["AvgGridPosition"] = current_race_results["GridPosition"].expanding().mean().tail(1)
                temp_df["AvgPosition"] = current_race_results["Position"].expanding().mean().tail(1)
                temp_df["DNFRate"] = current_race_results["isDNF"].expanding().mean().tail(1)
                temp_df["AvgPointsPerRace"] = current_race_results["Points"].expanding().mean().tail(1)
                temp_df["TotalPointFinishes"] = current_race_results["isPointsFinish"].cumsum()
                temp_df["TotalPodiums"] = current_race_results["isPodiumFinish"].cumsum()
                temp_df["TotalPoints"] = current_race_results["Points"].cumsum()
                temp_df["hadPenaltyThisYear"] = 1 if team_id in current_teams_with_penalties else 0
                temp_df["FinalRank"] = current_year_standings.loc[current_year_standings["TeamId"] == team_id, "FinalRanking"].iloc[0]
                temp_df = temp_df.dropna(how="any").reset_index(drop=True)

                # Add the team-level results of this race to the DataFrame for the team's results for the season
                current_team_results_df = pd.concat([current_team_results_df, temp_df], ignore_index=True)
            
            # Re-apply the expanding mean and cumulative sum functions to get the data features at each race throughout the season
            current_team_results_df["AvgGridPosition"] = current_team_results_df["AvgGridPosition"].expanding().mean()
            current_team_results_df["AvgPosition"] = current_team_results_df["AvgPosition"].expanding().mean()
            current_team_results_df["DNFRate"] = current_team_results_df["DNFRate"].expanding().mean()
            current_team_results_df["AvgPointsPerRace"] = current_team_results_df["AvgPointsPerRace"].expanding().mean()
            current_team_results_df["TotalPointFinishes"] = current_team_results_df["TotalPointFinishes"].cumsum()
            current_team_results_df["TotalPodiums"] = current_team_results_df["TotalPodiums"].cumsum()
            current_team_results_df["TotalPoints"] = current_team_results_df["TotalPoints"].cumsum()

            print(current_team_results_df[["Location", "RoundsCompleted", "RoundsRemaining", "AvgGridPosition", "AvgPosition", "DNFRate", \
                                           "AvgPointsPerRace", "TotalPointFinishes", "TotalPodiums", "TotalPoints", "hadPenaltyThisYear", "FinalRank"]], \
                                            end="\n\n")
            
            # Add the current team's results for the season to the overall training data DataFrame
            training_data_df = pd.concat([training_data_df, current_team_results_df], ignore_index=True)
    
    # Export the cleaned all seasons data and the training data to CSV files
    all_seasons_data_df.to_csv(os.path.join(CLEAN_DATA_PATH, "all_seasons_data.csv"), index=False)
    training_data_df.to_csv(os.path.join(CLEAN_DATA_PATH, "f1_clean_data.csv"), index=False)
    print("Data cleaning and feature engineering completed.", end="\n\n")
