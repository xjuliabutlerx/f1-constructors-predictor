from rich import print
from setup_paths import RAW_DATA_PATH, CACHE_PATH, DATA_PATH, PREPROCESSED_DATA_PATH, CLEAN_DATA_PATH

import argparse
import fastf1
import fastf1.logger
import os
import pandas as pd
import time

# -------------------- DOWNLOAD FUNCTIONS --------------------
def download_schedule(year: int, include_testing: bool = False):
    schedule = fastf1.get_event_schedule(year, include_testing=include_testing)
    return schedule

def download_session(year: int, gp, session_type: str = 'R', max_retries: int = 3, retry_delay: int = 5):
    for attempt in range(max_retries):
        try:
            session = fastf1.get_session(year, gp, session_type)
            session.load()
            return session
        except Exception as e:
            print(f"   - [yellow]WARNING[/yellow]: Error downloading session (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print(f"   - [red]ERROR[/red]: Failed to download after {max_retries} attempts.")
                return None

def get_locations_from_schedule(schedule):
    return schedule['Location'].tolist()

def get_locations_with_sprint_from_schedule(schedule):
    sprint_locations = schedule[schedule['EventFormat'].str.contains('sprint', case=False, na=False)]
    return sprint_locations['Location'].tolist()

def get_rounds_from_schedule(schedule):
    return schedule['RoundNumber'].tolist()

def download_all_data(data_years):
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(CACHE_PATH, exist_ok=True)
    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    
    fastf1.Cache.enable_cache(CACHE_PATH)
    fastf1.logger.set_log_level('ERROR')
    
    download_start_time = time.time()
    
    for year in data_years:
        print(f"Downloading data for the {year} season")
        schedule = download_schedule(year)
        
        print(schedule[['RoundNumber', 'EventName', 'Location', 'EventFormat']])
        print()
        
        schedule_rounds = get_rounds_from_schedule(schedule)
        schedule_locations = get_locations_from_schedule(schedule)
        sprint_locations = get_locations_with_sprint_from_schedule(schedule)
        
        for round in schedule_rounds:
            loc = schedule_locations[round - 1]
            is_sprint = loc in sprint_locations
            print(f" > Downloading data for Round {round} - {loc} | Is a Sprint Event? {is_sprint}")
            gp_session = download_session(year, round, 'R')
            
            if gp_session is None or gp_session.results is None or gp_session.results.empty:
                print(f"   - [yellow]WARNING[/yellow]: No race results data returned for [red]Round {round} - {loc}[/red], skipping...\n")
                continue
            
            data_file_name = f'{year}_Round_{round}_{loc}_results.csv'
            gp_session.results.to_csv(os.path.join(RAW_DATA_PATH, data_file_name), index=False)
            print(f"   - Saved results to [green]{RAW_DATA_PATH}/{data_file_name}[/green]\n")
            
            if is_sprint:
                s_session = download_session(year, round, 'S')
                
                if s_session is None or s_session.results is None or s_session.results.empty:
                    print(f"   - [yellow]WARNING[/yellow]: Although this was a sprint event, no sprint results data returned for [red]Round {round} - {loc}[/red], skipping...\n")
                    continue
                
                sprint_file_name = f'{year}_Round_{round}_{loc}_sprint_results.csv'
                s_session.results.to_csv(os.path.join(RAW_DATA_PATH, sprint_file_name), index=False)
                print(f"   - Saved sprint results to [green]{RAW_DATA_PATH}/{sprint_file_name}[/green]\n")
    
    print(f"Data download completed in {time.time() - download_start_time:.2f} seconds")

# -------------------- PREPROCESS DATA FUNCTIONS --------------------
def get_data_files_for_year(year: str, raw_data_files):
    return [file for file in raw_data_files if file.startswith(str(year))]

def combine_data_files_for_year(file_list: list):
    year_data_df = pd.DataFrame()
    for file in file_list:
        file_path = os.path.join(RAW_DATA_PATH, file)
        file_df = pd.read_csv(file_path)
        file_year = file.split('_')[0]
        file_df.insert(loc=0, column='Year', value=file_year)
        gp_round = file.split('_')[2]
        gp_round = int(gp_round) if gp_round.isdigit() else gp_round
        file_df.insert(loc=1, column='Round', value=gp_round)
        gp_location = file.split('_')[3]
        file_df.insert(loc=2, column='Location', value=gp_location)
        year_data_df = pd.concat([year_data_df, file_df], ignore_index=True)
    return year_data_df.sort_values(by='Round')

def autofill_driver_data_given_id(all_data_df: pd.DataFrame, driver_ids: list, id_column: str):
    result_df = pd.DataFrame()
    for driver in driver_ids:
        all_results = all_data_df[all_data_df[id_column] == driver]
        all_results = all_results[['DriverId', 'FirstName', 'LastName', 'FullName', 'Abbreviation', 'BroadcastName']].drop_duplicates().dropna()
        result_df = pd.concat([result_df, all_results], ignore_index=True)
    return result_df.sort_values(by='DriverId').reset_index(drop=True)

def preprocess_all_data(years):
    print("\nStarting data preprocessing...")
    
    all_data_df = pd.DataFrame()
    raw_data_files = os.listdir(RAW_DATA_PATH)
    
    os.makedirs(PREPROCESSED_DATA_PATH, exist_ok=True)
    
    for year in years:
        print(f" > Processing data for the {year} season")
        files = get_data_files_for_year(year, raw_data_files)
        year_df = combine_data_files_for_year(files)
        all_data_df = pd.concat([all_data_df, year_df], ignore_index=True)
        preprocessed_file_name = f'{year}_season_results.csv'
        year_df.to_csv(os.path.join(PREPROCESSED_DATA_PATH, preprocessed_file_name), index=False)
        print(f"   - Saved preprocessed data to [green]{PREPROCESSED_DATA_PATH}/{preprocessed_file_name}[/green]\n")
    
    print("Gathering data on drivers...")
    
    unique_drivers = all_data_df['DriverId'].unique()
    drivers_df = autofill_driver_data_given_id(all_data_df, unique_drivers, 'DriverId')
    drivers_df.to_csv(os.path.join(PREPROCESSED_DATA_PATH, 'drivers.csv'), index=False)
    
    print(f"   - Saved drivers data to [green]{PREPROCESSED_DATA_PATH}/drivers.csv[/green]\n")
    print("Data preprocessing completed.")

# -------------------- FEATURE ENGINEERING FUNCTIONS --------------------
def calculate_ranks_after_rounds(training_data_df:pd.DataFrame):
    # Calculate each team's rank after the given round
    training_data_df["CurrentRankAfterRound"] = training_data_df.groupby(["Year", "Round"])["TotalPoints"].rank(method="dense", ascending=False).astype(int)
    
    # Calculate the percentile of each team's rank after the given round (to provide an idea of where the team stands among the other teams)
    training_data_df["PercentileRankAfterRound"] = 1.0 - (training_data_df["CurrentRankAfterRound"] - 1) / (training_data_df.groupby(["Year","Round"])["TeamId"].transform('nunique') - 1)

    training_data_df = training_data_df[["Year", "TeamId", "TeamName", "Location", "Round", "RoundsCompleted", "RoundsRemaining", \
                                         "PointsEarnedThisRound", "DNFsThisRound", "PointsLast3Rounds", "DNFsLast3Rounds", "DNFRate", \
                                         "AvgGridPosition", "AvgPosition", "AvgPointsPerRace", "TotalPointFinishes", \
                                         "TotalPodiums", "TotalPoints", "hadPenaltyThisYear", "CurrentRankAfterRound", \
                                         "PercentileRankAfterRound", "FinalRank"]]
    
    return training_data_df

def normalize_teamids(df:pd.DataFrame):
    df["TeamId"] = df["TeamId"].replace("alfa", "sauber")
    df["TeamId"] = df["TeamId"].replace("renault", "alpine")
    df["TeamId"] = df["TeamId"].replace(["toro_rosso", "alphatauri"], ["rb", "rb"])
    df["TeamId"] = df["TeamId"].replace(["force_india", "racing_point"], ["aston_martin", "aston_martin"])
    df["TeamName"] = df["TeamName"].replace("Alfa Romeo Racing", "Alfa Romeo")
    df["TeamName"] = df["TeamName"].replace("Sauber", "Kick Sauber")
    return df

def feature_engineer_all_data(years, incomplete_years=None):
    print("\nStarting data cleaning and feature engineering...")
    penalties_dict = {
        2018: [{"aston_martin": 0}],
        2020: [{"aston_martin": 15}]
    }
    all_seasons_data_df = pd.DataFrame()
    preprocessed_files = os.listdir(PREPROCESSED_DATA_PATH)
    
    for file in preprocessed_files:
        if file.endswith("_season_results.csv"):
            current_year_df = pd.read_csv(os.path.join(PREPROCESSED_DATA_PATH, file))
            all_seasons_data_df = pd.concat([all_seasons_data_df, current_year_df])
    
    all_seasons_data_df = all_seasons_data_df.sort_values(by=["Year", "Round", "Points"], ascending=[True, True, False]).reset_index(drop=True)
    all_seasons_data_df = normalize_teamids(all_seasons_data_df)
    all_seasons_data_df["isDNF"] = all_seasons_data_df["ClassifiedPosition"].apply(lambda x: 1 if not str(x).isnumeric() else 0)
    all_seasons_data_df["isPointsFinish"] = all_seasons_data_df["Points"].apply(lambda x: 1 if x > 0 else 0)
    all_seasons_data_df["isPodiumFinish"] = all_seasons_data_df["Points"].apply(lambda x: 1 if x >= 15 else 0)
    
    final_standings_df = all_seasons_data_df[["Year", "TeamId", "TeamName", "Points"]].groupby(["Year", "TeamId", "TeamName"])["Points"].sum().reset_index()
    final_standings_df = final_standings_df.sort_values(by=["Year", "Points", "TeamName"], ascending=[True, False, True]).reset_index(drop=True)
    
    training_data_df = pd.DataFrame()
    incomplete_training_data_df = pd.DataFrame()
    
    os.makedirs(CLEAN_DATA_PATH, exist_ok=True)
    
    for year in years:
        print(f" > Processing data for the {year} season")
        print("   - Calculating final standings...")
        
        current_year_standings = final_standings_df[final_standings_df["Year"] == year]
        current_teams_with_penalties = [list(penalty.keys())[0] for penalty in penalties_dict[year]] if year in penalties_dict.keys() else []
        
        if year == 2018:
            current_year_standings = current_year_standings.drop(current_year_standings[current_year_standings["TeamName"] == "Force India"].index)
        
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
                
                # Get the current race results for a specific team from a specific year from a specific round
                current_race_results = all_seasons_data_df.loc[(all_seasons_data_df["TeamId"] == team_id) & (all_seasons_data_df["Year"] == year) & (all_seasons_data_df["Location"] == location)]
                
                # Raw statistics
                temp_df["Year"] = current_race_results["Year"]
                temp_df["TeamId"] = current_race_results["TeamId"]
                temp_df["TeamName"] = current_race_results["TeamName"]
                temp_df["Location"] = current_race_results["Location"]
                temp_df["Round"] = current_race_results["Round"]
                temp_df["RoundsCompleted"] = current_race_results["Round"] - 1
                temp_df["RoundsRemaining"] = len(current_race_locations) - current_race_results["Round"]
                temp_df["DNFsThisRound"] = current_race_results["isDNF"].sum()
                temp_df["PointsEarnedThisRound"] = current_race_results["Points"].sum()

                # Statistics that will be re-calculated over the course of the season
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
                
                current_team_results_df = pd.concat([current_team_results_df, temp_df], ignore_index=True)
            
            current_team_results_df["PointsLast3Rounds"] = current_team_results_df.groupby(["Year", "TeamId"])["PointsEarnedThisRound"] \
                                                                                .rolling(window=3, min_periods=1).sum().reset_index(level=[0, 1], drop=True)
            current_team_results_df["DNFsLast3Rounds"] = current_team_results_df.groupby(["Year", "TeamId"])["DNFsThisRound"] \
                                                                                .rolling(window=3, min_periods=1).sum().reset_index(level=[0, 1], drop=True)
            
            current_team_results_df["AvgGridPosition"] = current_team_results_df["AvgGridPosition"].expanding().mean()
            current_team_results_df["AvgPosition"] = current_team_results_df["AvgPosition"].expanding().mean()
            current_team_results_df["DNFRate"] = current_team_results_df["DNFRate"].expanding().mean()
            current_team_results_df["AvgPointsPerRace"] = current_team_results_df["AvgPointsPerRace"].expanding().mean()
            current_team_results_df["TotalPointFinishes"] = current_team_results_df["TotalPointFinishes"].cumsum()
            current_team_results_df["TotalPodiums"] = current_team_results_df["TotalPodiums"].cumsum()
            current_team_results_df["TotalPoints"] = current_team_results_df["TotalPoints"].cumsum()
            
            # Split into full/incomplete years
            if incomplete_years and year in incomplete_years:
                schedule = download_schedule(year)
                total_rounds = len(get_rounds_from_schedule(schedule))
                current_team_results_df["RoundsRemaining"] = total_rounds - current_team_results_df["RoundsCompleted"]

                incomplete_training_data_df = pd.concat([incomplete_training_data_df, current_team_results_df], ignore_index=True)
            else:
                training_data_df = pd.concat([training_data_df, current_team_results_df], ignore_index=True)

            print(current_team_results_df[["Location", "RoundsCompleted", "RoundsRemaining", "PointsEarnedThisRound", "DNFsThisRound", \
                                           "PointsLast3Rounds", "DNFsLast3Rounds", "DNFRate", "AvgGridPosition", "AvgPosition", \
                                           "AvgPointsPerRace", "TotalPointFinishes", "TotalPodiums", "TotalPoints", \
                                           "hadPenaltyThisYear", "FinalRank"]], end="\n\n")

    if not training_data_df.empty:
        # Calculate approximate rank and percentile of rank after each round
        training_data_df = calculate_ranks_after_rounds(training_data_df)

        training_data_df.to_csv(os.path.join(CLEAN_DATA_PATH, "f1_clean_data.csv"), index=False)
        print(f"Saved full seasons to [magenta]f1_clean_data.csv[/magenta]")
    
    if not incomplete_training_data_df.empty:
        # Calculate approximate rank and percentile of rank after each round
        incomplete_training_data_df = calculate_ranks_after_rounds(incomplete_training_data_df)

        # Rename the "FinalRank" column to "CurrentRank"
        incomplete_training_data_df = incomplete_training_data_df.rename(columns={"FinalRank": "CurrentRank"})

        incomplete_training_data_df.to_csv(os.path.join(CLEAN_DATA_PATH, "f1_clean_prediction_data.csv"), index=False)
        print(f"Saved incomplete seasons to [magenta]f1_clean_prediction_data.csv[/magenta]")
    
    all_seasons_data_df.to_csv(os.path.join(CLEAN_DATA_PATH, "all_seasons_data.csv"), index=False)
    print(f"Saved all seasons data to [magenta]all_seasons_data.csv[/magenta]", end="\n\n")

    print("Data cleaning and feature engineering completed.", end="\n\n")

# -------------------- MAIN PIPELINE --------------------
def main():
    print()
    parser = argparse.ArgumentParser(description="F1 Data Pipeline: Download, preprocess, and feature engineer F1 data.")
    parser.add_argument('--step', choices=['all', 'download', 'preprocess', 'features'], default='all', help='Which step(s) to run')
    parser.add_argument('--years', nargs='+', type=int, default=[2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025], help='Years to process')
    parser.add_argument('--incomplete-years', nargs='*', type=int, default=[2025], help='Years with incomplete data (will be saved separately)')

    args = parser.parse_args()

    # Warn if any incomplete_years are not in years
    missing_incomplete = [y for y in args.incomplete_years if y not in args.years]
    if missing_incomplete:
        print(f"[yellow]WARNING:[/yellow] The following incomplete years are not in --years and will be ignored: {missing_incomplete}")

    if args.step in ['all', 'download']:
        download_all_data(args.years)

    if args.step in ['all', 'preprocess']:
        preprocess_all_data(args.years)

    if args.step in ['all', 'features']:
        feature_engineer_all_data(args.years, incomplete_years=args.incomplete_years)

if __name__ == "__main__":
    main()
