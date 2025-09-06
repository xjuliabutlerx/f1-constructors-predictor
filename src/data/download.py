from rich import print
from setup_paths import RAW_DATA_PATH, CACHE_PATH, DATA_PATH

import fastf1
import fastf1.logger
import os
import time

def download_schedule(year:int, include_testing:bool=False):
    schedule = fastf1.get_event_schedule(year, include_testing=include_testing)
    return schedule

def download_session(year:int, gp:str, session_type:str='R'):
    session = fastf1.get_session(year, gp, session_type)
    session.load()
    return session

def get_locations_from_schedule(schedule):
    locations = schedule['Location'].tolist()
    return locations

def get_locations_with_sprint_from_schedule(schedule):
    sprint_locations = schedule[schedule['EventFormat'].str.contains('sprint', case=False, na=False)]
    return sprint_locations['Location'].tolist()

def get_rounds_from_schedule(schedule):
    rounds = schedule['RoundNumber'].tolist()
    return rounds

if __name__ == "__main__":
    if not os.path.exists(os.path.join(os.getcwd(), DATA_PATH)):
        os.mkdir(DATA_PATH)

    if not os.path.exists(os.path.join(os.getcwd(), CACHE_PATH)):
        os.mkdir(CACHE_PATH)

    if not os.path.exists(os.path.join(os.getcwd(), RAW_DATA_PATH)):
        os.mkdir(RAW_DATA_PATH)

    fastf1.Cache.enable_cache(CACHE_PATH) # Enable caching to speed up data retrieval
    fastf1.logger.set_log_level('ERROR')  # Set log level to ERROR to reduce verbosity

    # The years for which data for the full season is available
    data_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024] 

    # Start a timer to measure download duration
    download_start_time = time.time()
    print()

    for year in data_years:
        print(f"Downloading data for the {year} season")

        # Get the schedule for the year
        schedule = download_schedule(year)
        print(schedule[['RoundNumber', 'EventName', 'Location', 'EventFormat']])
        print()

        # Get the round numbers from the schedule
        schedule_rounds = get_rounds_from_schedule(schedule)
        schedule_locations = get_locations_from_schedule(schedule)
        sprint_locations = get_locations_with_sprint_from_schedule(schedule)

        for round in schedule_rounds:
            print(f" > Downloading data for Round {round} - {schedule_locations[round - 1]} | Is a Sprint Event? {True if schedule_locations[round - 1] in sprint_locations else False}")

            # For each round, download the results from the session
            gp_session = download_session(year, round, 'R') # for this project, we only want the race results
            gp_session.load()

            if gp_session is None or gp_session.results is None or gp_session.results.empty:
                print(f"   - [yellow]WARNING[/yellow]: No race results data returned for [red]Round {round} - {schedule_locations[round - 1]}[/red], skipping...", end='\n\n')
                continue

            # Save the results to a CSV file in the raw data directory
            data_file_name = f'{year}_Round_{round}_{schedule_locations[round - 1]}_results.csv'
            gp_session.results.to_csv(f'../../data/raw/{data_file_name}', index=False)
            print(f"   - Saved results to [green]data/raw/{data_file_name}[/green]", end='\n\n')

            if schedule_locations[round - 1] in sprint_locations:
                s_session = download_session(year, round, 'S')
                s_session.load()

                if s_session is None or s_session.results is None or s_session.results.empty:
                    print(f"   - [yellow]WARNING[/yellow]: Although this was a sprint event, no sprint results data returned for [red]Round {round} - {schedule_locations[round - 1]}[/red], skipping...", end='\n\n')
                    continue

                sprint_file_name = f'{year}_Round_{round}_{schedule_locations[round - 1]}_sprint_results.csv'
                s_session.results.to_csv(f'../../data/raw/{sprint_file_name}', index=False)
                print(f"   - Saved sprint results to [green]data/raw/{sprint_file_name}[/green]", end='\n\n')
    
    download_end_time = time.time()
    total_download_time = download_end_time - download_start_time
    print(f"Data download completed in {total_download_time:.2f} seconds")