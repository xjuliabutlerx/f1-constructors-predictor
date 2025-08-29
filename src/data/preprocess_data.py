from rich import print
from setup import RAW_DATA_PATH, PREPROCESSED_DATA_PATH

import os
import pandas as pd

RAW_DATA_FILES = os.listdir(RAW_DATA_PATH)

def get_data_files_for_year(year:str):
    return [file for file in RAW_DATA_FILES if file.startswith(str(year))]

def combine_data_files_for_year(file_list:list):
    year_data_df = pd.DataFrame()
    for file in file_list:
        file_path = os.path.join(RAW_DATA_PATH, file)
        file_df = pd.read_csv(file_path)

        # Add a column for the year extracted from the file name
        file_year = file.split('_')[0]
        file_df.insert(loc=0, column='Year', value=file_year)

        # Add a column for the round number extracted from the file name
        gp_round = file.split('_')[2]
        gp_round = int(gp_round) if gp_round.isdigit() else gp_round
        file_df.insert(loc=1, column='Round', value=gp_round)

        # Add a column for the grand prix location extracted from the file name
        gp_location = file.split('_')[3]
        file_df.insert(loc=2, column='Location', value=gp_location)

        year_data_df = pd.concat([year_data_df, file_df], ignore_index=True)
    return year_data_df.sort_values(by='Round')

def autofill_driver_data_given_id(all_data_df:pd.DataFrame, driver_ids:list, id_column:str):
    result_df = pd.DataFrame()
    for driver in driver_ids:
        all_results = all_data_df[all_data_df[id_column] == driver]
        all_results = all_results[['DriverId', 'FirstName', 'LastName', 'FullName', 'Abbreviation', 'BroadcastName']].drop_duplicates().dropna()
        result_df = pd.concat([result_df, all_results], ignore_index=True)
    return result_df.sort_values(by='DriverId').reset_index(drop=True)

if __name__ == "__main__":
    print()
    years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    all_data_df = pd.DataFrame()

    print("Starting data preprocessing...")
    for year in years:
        print(f" > Processing data for the {year} season")
        files = get_data_files_for_year(year)
        year_df = combine_data_files_for_year(files)

        all_data_df = pd.concat([all_data_df, year_df], ignore_index=True)

        preprocessed_file_name = f'{year}_season_results.csv'
        year_df.to_csv(os.path.join(PREPROCESSED_DATA_PATH, preprocessed_file_name), index=False)
        print(f"   - Saved preprocessed data to [green]data/preprocessed/{preprocessed_file_name}[/green]", end='\n\n')

    print("Gathering data on drivers...")
    unique_drivers = all_data_df['DriverId'].unique()
    drivers_df = autofill_driver_data_given_id(all_data_df, unique_drivers, 'DriverId')
    drivers_df.to_csv(os.path.join(PREPROCESSED_DATA_PATH, 'drivers.csv'), index=False)
    print(f"   - Saved drivers data to [green]data/preprocessed/drivers.csv[/green]", end='\n\n')
    
    print("Data preprocessing completed.")