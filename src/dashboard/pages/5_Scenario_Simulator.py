from datetime import datetime

import numpy as np
import os
import pandas as pd
import random
import statistics
import streamlit as st

def validate_f1_points(edited_df: pd.DataFrame):
    valid_points = {25, 18, 15, 12, 10, 8, 6, 4, 2, 1, 0}

    # Find invalid entries
    invalid_rows = edited_df.loc[~edited_df["PredictedPoints"].isin(valid_points)]

    if not invalid_rows.empty:
        st.error(
            "🚨 Invalid Predicted Points Detected!\n\n"
            "Allowed values: 25, 18, 15, 12, 10, 8, 6, 4, 2, 1, or 0.\n\n"
            "Invalid rows:\n" + str(invalid_rows[["BroadcastName", "PredictedPoints"]].values.tolist())
        )
        return False

    return True

def is_podium_finish(row):
    if row["PredictedPoints"] >= 15:
        return 1
    else:
        return 0

def is_points_finish(row):
    if row["PredictedPoints"] > 0:
        return 1
    else:
        return 0

def is_dnf(row):
    if row["DNF"]:
        return 1
    else:
        return 0

def generate_features(input_df:pd.DataFrame, teams:list):
    columns = data_features = ['Year', 'TeamName', 'Round', 'RoundsCompleted', 'RoundsRemaining',
       'DNFsThisRound', 'PointsEarnedThisRound', 'MaxDriverPoints',
       'MinDriverPoints', 'DriverPointsGap', 'AvgGridPosition', 'AvgPosition',
       'DNFRate', 'AvgPointsPerRace', 'TotalPointFinishes', 'TotalPodiums',
       'TotalPoints', 'hadPenaltyThisYear', 'PointsLast3Rounds',
       'DNFsLast3Rounds', 'FormRatio', 'Consistency', 'ProjectedGrowth',
       'ProjectedSeasonTotalPoints']

    data_features = ['Year', 'TeamName', 'Round', 'RoundsCompleted', 'RoundsRemaining',
       'DNFsThisRound', 'PointsEarnedThisRound', 'MaxDriverPoints',
       'MinDriverPoints', 'DriverPointsGap', 'AvgGridPosition', 'AvgPosition',
       'DNFRate', 'AvgPointsPerRace', 'TotalPointFinishes', 'TotalPodiums',
       'TotalPoints', 'hadPenaltyThisYear', 'PointsLast3Rounds',
       'DNFsLast3Rounds', 'FormRatio', 'Consistency', 'ProjectedGrowth',
       'ProjectedSeasonTotalPoints', 'RelativePointsShare',
       'CurrentRankAfterRound', 'PercentileRankAfterRound']

    previous_round_df = pd.read_csv(os.path.join(os.getcwd(), "templates/input_data_template.csv"))
    previous_round_df = previous_round_df[data_features]

    scenario_df = input_df.copy()

    round_to_predict = 17

    scenario_df.insert(loc=0, column="Year", value=datetime.now().year)
    scenario_df.insert(loc=1, column="Round", value=round_to_predict)

    random_grid_positions = np.arange(1, 21)
    random_grid_positions = random_grid_positions.tolist()
    random.shuffle(random_grid_positions)
    scenario_df.insert(loc=4, column="GridPosition", value=random_grid_positions)

    scenario_df["isPointsFinish"] = scenario_df.apply(is_points_finish, axis=1)
    scenario_df["isPodiumFinish"] = scenario_df.apply(is_podium_finish, axis=1)
    scenario_df["isDNF"] = scenario_df.apply(is_dnf, axis=1)

    scenario_df["Position"] = scenario_df["PredictedPoints"].rank(method="first", ascending=False).astype(float)

    new_scenario_df = pd.DataFrame(columns=columns)
    new_rows = []

    # Group last race data by team
    for team in teams:
        current_team_results = scenario_df.loc[scenario_df["TeamName"] == team].copy()
        previous_team_results = previous_round_df.loc[previous_round_df["TeamName"] == team].copy()

        row = []

        # Raw statistics
        year = current_team_results["Year"].iloc[0]
        team_name = current_team_results["TeamName"].iloc[0]
        round = current_team_results["Round"].iloc[0]
        rounds_completed = current_team_results["Round"].iloc[0] - 1
        rounds_remaining = 24 - current_team_results["Round"].iloc[0]
        dnfs_this_round = current_team_results["isDNF"].sum()
        points_earned_this_round = current_team_results["PredictedPoints"].sum()

        row.append(year)
        row.append(team_name)
        row.append(round)
        row.append(rounds_completed)
        row.append(rounds_remaining)
        row.append(dnfs_this_round)
        row.append(points_earned_this_round)
        
        # Raw driver statistics from the round
        max_driver_points = current_team_results["PredictedPoints"].max()
        min_driver_points = current_team_results["PredictedPoints"].min()
        driver_points_gap = max_driver_points - min_driver_points

        row.append(max_driver_points)
        row.append(min_driver_points)
        row.append(driver_points_gap)
        
        # Statistics that will be re-calculated over the course of the season
        avg_grid_position = current_team_results["GridPosition"].mean()
        avg_position = current_team_results["Position"].mean()
        dnf_rate = current_team_results["isDNF"].mean()
        avg_points_per_race = current_team_results["PredictedPoints"].mean()
        total_point_finishes = current_team_results["isPointsFinish"].sum()
        total_podiums = current_team_results["isPodiumFinish"].sum()
        total_points = current_team_results["PredictedPoints"].sum()
        had_penalty_this_year =  0

        # Re-calculating stats for "point-in-season" values
        avg_grid_position = statistics.mean([float(previous_team_results["AvgGridPosition"].iloc[0]), float(avg_grid_position)])
        avg_position = statistics.mean([float(previous_team_results["AvgPosition"].iloc[0]), float(avg_position)])
        dnf_rate = statistics.mean([float(previous_team_results["DNFRate"].iloc[0]), float(dnf_rate)])
        avg_points_per_race = statistics.mean([float(previous_team_results["AvgPointsPerRace"].iloc[0]), float(avg_points_per_race)])
        total_point_finishes = statistics.mean([float(previous_team_results["TotalPointFinishes"].iloc[0]), float(total_point_finishes)])
        total_podiums = statistics.mean([float(previous_team_results["TotalPodiums"].iloc[0]), float(total_podiums)])
        total_points = statistics.mean([float(previous_team_results["TotalPoints"].iloc[0]), float(total_points)])

        row.append(avg_grid_position)
        row.append(avg_position)
        row.append(dnf_rate)
        row.append(avg_points_per_race)
        row.append(total_point_finishes)
        row.append(total_podiums)
        row.append(total_points)
        row.append(had_penalty_this_year)
        
        # Consistency and form statistics
        points_last_three_rounds = previous_team_results["PointsEarnedThisRound"].iloc[0] + points_earned_this_round
        dnfs_last_three_rounds = previous_team_results["DNFsThisRound"].iloc[0] + dnfs_this_round
                    
        form_ratio = points_last_three_rounds / (avg_points_per_race * 3 + 1e-6)
                    
        rolling_mean_last_5_rounds = statistics.mean([float(previous_team_results["PointsEarnedThisRound"].iloc[0]), float(points_earned_this_round)])
        rolling_std_last_5_rounds = statistics.stdev([float(previous_team_results["PointsEarnedThisRound"].iloc[0]), float(points_earned_this_round)])
        consistency = 1 / (1 + (rolling_std_last_5_rounds / (rolling_mean_last_5_rounds + 1e-6)))        # Rescaling with a sigmoid to ensure values between 0 and 1
        
        # Projected growth statistics
        projected_growth = rolling_mean_last_5_rounds * rounds_remaining
        projected_season_total_points = total_points + projected_growth

        row.append(points_last_three_rounds)
        row.append(dnfs_last_three_rounds)
        row.append(form_ratio)
        row.append(consistency)
        row.append(projected_growth)
        row.append(projected_season_total_points)

        new_rows.append(row)
        
    for row in new_rows:
        new_scenario_df.loc[len(new_scenario_df)] = row
                        
    new_scenario_df["RelativePointsShare"] = new_scenario_df["TotalPoints"] / new_scenario_df.groupby(["Year", "Round"])["TotalPoints"].transform("sum")
    new_scenario_df["CurrentRankAfterRound"] = new_scenario_df.groupby(["Year", "Round"])["TotalPoints"].rank(method="dense", ascending=False).astype(int)
    new_scenario_df["PercentileRankAfterRound"] = 1.0 - (new_scenario_df["CurrentRankAfterRound"] - 1) / (new_scenario_df.groupby(["Year", "Round"])["TeamName"].transform('nunique') - 1)

    combined_df = pd.concat([previous_round_df, new_scenario_df], ignore_index=True)
    combined_df = combined_df.sort_values(by=["Round", "CurrentRankAfterRound"], ascending=[True, True]).reset_index(drop=True)

    return combined_df

if __name__ == "__main__":
    st.set_page_config(page_title="Scenario Simulator", page_icon="🎰", layout="wide")

    st.title("F1 Constructor's Championship Ranking Model Scenario Simulator")

    cwd = os.getcwd()

    scenario_df = pd.read_csv(os.path.join(cwd, "templates/scenario_template_data.csv"))
    scenario_df = scenario_df[["TeamName", "BroadcastName"]].sort_values(by="TeamName", ascending=True).reset_index(drop=True)
    scenario_df["PredictedPoints"] = 0
    scenario_df["DNF"] = False

    teams = scenario_df["TeamName"].unique().tolist()
    teams = sorted(teams)

    if 'scenario_df' not in st.session_state:
        st.session_state.scenario_df = scenario_df

    df = st.session_state.scenario_df

    st.subheader("Set Predicted Driver Outcomes")
    st.write("Current F1 Point System:")

    points_df = pd.DataFrame({
        "Postion": ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th"],
        "Points": [25, 18, 15, 12,10, 8, 6, 4, 2, 1]
    })

    st.dataframe(points_df, hide_index=True, width=250, height="auto")

    edited_df = st.data_editor(
        df,
        key="edits",
        hide_index=True,
        width="content",
        height=738,
        column_config={
            "PredictedPoints": st.column_config.NumberColumn("Predicted Points", min_value=0, max_value=25, step=1),
            "DNF": st.column_config.CheckboxColumn("DNF"),
        }
    )

    if st.session_state.edits["edited_rows"]:
        for row, edits in st.session_state.edits["edited_rows"].items():
            for col, value in edits.items():
                st.session_state.scenario_df.loc[row, col] = value
            if "DNF" in edits and edits["DNF"] == True:
                st.session_state.scenario_df.loc[row, "PredictedPoints"] = 0
        st.rerun()

    st.subheader("Predict Using the v3 Model Ensemble")
    if st.button("Predict", type="primary"):
        if not validate_f1_points(st.session_state.scenario_df):
            st.stop()

        input_df = generate_features(st.session_state.scenario_df, teams)

        st.dataframe(input_df, hide_index=True)
