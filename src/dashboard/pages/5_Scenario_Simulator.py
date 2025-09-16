from datetime import datetime

import numpy as np
import os
import pandas as pd
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

def generate_features(scenario_df:pd.DataFrame, teams:list):
    # previous_round_df = pd.read_csv(os.path.join(os.getcwd(), "../templates/input_data_template.csv"))

    print(scenario_df)

    round_to_predict = 17

    if "Year" not in scenario_df.columns:
        scenario_df.insert(loc=0, column="Year", value=datetime.now().year)

    if "Round" not in scenario_df.columns:
        scenario_df.insert(loc=1, column="Round", value=round_to_predict)

    if "GridPosition" not in scenario_df.columns:
        random_grid_positions = np.arange(1, 21)
        scenario_df.insert(loc=4, column="GridPosition", value=random_grid_positions.tolist())

    scenario_df["isPointsFinish"] = scenario_df.apply(is_points_finish, axis=1)
    scenario_df["isPodiumFinish"] = scenario_df.apply(is_podium_finish, axis=1)
    scenario_df["isDNF"] = scenario_df.apply(is_dnf, axis=1)

    scenario_df["Position"] = scenario_df["PredictedPoints"].rank(method="first", ascending=False).astype(float)

    new_scenario_df = pd.DataFrame()

    for team in teams:
        temp_df = pd.DataFrame()
        current_team_results = scenario_df.loc[scenario_df["TeamName"] == team]

        # Raw statistics
        temp_df["Year"] = current_team_results["Year"]
        temp_df["TeamName"] = current_team_results["TeamName"]
        temp_df["Round"] = current_team_results["Round"]
        temp_df["RoundsCompleted"] = current_team_results["Round"] - 1
        temp_df["RoundsRemaining"] = 24 - current_team_results["Round"]
        temp_df["DNFsThisRound"] = current_team_results["isDNF"].sum()
        temp_df["PointsEarnedThisRound"] = current_team_results["PredictedPoints"].sum()
        
        # Raw driver statistics from the round
        temp_df["MaxDriverPoints"] = current_team_results["PredictedPoints"].max()
        temp_df["MinDriverPoints"] = current_team_results["PredictedPoints"].min()
        temp_df["DriverPointsGap"] = temp_df["MaxDriverPoints"] - temp_df["MinDriverPoints"]
        
        # Statistics that will be re-calculated over the course of the season
        temp_df["AvgGridPosition"] = current_team_results["GridPosition"].expanding().mean().tail(1)
        temp_df["AvgPosition"] = current_team_results["Position"].expanding().mean().tail(1)
        temp_df["DNFRate"] = current_team_results["isDNF"].expanding().mean().tail(1)
        temp_df["AvgPointsPerRace"] = current_team_results["PredictedPoints"].expanding().mean().tail(1)
        temp_df["TotalPointFinishes"] = current_team_results["isPointsFinish"].cumsum()
        temp_df["TotalPodiums"] = current_team_results["isPodiumFinish"].cumsum()
        temp_df["TotalPoints"] = current_team_results["PredictedPoints"].cumsum()
        temp_df["hadPenaltyThisYear"] =  0

        # Re-calculating stats for "point-in-season" values
        temp_df["AvgGridPosition"] = temp_df["AvgGridPosition"].expanding().mean()
        temp_df["AvgPosition"] = temp_df["AvgPosition"].expanding().mean()
        temp_df["DNFRate"] = temp_df["DNFRate"].expanding().mean()
        temp_df["AvgPointsPerRace"] = temp_df["AvgPointsPerRace"].expanding().mean()
        temp_df["TotalPointFinishes"] = temp_df["TotalPointFinishes"].cumsum()
        temp_df["TotalPodiums"] = temp_df["TotalPodiums"].cumsum()
        temp_df["TotalPoints"] = temp_df["TotalPoints"].cumsum()
        
        # Consistency and form statistics
        temp_df["PointsLast3Rounds"] = temp_df.groupby(["Year", "TeamId"])["PointsEarnedThisRound"] \
                                                                            .rolling(window=3, min_periods=1).sum().reset_index(level=[0, 1], drop=True)
        temp_df["DNFsLast3Rounds"] = temp_df.groupby(["Year", "TeamId"])["DNFsThisRound"] \
                                                                            .rolling(window=3, min_periods=1).sum().reset_index(level=[0, 1], drop=True)
                    
        temp_df["FormRatio"] = temp_df["PointsLast3Rounds"] / (temp_df["AvgPointsPerRace"] * 3 + 1e-6)
                    
        rolling_mean_last_5_rounds = temp_df.groupby(["Year", "TeamId"])["PointsEarnedThisRound"] \
                                            .rolling(window=5, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
        rolling_std_last_5_rounds = temp_df.groupby(["Year", "TeamId"])["PointsEarnedThisRound"] \
                                            .rolling(window=5, min_periods=1).std().fillna(0).reset_index(level=[0, 1], drop=True)
        temp_df["Consistency"] = 1 / (1 + (rolling_std_last_5_rounds / (rolling_mean_last_5_rounds + 1e-6)))        # Rescaling with a sigmoid to ensure values between 0 and 1
        
        # Projected growth statistics
        temp_df["ProjectedGrowth"] = rolling_mean_last_5_rounds * temp_df["RoundsRemaining"]
        temp_df["ProjectedSeasonTotalPoints"] = temp_df["TotalPoints"] + temp_df["ProjectedGrowth"]
        
        temp_df = temp_df.dropna(how="any").reset_index(drop=True)
                        
        new_scenario_df = pd.concat([new_scenario_df, temp_df], ignore_index=True)

    print(new_scenario_df)
    return new_scenario_df

if __name__ == "__main__":
    st.set_page_config(page_title="Scenario Simulator", page_icon="🎰", layout="wide")

    st.write("# F1 Constructor's Championship Ranking Model Scenario Simulator")

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
    edited_df = st.data_editor(
        df,
        key="edits",
        hide_index=True,
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

    if st.button("Predict", type="primary"):
        if not validate_f1_points(st.session_state.scenario_df):
            st.stop()

        input_df = generate_features(st.session_state.scenario_df, teams)
