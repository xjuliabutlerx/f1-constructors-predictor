import streamlit as st
import os
import pandas as pd
import numpy as np

st.set_page_config(page_title="Current Predictions", page_icon="📈", layout="wide")

st.title("📈 Current 2025 Constructors' Championship Ranking Predictions")

st.write("As of September 13, 2025, these are the current 2025 F1 Constructor's Championship ranking predictions of my models. These predictions were run with data collected for all the races in the 2025 season up to the Italian Grand Prix in Monza.")

cwd = os.getcwd()

V1_PREDICTIONS = pd.read_csv(os.path.join("predictions", "2025-09-13 v1 ensemble predictions.csv"))
V2_PREDICTIONS = pd.read_csv(os.path.join("predictions", "2025-09-13 v2 ensemble predictions.csv"))
V3_PREDICTIONS = pd.read_csv(os.path.join("predictions", "2025-09-13 v3 ensemble predictions.csv"))

# V1_PREDICTIONS = pd.read_csv(os.path.join(cwd, "../../predictions/2025-09-13 v1 ensemble predictions.csv"))
# V2_PREDICTIONS = pd.read_csv(os.path.join(cwd, "../../predictions/2025-09-13 v2 ensemble predictions.csv"))
# V3_PREDICTIONS = pd.read_csv(os.path.join(cwd, "../../predictions/2025-09-13 v3 ensemble predictions.csv"))

V1_PREDICTIONS = V1_PREDICTIONS[["Team", "Current Points", "Monaco Model", "Silverstone Model", "Suzuka Model", "Spa-Francorchamps Model", "Baku Model"]]
V2_PREDICTIONS = V2_PREDICTIONS[["Team", "Current Points", "Monaco Model v2", "Silverstone Model v2", "Suzuka Model v2", "Spa-Francorchamps Model v2", "Baku Model v2"]]
V3_PREDICTIONS = V3_PREDICTIONS[["Team", "Current Points", "Monaco Model v3", "Silverstone Model v3", "Suzuka Model v3", "Spa-Francorchamps Model v3", "Baku Model v3"]]

st.header("v3 Model Ensemble Predictions")
st.subheader("*The Best of Both Worlds - balancing recency and season-long consistency*")
st.dataframe(V3_PREDICTIONS, width="content", hide_index=True)
st.markdown('''
Key Points:
- McLaren is firmly in the lead despite recent successes by Max Verstappen / Red Bull Racing and steady point gains by Ferrari and Mercedes.
- Due to Isack Hadjar's / Racing Bulls' recent podium and Aston Martin's slim 1-point lead, there will definitely be a battle for 6th.
- The only other change this point in the season could be between the Kick Sauber and Haas F1 Team for 8th.
''')

st.header("v2 Model Ensemble Predictions")
st.subheader("*Slow and Steady Wins the Race - focused on long-term performances than short-term swings*")
st.dataframe(V2_PREDICTIONS, width="content", hide_index=True)
st.markdown('''
Key Points:
- The point gaps between the top 4 teams are too large (20+) to cause much change at this stage in the 2025 season.
- There may be some movement in the mid- and low-ranking positions between Aston Martin and Racing Bulls and even between Kick Sauber and Haas F1 Team.
- Unless McLaren doesn't lock down the top spot in the World Constructor's Championship, there *could* be an opportunity for Ferrari to win—but this is unlikely given the statistics this year.
''')

st.header("v1 Model Ensemble Predictions")
st.subheader("*Fast to React - sensitive to recent races*")
st.dataframe(V1_PREDICTIONS, width="content", hide_index=True)
st.markdown('''
Key Points:
- McLaren is firmly in the lead despite recent successes by Max Verstappen / Red Bull Racing and steady point gains by Ferrari and Mercedes.
- There is likely going to be a fight for 3rd between Mercedes and Red Bull Racing in the future, despite a 21-point gap. Although the v1 models are in disagreement, seems like Mercedes might come out on top.
- Due to Isack Hadjar's / Racing Bulls' recent podium and Aston Martin's slim 1-point lead, there will definitely be a battle for 6th.
''')