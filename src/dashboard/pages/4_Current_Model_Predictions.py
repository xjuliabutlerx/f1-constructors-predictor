import streamlit as st
import os
import pandas as pd

st.set_page_config(page_title="Current Predictions", page_icon="📈", layout="wide")

st.title("📈 Current 2025 Constructors' Championship Ranking Predictions")

st.write("As of September 21, 2025, these are the current 2025 F1 Constructor's Championship ranking predictions of my models. These predictions were run with data collected for all the races in the 2025 season up to the Azerbaijan Grand Prix in Baku.")

st.write("While the results from all 3 model ensembles are presented, the overall, most accurate predictions come from the v3 generation. The projections from v2 and v1 should be used to provide the viewer with a more holistic picture of recent trends and long-term outlook to help guide any inferences.")

cwd = os.getcwd()

V1_PREDICTIONS = pd.read_csv(os.path.join("predictions", "2025-09-21 v1 ensemble predictions.csv"))
V2_PREDICTIONS = pd.read_csv(os.path.join("predictions", "2025-09-21 v2 ensemble predictions.csv"))
V3_PREDICTIONS = pd.read_csv(os.path.join("predictions", "2025-09-21 v3 ensemble predictions.csv"))

# V1_PREDICTIONS = pd.read_csv(os.path.join(cwd, "..", "..", "predictions", "2025-09-21 v1 ensemble predictions.csv"))
# V2_PREDICTIONS = pd.read_csv(os.path.join(cwd, "..", "..", "predictions", "2025-09-21 v2 ensemble predictions.csv"))
# V3_PREDICTIONS = pd.read_csv(os.path.join(cwd, "..", "..", "predictions", "2025-09-21 v3 ensemble predictions.csv"))

V1_PREDICTIONS = V1_PREDICTIONS[["Team", "Current Points", "Monaco Model", "Silverstone Model", "Suzuka Model", "Spa-Francorchamps Model", "Baku Model"]]
V2_PREDICTIONS = V2_PREDICTIONS[["Team", "Current Points", "Monaco Model v2", "Silverstone Model v2", "Suzuka Model v2", "Spa-Francorchamps Model v2", "Baku Model v2"]]
V3_PREDICTIONS = V3_PREDICTIONS[["Team", "Current Points", "Monaco Model v3", "Silverstone Model v3", "Suzuka Model v3", "Spa-Francorchamps Model v3", "Baku Model v3"]]

st.header("v3 Model Ensemble Predictions")
st.subheader("*The Best of Both Worlds - balancing recency and season-long consistency*")
st.dataframe(V3_PREDICTIONS, width="content", hide_index=True)
st.markdown('''
Key Points:
- McLaren is still firmly in the lead despite the recent stream of wins by Max Verstappen / Red Bull Racing and steady point gains by Ferrari and Mercedes.
- Given the latest momentum by Mercedes, they have taken a narrow lead over Ferrari and are generally predicted to maintain this position.
- Due to Racing Bulls' excellent qualifying and race performance in Baku, they have surged past Aston Martin for 6th position.
''')

st.header("v2 Model Ensemble Predictions")
st.subheader("*Slow and Steady Wins the Race - focused on long-term performances than short-term swings*")
st.dataframe(V2_PREDICTIONS, width="content", hide_index=True)
st.markdown('''
Key Points:
- There will be a fight for 2nd due to the recent victories by Max Verstappen / Red Bull Racing and improvement in form by Mercedes. Although Ferrari has built up a lead throughout the season, their recent performances haven't been able to match Red Bull Racing or Mercedes, which may cost them the position. The front-runners for 2nd position are Ferrari and Mercedes.
- The gap between Racing Bulls and Aston Martin is still narrow, but the models generally agree that Racing Bulls will triumph.
- Finally, since Kick Sauber's form has been declining and Haas F1 Team's form has been steadily rising, the models are suggesting there is a chance for the Haas F1 Team to steal 8th place from Kick Sauber.
''')

st.header("v1 Model Ensemble Predictions")
st.subheader("*Fast to React - sensitive to recent races*")
st.dataframe(V1_PREDICTIONS, width="content", hide_index=True)
st.markdown('''
Key Points:
- McLaren is still firmly in the lead despite recent successes by Max Verstappen / Red Bull Racing and steady point gains by Ferrari and Mercedes.
- Despite the recent race and updated Constructors' Championship standings, the models think that Ferrari will take back 2nd place and that Red Bull Racing will also have a significant opportunity to even take 3rd place from Mercedes.
- As the season enters the final 7 races, there will be significant changes for the last 5 teams, especially given how small the gaps in points are. In general, Williams and Aston Martin sit comfortably in 5th and 7th positions respectively. Either Kick Sauber or Racing Bulls would take 6th. Finally, this ensemble of models believe the last three teams will be Haas F1 Team, either Kick Sauber or Racing Bulls, and Alpine in that order.
''')