import streamlit as st
import os
import pandas as pd

st.set_page_config(page_title="Current Predictions", page_icon="📈", layout="wide")

st.title("📈 Current 2025 Constructors' Championship Ranking Predictions")

st.write("As of October 5, 2025, McLaren has won the 2025 World Constructors' Championship. However, the remaining 9 ranks are still yet to be determined. The following predictions were run with data collected for all the races in the 2025 season up through the Singapore Grand Prix.")

st.write("While the results from all 3 model ensembles are presented, the overall, most accurate predictions come from the v3 generation. The projections from v2 and v1 should be used to provide the viewer with a more holistic picture of recent trends and long-term outlook to help guide any inferences.")

cwd = os.getcwd()

V1_PREDICTIONS = pd.read_csv(os.path.join("predictions", "2025-10-06 v1 ensemble predictions.csv"))
V2_PREDICTIONS = pd.read_csv(os.path.join("predictions", "2025-10-06 v2 ensemble predictions.csv"))
V3_PREDICTIONS = pd.read_csv(os.path.join("predictions", "2025-10-06 v3 ensemble predictions.csv"))

# V1_PREDICTIONS = pd.read_csv(os.path.join(cwd, "..", "..", "predictions", "2025-10-06 v1 ensemble predictions.csv"))
# V2_PREDICTIONS = pd.read_csv(os.path.join(cwd, "..", "..", "predictions", "2025-10-06 v2 ensemble predictions.csv"))
# V3_PREDICTIONS = pd.read_csv(os.path.join(cwd, "..", "..", "predictions", "2025-10-06 v3 ensemble predictions.csv"))

V1_PREDICTIONS = V1_PREDICTIONS[["Team", "Current Points", "Monaco Model", "Silverstone Model", "Suzuka Model", "Spa-Francorchamps Model", "Baku Model"]]
V2_PREDICTIONS = V2_PREDICTIONS[["Team", "Current Points", "Monaco Model v2", "Silverstone Model v2", "Suzuka Model v2", "Spa-Francorchamps Model v2", "Baku Model v2"]]
V3_PREDICTIONS = V3_PREDICTIONS[["Team", "Current Points", "Monaco Model v3", "Silverstone Model v3", "Suzuka Model v3", "Spa-Francorchamps Model v3", "Baku Model v3"]]

st.header("v3 Model Ensemble Predictions")
st.subheader("*The Best of Both Worlds - balancing recency and season-long consistency*")
st.dataframe(V3_PREDICTIONS, width="content", hide_index=True)
st.markdown('''
Key Points:
- Since there are only 6 races remaining, the models are predicting less movement from the current positions.
- Of course, given the tie between Aston Martin and Racing Bulls, the models present some disagreement over who will take 6th place. However, Racing Bulls seem to have an edge given their season-long performance.
''')

st.header("v2 Model Ensemble Predictions")
st.subheader("*Slow and Steady Wins the Race - focused on long-term performances than short-term swings*")
st.dataframe(V2_PREDICTIONS, width="content", hide_index=True)
st.markdown('''
Key Points:
- After McLaren, Mercedes is strongly predicted to take 2nd place with only one model suggesting Ferrari could make a comeback.
- Since the gap for 3rd is so small, the models are displaying disagreement over which team will earn this spot. Given Max Verstappen / Red Bull Racing's recent string of podiums and wins and Ferrari's decline in performance, Red Bull Racing is favored to triumph.
- As seen in other ensemble predictions, there is a battle for 6th, but Racing Bulls seems to have an edge over Aston Martin and other teams given their overall performance this season.
- Finally, since Kick Sauber's form has been declining and Haas F1 Team's form has been steadily rising, the models are suggesting there is a chance for the Haas F1 Team to steal 8th place from Kick Sauber.
''')

st.header("v1 Model Ensemble Predictions")
st.subheader("*Fast to React - sensitive to recent races*")
st.dataframe(V1_PREDICTIONS, width="content", hide_index=True)
st.markdown('''
Key Points:
- Mercedes is generally predicted to take 2nd place, but two of the less accurate models suggest Ferrari could make a comeback since the gap isn't too large.
- Despite what the Tifosi-biased models say, a majority of the models think Red Bull Racing could even top Ferrari for 3rd.
- Midfield battles are heating up. While a majority of the models are in agreement that Williams will hold onto their 5th place, some are suggesting a sudden drop in performance, given recent race results, leaving the door open for other teams.
- The v1 models are not as accurate in predicting lower ranked teams, but the general consensus is that Kick Sauber, Haas F1 Team, and Alpine are on the bottom.
''')