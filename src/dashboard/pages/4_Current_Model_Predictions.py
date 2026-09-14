import streamlit as st
import os
import pandas as pd

st.set_page_config(page_title="Current Predictions", page_icon="📈", layout="wide")

st.title("📈 Current 2026 Constructors' Championship Ranking Predictions")

st.write("As of September 13, 2026, Mercedes has extended its lead atop the 2026 World Constructors' Championship to 145 points over Ferrari after 14 of 23 rounds, though the title is not yet mathematically decided and the remaining 9 ranks are still very much in flux. The following predictions were run with data collected for all races in the 2026 season up through the Spanish Grand Prix in Madrid, the newest circuit added to the calendar this season.")

st.write("While the results from all 3 model ensembles are presented, the overall, most statistically accurate predictions come from the v3 generation. The projections from v2 and v1 should be used to provide the viewer with a more holistic picture of recent trends and long-term outlook to help guide any inferences.")

cwd = os.getcwd()

V1_PREDICTIONS = pd.read_csv(os.path.join("predictions", "2026-09-13_20:49_v1_ensemble_predictions.csv"))
V2_PREDICTIONS = pd.read_csv(os.path.join("predictions", "2026-09-13_20:49_v2_ensemble_predictions.csv"))
V3_PREDICTIONS = pd.read_csv(os.path.join("predictions", "2026-09-13_20:49_v3_ensemble_predictions.csv"))

V1_PREDICTIONS = V1_PREDICTIONS[["Team", "Current Points", "Monaco Model", "Silverstone Model", "Suzuka Model", "Spa-Francorchamps Model", "Baku Model"]]
V2_PREDICTIONS = V2_PREDICTIONS[["Team", "Current Points", "Monaco Model v2", "Silverstone Model v2", "Suzuka Model v2", "Spa-Francorchamps Model v2", "Baku Model v2"]]
V3_PREDICTIONS = V3_PREDICTIONS[["Team", "Current Points", "Monaco Model v3", "Silverstone Model v3", "Suzuka Model v3", "Spa-Francorchamps Model v3", "Baku Model v3"]]

st.header("v3 Model Ensemble Predictions")
st.subheader("*The Best of Both Worlds - balancing recency and season-long consistency*")
st.dataframe(V3_PREDICTIONS, width="content", hide_index=True)
st.markdown('''
Key Points:
- All five v3 models now unanimously agree on the top 9, exactly matching the current standings — up from 6 unanimous positions the week before, including a newly settled agreement that Audi holds 8th over Williams.
- Positions 10 and 11 remain the exception: every v3 model still projects winless Cadillac to leapfrog Aston Martin, a call that's held steady since Monza despite Cadillac's continued scoreless streak.
''')

st.header("v2 Model Ensemble Predictions")
st.subheader("*Slow and Steady Wins the Race - focused on long-term performances than short-term swings*")
st.dataframe(V2_PREDICTIONS, width="content", hide_index=True)
st.markdown('''
Key Points:
- Three of five v2 models keep the top 6 unchanged, but the Baku Model is now the biggest outlier yet, reshuffling the entire podium to Ferrari-McLaren-Mercedes even as Mercedes' actual lead grew to 145 points.
- The Spa-Francorchamps Model also disagrees, swapping Ferrari and McLaren for 2nd/3rd and Racing Bulls and Alpine for 5th/6th.
- Unlike v3's unanimous call, v2 remains mostly unconvinced Cadillac will pass Aston Martin — only the Silverstone Model sees that swap, while the other four keep Aston Martin in 10th.
''')

st.header("v1 Model Ensemble Predictions")
st.subheader("*Fast to React - sensitive to recent races*")
st.dataframe(V1_PREDICTIONS, width="content", hide_index=True)
st.markdown('''
Key Points:
- All five v1 models continue to unanimously agree on the top 5, matching the current standings: Mercedes, Ferrari, McLaren, Red Bull Racing, and Racing Bulls.
- The Suzuka Model remains the outlier at 6th, again projecting Aston Martin to jump from 10th into 6th despite Aston Martin trailing Alpine by 65 points — the same anomaly seen after Monza, suggesting it's a persistent quirk of that model rather than a one-off.
- The back of the grid (Audi, Haas F1 Team, Williams, Aston Martin, and Cadillac) still appears in a different order under nearly every model, reinforcing that v1 remains the least reliable ensemble for the bottom of the standings, consistent with its recency-sensitive design.
''')