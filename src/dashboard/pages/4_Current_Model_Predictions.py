import streamlit as st
import os
import pandas as pd

st.set_page_config(page_title="Current Predictions", page_icon="📈", layout="wide")

st.title("📈 Current 2026 Constructors' Championship Ranking Predictions")

st.write("As of September 11, 2026, Mercedes holds a commanding 122-point lead atop the 2026 World Constructors' Championship after 13 of 23 rounds, though the title is not yet mathematically decided and the remaining 10 ranks are still very much in flux. The following predictions were run with data collected for all races in the 2026 season up through the Italian Grand Prix in Monza.")

st.write("While the results from all 3 model ensembles are presented, the overall, most statistically accurate predictions come from the v3 generation. The projections from v2 and v1 should be used to provide the viewer with a more holistic picture of recent trends and long-term outlook to help guide any inferences.")

cwd = os.getcwd()

V1_PREDICTIONS = pd.read_csv(os.path.join("predictions", "2026-09-11_21:55_v1_ensemble_predictions.csv"))
V2_PREDICTIONS = pd.read_csv(os.path.join("predictions", "2026-09-11_21:55_v2_ensemble_predictions.csv"))
V3_PREDICTIONS = pd.read_csv(os.path.join("predictions", "2026-09-11_21:55_v3_ensemble_predictions.csv"))

V1_PREDICTIONS = V1_PREDICTIONS[["Team", "Current Points", "Monaco Model", "Silverstone Model", "Suzuka Model", "Spa-Francorchamps Model", "Baku Model"]]
V2_PREDICTIONS = V2_PREDICTIONS[["Team", "Current Points", "Monaco Model v2", "Silverstone Model v2", "Suzuka Model v2", "Spa-Francorchamps Model v2", "Baku Model v2"]]
V3_PREDICTIONS = V3_PREDICTIONS[["Team", "Current Points", "Monaco Model v3", "Silverstone Model v3", "Suzuka Model v3", "Spa-Francorchamps Model v3", "Baku Model v3"]]

st.header("v3 Model Ensemble Predictions")
st.subheader("*The Best of Both Worlds - balancing recency and season-long consistency*")
st.dataframe(V3_PREDICTIONS, width="content", hide_index=True)
st.markdown('''
Key Points:
- All five v3 models unanimously agree on the top 6, matching the current standings exactly: Mercedes, Ferrari, McLaren, Red Bull Racing, Racing Bulls, and Alpine.
- Haas F1 Team is unanimously projected to hold 7th as well. The real intrigue is 8th, where the Monaco Model is the lone dissenter, projecting Williams to overtake Audi despite trailing by 5 points.
- Most strikingly, every v3 model projects winless Cadillac to leapfrog Aston Martin for 10th, despite currently sitting 3 points behind them.
''')

st.header("v2 Model Ensemble Predictions")
st.subheader("*Slow and Steady Wins the Race - focused on long-term performances than short-term swings*")
st.dataframe(V2_PREDICTIONS, width="content", hide_index=True)
st.markdown('''
Key Points:
- Four of five v2 models keep the top 3 unchanged, but the Baku Model breaks from the pack, projecting Ferrari to close the 122-point gap and overtake Mercedes for the championship lead.
- The Spa-Francorchamps Model disagrees over 2nd, projecting McLaren to leapfrog Ferrari despite Ferrari's 59-point cushion over them.
- Unlike the v3 ensemble, v2 does not see Cadillac closing the gap on Aston Martin — every v2 model keeps Cadillac locked into last place.
''')

st.header("v1 Model Ensemble Predictions")
st.subheader("*Fast to React - sensitive to recent races*")
st.dataframe(V1_PREDICTIONS, width="content", hide_index=True)
st.markdown('''
Key Points:
- All five v1 models unanimously agree on the top 5, matching the current standings: Mercedes, Ferrari, McLaren, Red Bull Racing, and Racing Bulls.
- Alpine mostly holds 6th, but the Suzuka Model is a striking outlier, projecting Aston Martin to jump from 10th all the way into 6th — a sign of the volatility v1 is known for.
- The back of the grid (Audi, Haas F1 Team, Williams, Aston Martin, and Cadillac) appears in a different order under nearly every model, reinforcing that v1 is the least reliable ensemble for the bottom of the standings, consistent with its recency-sensitive design.
''')