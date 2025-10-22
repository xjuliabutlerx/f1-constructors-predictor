import streamlit as st
import os
import pandas as pd

st.set_page_config(page_title="Current Predictions", page_icon="📈", layout="wide")

st.title("📈 Current 2025 Constructors' Championship Ranking Predictions")

st.write("As of October 5, 2025, McLaren has won the 2025 World Constructors' Championship. However, the remaining 9 ranks are still yet to be determined. The following predictions were run with data collected for all the races in the 2025 season up through the United States Grand Prix in Austin, Texas.")

st.write("While the results from all 3 model ensembles are presented, the overall, most statistically accurate predictions come from the v3 generation. The projections from v2 and v1 should be used to provide the viewer with a more holistic picture of recent trends and long-term outlook to help guide any inferences.")

cwd = os.getcwd()

V1_PREDICTIONS = pd.read_csv(os.path.join("predictions", "2025-10-22 v1 ensemble predictions.csv"))
V2_PREDICTIONS = pd.read_csv(os.path.join("predictions", "2025-10-22 v2 ensemble predictions.csv"))
V3_PREDICTIONS = pd.read_csv(os.path.join("predictions", "2025-10-22 v3 ensemble predictions.csv"))

# V1_PREDICTIONS = pd.read_csv(os.path.join(cwd, "..", "..", "predictions", "2025-10-22 v1 ensemble predictions.csv"))
# V2_PREDICTIONS = pd.read_csv(os.path.join(cwd, "..", "..", "predictions", "2025-10-22 v2 ensemble predictions.csv"))
# V3_PREDICTIONS = pd.read_csv(os.path.join(cwd, "..", "..", "predictions", "2025-10-22 v3 ensemble predictions.csv"))

V1_PREDICTIONS = V1_PREDICTIONS[["Team", "Current Points", "Monaco Model", "Silverstone Model", "Suzuka Model", "Spa-Francorchamps Model", "Baku Model"]]
V2_PREDICTIONS = V2_PREDICTIONS[["Team", "Current Points", "Monaco Model v2", "Silverstone Model v2", "Suzuka Model v2", "Spa-Francorchamps Model v2", "Baku Model v2"]]
V3_PREDICTIONS = V3_PREDICTIONS[["Team", "Current Points", "Monaco Model v3", "Silverstone Model v3", "Suzuka Model v3", "Spa-Francorchamps Model v3", "Baku Model v3"]]

st.header("v3 Model Ensemble Predictions")
st.subheader("*The Best of Both Worlds - balancing recency and season-long consistency*")
st.dataframe(V3_PREDICTIONS, width="content", hide_index=True)
st.markdown('''
Key Points:
- Despite the poor race results from Mercedes, the models are not suggesting that any team could take 2nd place from them.
- However, there will be a true battle for 3rd between Red Bull Racing and Ferrari, given the former's recent surge in race wins, podiums, and overall points earnings.
- The Monaco Model, the most accurate of this ensemble, predicts Red Bull Racing will come out on top of Ferrari but just behind Mercedes.
''')

st.header("v2 Model Ensemble Predictions")
st.subheader("*Slow and Steady Wins the Race - focused on long-term performances than short-term swings*")
st.dataframe(V2_PREDICTIONS, width="content", hide_index=True)
st.markdown('''
Key Points:
- Since Mercedes had a tepid performance at the US Grand Prix, the models show an indication that Ferrari or even Red Bull Racing may being able to steal 2nd from them.
- The gap for 3rd is already small, so the models are displaying disagreement over which team will earn this spot. The general prediction maintains that Ferrari stays just ahead of Red Bull Racing.
- As the season closes in the final 5 races, there is also movement predicted for the mid-field teams: Aston Martin and Racing Bulls fighting over 6th and Kick Sauber fending off the Haas F1 Team for 8th.
''')

st.header("v1 Model Ensemble Predictions")
st.subheader("*Fast to React - sensitive to recent races*")
st.dataframe(V1_PREDICTIONS, width="content", hide_index=True)
st.markdown('''
Key Points:
- Given Mercedes' declining performance, these models present a strong indication that they will lose out on 2nd to either Red Bull Racing or Ferrari. The likely suggestion is Red Bull could end up in 2nd, Mercedes in 3rd, and Ferrari in 4th.
- Midfield battles continue. While a majority of the models are in agreement that Williams will hold onto their 5th place, some are suggesting a sudden drop in performance, given recent race results, leaving the door open for Racing Bulls.
- The v1 models are not as accurate in predicting lower ranked teams, but the general consensus is that Kick Sauber, Haas F1 Team, and Alpine are on the bottom.
''')