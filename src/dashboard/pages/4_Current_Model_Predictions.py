import streamlit as st
import os
import pandas as pd
import numpy as np

st.set_page_config(page_title="Current Predictions", page_icon="📈", layout="wide")

st.write("# Current Constructor's Championship Ranking Predictions")

st.write("As of September 13, 2025, these are the current 2025 F1 Constructor's Championship ranking predictions of my models. These predictions were run with data collected for all the races in the 2025 season up to the Italian Grand Prix in Monza.")

cwd = os.getcwd()

V1_PREDICTIONS = pd.read_csv(os.path.join(cwd, "../../predictions/2025-09-13 v1 ensemble predictions.csv"))
V2_PREDICTIONS = pd.read_csv(os.path.join(cwd, "../../predictions/2025-09-13 v2 ensemble predictions.csv"))
V3_PREDICTIONS = pd.read_csv(os.path.join(cwd, "../../predictions/2025-09-13 v3 ensemble predictions.csv"))

st.write("### v3 Model Ensemble Predictions")
st.dataframe(V3_PREDICTIONS, width="content", hide_index=True)

st.write("### v2 Model Ensemble Predictions")
st.dataframe(V2_PREDICTIONS, width="content", hide_index=True)

st.write("### v1 Model Ensemble Predictions")
st.dataframe(V1_PREDICTIONS, width="content", hide_index=True)