import pandas as pd
import streamlit as st

st.set_page_config(page_title="F1 Ranking Models", page_icon="🏎️", layout="wide")

st.title("F1 Constructor's Championship Ranking Model Project")

st.header("About This Project")
st.write("This project contains 3 generations of machine learning ranking models used to predict Formula 1 Constructor's Championship standings.")

st.markdown("""
What these models can do:
- Help explain which teams are on the rise or decline based on recent form, driver consistency, and reliability
- Use past races to predict how teams will perform across the season
- Show how standings could evolve as the season continues
""")

st.markdown("""
The results of this project:
- Highlight how important reliability and consistency are
- Lets fans and analysts compare different prediction strategies
- Can be used regardless of how many teams are current in Formula 1 (meaning, the models from this project can be used immediately for the 2026 season when there will be 11 teams)
""")

st.header("The Model Generations - In a Nutshell")

st.markdown("""
- v1: a set of models that react strongly to recent races and performances
- v2: a set of models that focuses more on historical and long-term performance rather than recent events
- v3: a set of models that balances the approaches of v1 and v2 models, balancing consistency with recent form
""")

st.header("Tech Stack")
st.html(
    """
    <li>Python 3.11.5</li>
    <li><a href='https://github.com/theOehrly/Fast-F1' target='_blank'>fastf1</a></li>
    <li>torch (PyTorch)</li>
    <li>pandas</li>
    <li>numpy</li>
    """
)

st.html("For the full training and model architecture code, please see the GitHub repository <a href='https://github.com/xjuliabutlerx/f1-constructors-predictor' target='_blank'>f1-constructors-predictor</a>.")