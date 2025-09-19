import pandas as pd
import streamlit as st

st.set_page_config(page_title="F1 Ranking Models", page_icon="🏎️", layout="wide")

st.title("Formula 1 Constructors' Championship Ranking Model")
st.subheader("A Fan Project")

st.header("About This Project")
st.write("This project contains 3 generations of machine learning ranking models used to predict Formula 1 Constructors' Championship standings.")

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
- Can be used **regardless** of how many teams are currently in Formula 1 (meaning, the models from this project can be used immediately for the 2026 season when there will be 11 teams)
""")

st.header("Why Not Predict the Results for the World Drivers' Championship?")
st.write("Creating a ranking model for the World Drivers' Championship is definitely possible and may be more engaging for fans. However, while determining the scope of this project, the challenge of predicting standings for the World Constructors' Championship seemed more interesting. Given that Formula 1 is a motorsport and an engineering competition, the best drivers don't necessarily win the World Drivers' Championship if they don't have the best car. Additionally, the skill gap between Formula 1 drivers is not as large as some people might think. The car—and the team—ultimately determines which driver will be the World Champion.")
st.write("That being said, if this project is popular and if the models from this project prove to be accurate, there may be a World Drivers' Championship ranking model project next. :smile:")

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
    <li>scipy</li>
    <li>matplotlib</li>
    <li>seaborn</li>
    """
)

st.html("For the full training and model architecture code, please see the GitHub repository <a href='https://github.com/xjuliabutlerx/f1-constructors-predictor' target='_blank'>f1-constructors-predictor</a>.")