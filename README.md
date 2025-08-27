# F1 Constructors Predictor

## About the Project

This project applies machine learning to forecast the final rankings of Formula 1 constructors throughout a season. Using historical race, driver, and team performance data, the model leverages an ordinal regression approach to capture the ordered nature of championship standings. The goal is not only to evaluate predictive accuracy, but also to explore which features—such as qualifying performance, reliability, or prior season results—most strongly influence a team’s success.

## The Plan

I will be using the `fastf1` Python library to gather historical data, the `mord` Python library for ordinal logistic regression, the `torch` library for a classification model, and Streamlit to create an interactive dashboard for users to experiment with different scenarios and view predictions in real time.