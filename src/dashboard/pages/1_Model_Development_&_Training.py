import streamlit as st
import os
import pandas as pd

cwd = os.getcwd()

st.set_page_config(page_title="Model Development & Training", page_icon="🔧", layout="wide")

st.title("🔧 Model Development & Training")

st.header("Project Development Timeline")

st.subheader("0. Curating the Dataset")
st.write("After finalizing the topic and scope of this project, I began exploring the `fastf1` data and library. Once I familiarized myself with this tool, I determined that my training dataset would ultimately contain data from the 2018 through the 2024 Formula 1 seasons. While I could have used even more data, the `fastf1` documentation stated that it only supported data starting in 2018. Anything older came from a deprecated library, `Ergast`, which has a slightly different format and information than `fastf1`. Additionally, the current points system began in 2010, so using older data may have led to unexpected model behavior.")

st.write("Once I had decided to use 2018 - 2024 as my training data, I began to download the raw race data. Since sprint races started in 2021 and contributed to the World Constructors' Championship, I also downloaded those sprint race results. I reviewed the data from each event and began to calculate some of the first data features for my models.")

st.subheader("1. v1 Model Development")
st.markdown('''
- Data Features (12): Year, Round, RoundsCompleted, RoundsRemaining, AvgGridPosition, AvgPosition, DNFRate, AvgPointsPerRace, TotalPointFinishes, TotalPodiums, TotalPoints, hadPenaltyThisYear
- Most Influential Features (according to SHAP analysis): AvgPointsPerRace, AvgPosition, AvgGridPosition, and RoundsRemaining
- Model Architecture: 3 fully-connected layers with ReLU (Rectified Linear Unit) activation function and dropout configured after the first 2 layers
''')

st.write("This, of course, was my first attempt at developing these ranking models. I started with the most basic data features that I could think of, as well as a very simple neural network. However, these features didn't truly capture the complexity of race results and performance. Statistics like averages are easily influenced, especially with a few stellar races. Additionally, I was using `RoundsRemaining` and `RoundsCompleted` which essentially captured the same temporal concept and introduced some multicollinearity into the model.")

st.write("Overall, due to the limited data features and marginal multicollinearity, the v1 models are all sensitive recent race performances and lack the accuracy required for the complex nature of predicting mid- and lower-ranking teams. Here is a summary of the v1 models' performance:")

v1_summary_df = pd.DataFrame({
        "v1 Model": ["Monaco", "Silverstone", "Suzuka", "Spa-Francorchamps", "Baku"],
        "Spearman's Rho": [0.9758, 0.9740, 0.9325, 0.9325, 0.9671],
        "Kendall's Tau": [0.92, 0.92, 0.83, 0.83, 0.90],
        "Mean Absolute Error": [0.29, 0.31, 0.66, 0.71, 0.77],
        "Max Error": [2.0, 2.0, 3.0, 4.0, 5.0]
    })

st.dataframe(v1_summary_df, hide_index=True, width="content")

st.write("Although the accuracy, represented by Spearman's Rho, seems relatively high at above 0.95 for each model, the heatmaps below display *where* the accuracy came from. For all the v1 models, you can see the precision taper off after the 4th or 5th positions and then increase towards the last 2 or 3 positions.")

with st.expander("Monaco v1 Heatmap"):
    st.image(os.path.join(cwd, "..", "models", "v1", "heatmaps", "Monaco Model Heatmap 2025-09-07.png"))

with st.expander("Silverstone v1 Heatmap"):
    st.image(os.path.join(cwd, "..", "models", "v1", "heatmaps", "Silverstone Model Heatmap 2025-09-06.png"))

with st.expander("Suzuka v1 Heatmap"):
    st.image(os.path.join(cwd, "..", "models", "v1", "heatmaps", "Suzuka Model Heatmap 2025-09-07.png"))

with st.expander("Spa-Francorchamps v1 Heatmap"):
    st.image(os.path.join(cwd, "..", "models", "v1", "heatmaps", "Spa-Francorchamps Model Heatmap 2025-09-07.png"))

with st.expander("Baku v1 Heatmap"):
    st.image(os.path.join(cwd, "..", "models", "v1", "heatmaps", "Baku Model Heatmap 2025-09-07.png"))

st.subheader("2. v2 Model Development")
st.markdown('''
- Data Features (16): Year, Round, RoundsRemaining, PointsEarnedThisRound, DNFsThisRound, PointsLast3Rounds, DNFsLast3Rounds, DNFRate, AvgGridPosition, AvgPosition, AvgPointsPerRace, TotalPointFinishes, TotalPodiums, TotalPoints, hadPenaltyThisYear, PercentileRankAfterRound
- Most Influential Features (according to SHAP analysis): TotalPoints, PointsLast3Rounds, and AvgPointsPerRace
- Model Architecture: 4 fully-connected layers only
''')

st.write("For my second go-around, I wanted to develop an ensemble of models that were resistent to recent changes and had more of a long-term focus. As a result, I began by developing some new features that may be indicators of a team's ranking, increasing the number of model inputs from 12 to 16. Many of these new features introducing rolling counts and basic interaction features. Additionally, I ditched the `RoundsCompleted` feature to get rid of that multicollinearity issue.")

st.write("To accomplish my goal of a more steady, long-term ranking model, I also tweaked the model architecture. I added another fully-connected layer and modified the input and output dimensions of the overall model to account for more complexity. Moreover, although a controversial decision, I opted to drop the activation function and random dropout. While these two components of model architecture have been known to reduce model overfitting, they also introduce nonlinearity and instability in ranking tasks. By forgoing these parts of the architecture, the v2 models were more deterministic and stable, ultimately accomplishing my goal for this ensemble.")

v2_summary_df = pd.DataFrame({
        "v2 Model": ["Monaco", "Silverstone", "Suzuka", "Spa-Francorchamps", "Baku"],
        "Spearman's Rho": [0.9760, 0.9621, 0.9550, 0.9463, 0.9760],
        "Kendall's Tau": [0.93, 0.87, 0.89, 0.87, 0.92],
        "Mean Absolute Error": [0.43, 0.43, 0.60, 0.51, 0.81],
        "Max Error": [3.0, 3.0, 3.0, 3.0, 4.0]
    })

st.dataframe(v2_summary_df, hide_index=True, width="content")

st.write("While the accuracy for some models decreased and error increased, it's important to note the more consistent accuracy across all positions in the heatmaps below.")

with st.expander("Monaco v2 Heatmap"):
    st.image(os.path.join(cwd, "..", "models", "v2", "heatmaps", "Monaco Model Heatmap 2025-09-09.png"))

with st.expander("Silverstone v2 Heatmap"):
    st.image(os.path.join(cwd, "..", "models", "v2", "heatmaps", "Silverstone Model Heatmap 2025-09-09.png"))

with st.expander("Suzuka v2 Heatmap"):
    st.image(os.path.join(cwd, "..", "models", "v2", "heatmaps", "Suzuka Model Heatmap 2025-09-09.png"))

with st.expander("Spa-Francorchamps v2 Heatmap"):
    st.image(os.path.join(cwd, "..", "models", "v2", "heatmaps", "Spa-Francorchamps Model Heatmap 2025-09-09.png"))

with st.expander("Baku v2 Heatmap"):
    st.image(os.path.join(cwd, "..", "models", "v2", "heatmaps", "Baku Model Heatmap 2025-09-09.png"))

st.subheader("3. v3 Model Development")
st.markdown('''
- Data Features (21): Year, Round, RoundsRemaining, PointsEarnedThisRound, DriverPointsGap, DNFsThisRound, PointsLast3Rounds, DNFsLast3Rounds, DNFRate, AvgGridPosition, AvgPosition, AvgPointsPerRace, TotalPointFinishes, FormRatio, Consistency, TotalPodiums, TotalPoints, hadPenaltyThisYear, ProjectedSeasonTotalPoints, RelativePointsShare, PercentileRankAfterRound
- Most Influential Features (according to SHAP analysis): ProjectedSeasonTotalPoints, AvgPointsPerRace, AvgPosition, and AvgGridPosition
- Model Architecture: 4 fully-connected layers with ReLU (Rectified Linear Unit) activation function and dropout configured after the first 3 layers
''')

st.write("On my third and final attempt, I knew I wanted a goldilocks model; a hybrid of the v1 and v2 models. Similar to my previous try, I began by developing my most complex input variables and interaction features to date with the parameters `DriverPointsGap`, `FormRatio`, `Consistency`, `RelativePointsShare`, and `ProjectedSeasonTotalPoints`. This brought the number of input variables from 16 to 21 from v2 to v3 and from 12 to 21 from v1 to v3. While the increase in number of features can help, I also crafted more indicative input variables than used in previous models. Additionally, I was careful to not to have features that caused data leakage by only using information available at the time of each round.")

st.write("As far as model architecture goes, I kept the 4 fully-connected layers and reintroduced an activation function and dropout. However, I did modify the input and output dimensions of the neural network to, again, account for the additional complexity of my new interaction features.")

st.write("In summary, I mainly used the model architecture from the v1 ensembles and built upon the features from the v2 ensembles to create the v3 models. Ultimately, I did accomplish my goal of stronger accuracy across all prediction positions, and especially the mid-field positions.")

v3_summary_df = pd.DataFrame({
        "v2 Model": ["Monaco", "Silverstone", "Suzuka", "Spa-Francorchamps", "Baku"],
        "Spearman's Rho": [0.9931, 0.9792, 0.9827, 0.9758, 0.9775],
        "Kendall's Tau": [0.98, 0.94, 0.95, 0.94, 0.93],
        "Mean Absolute Error": [0.17, 0.26, 0.23, 0.26, 0.49],
        "Max Error": [2.0, 2.0, 2.0, 3.0, 3.0]
    })

st.dataframe(v3_summary_df, hide_index=True, width="content")

st.write("The accuracies for the v3 models noticeably increased from v1 and v2, and their relatively consistent accuracies across all positions can be seen in the heatmaps below:")

with st.expander("Monaco v3 Heatmap"):
    st.image(os.path.join(cwd, "..", "models", "v3", "heatmaps", "Monaco Model Heatmap 2025-09-13.png"))

with st.expander("Silverstone v3 Heatmap"):
    st.image(os.path.join(cwd, "..", "models", "v3", "heatmaps", "Silverstone Model Heatmap 2025-09-13.png"))

with st.expander("Suzuka v3 Heatmap"):
    st.image(os.path.join(cwd, "..", "models", "v3", "heatmaps", "Suzuka Model Heatmap 2025-09-13.png"))

with st.expander("Spa-Francorchamps v3 Heatmap"):
    st.image(os.path.join(cwd, "..", "models", "v3", "heatmaps", "Spa-Francorchamps Model Heatmap 2025-09-13.png"))

with st.expander("Baku v3 Heatmap"):
    st.image(os.path.join(cwd, "..", "models", "v3", "heatmaps", "Baku Model Heatmap 2025-09-13.png"))

st.header("A Brief Overview of the Model Training & Testing Methdology")
st.write("While the model features and architecture varied across the 3 generations of models, the training methodology remained the same. The target y variable was always 'FinalRank', which was an integer value 1 through n, where n is the number of teams in the current Formula 1 season.")

st.write("Prior to training, I had the dataset of race data from 2018 through 2024 randomly split into a training dataset and testing/validation dataset. I used a random seed so I could replicate the randomness, if necessary. I typically used a test size of 20-25% while developing these models.")

st.write("During the model training, each epoch would randomly select 2 race results from the same year. I enforced sampling from the same year since it doesn't make sense to compare a 2018 Williams car to a 2024 Red Bull Racing car. Then, I added weights to ensure that approximately 60% of the random pairs it used came from mid-ranking teams (where y_test was 5-7 inclusive) to ensure the model had enough information to accurately predict these positions.")

st.write("Once the model had selected a bunch of race record pairs from the same year, it would learn to rank only these 2 records. It would train on this pair-wise comparison for an entire epoch before being set to an evaluation mode. Then, while validating, I would give it the test data year by year and have the model predict the overall team rankings for each year. As a result, the model would have 7 opportunities (number of years from 2018 to 2024) to rank each position 1 through n. This is why the heatmap values for each rank adds up to 7 for each position.")

st.write("After the model has attempted the rank all 7 years, a Spearman's rho and Kendall's tau was calculated for each epoch. These values, along with loss and learning rate were tracked and persisted into training data files I kept on my local machine.")

st.header("Why Neural Networks Instead of Ordinal Regression?")
st.write("While in the early phases of this project, I did consider using ordinal regression model architecture, namely from the Python package `mord`. It seemed like a good use case for these kinds of models given that the World Constructors' standings are naturally ordered 1st to 10th and points gap between teams may vary drastically. However, validating early versions of these models highlighted two key issues:")
st.markdown('''
1. **Ordinal regression assumes monotonicity.** In other words, ordinal regression assumes that the ranks change in a particular order, such as constant increase or decrease. However, this is unsuitable since some constructor teams may gain positions and then drop back in the championship throughout the season. This is especially evident for mid-field teams and championship contenders.
2. **Feature relationships are nonlinear and complex.** After the first model run, I noticed a need for more complex and higher-order data features, like rolling averages, standard deviations, and feature interactions. While ordinal regression can manage these inputs, the relationship between these data features can add even more complexity. For instance, both drivers in a team may do well in qualifying (`AvgGridPosition`), but one may not perform well during the race or DNF (`DNFRate`), creating a larger gap in points earned between the two drivers (`DriverPointsGap`) and ultimately causing the team to lose out on points.
''')

st.write("Additionally, while I was interested in learning about a new type of model architecture, I didn't have any prior experience with it. In school, I studied neural network architecture (as well as convolutional neural networks, long short term models, and transformers), so I ultimately transitioned to this model architecture and trained it as classification/regression hybrids to predict final ranks.")

st.header("Why is Spearman's Rho Used Over Traditional Accuracy?")
st.write("Traditional accuracy simply answers the question: *how often did the model predict the correct rank?* However, in ranking tasks, this metric can be misleading because although a model may predict one team to be 5th instead of 6th, this is penalized the same as predicting a team to be 5th instead of 10th. The traditional accuracy metric fails to capture how close the model is when it is incorrect.")

st.write("Therefore, Spearman's rho is a better measure of these models' accuracy because it measures correlation rather than straight accuracy. It looks at the overall predicted ordering of the teams compared to the true ordering. A high Spearman's rho means that the model's predicted ranking is close to the true ranking, even if the exact ranks are off.")

st.header("Why is Spearman's Rho Used Over Pearson's Rho?")
st.write("Pearson's rho (or Pearson's correlation coefficient) measures a linear relationship between two variables. Spearman's rho (or Spearman's rank correlation coefficient) measures monotonic relationship between two variables regardless of whether it's linear.")

st.write("In the context of this project, Pearson's rho would be used to measure how predicted points correlates with actual points. It's possible for the predicted points to be wildly incorrect, but still keep the relative ordering of the teams correct. Therefore, Pearson's rho could give a the models a lower score than it actually deserves.")

st.write("Meanwhile, Spearman's rho is similar, but converts the predicted points into a scale. In this case, the predicted points are converted into ranks 1 through n. Therefore, even if the predict points are wildly under- or over-estimating, the scaled ranks or orders are judged and an appropriate accuracy metric is reported.")

st.header("What is Kendall's Tau?")
st.write("Kendall's tau is commonly used to measure the accuracy of ranking models alongside Spearman's rho. Similar to the correlation coefficient, Kendall's tau measures the agreement between 2 pairs using a scaled measure, such as rank. It assigns a value of either -1 for complete disagreement or +1 for complete agreement.")

st.write("Since this metric looks at individual pairs rather than the full ordered ranking, Kendall's Tau is more sensitive to small ranking inaccurcies than Spearman's rho, and therefore, a harsher measure of how correct the model is.")