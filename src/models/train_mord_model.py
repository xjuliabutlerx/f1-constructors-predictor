import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import date
from mord import LogisticAT
from rich import print
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

CLEAN_DATA_PATH = "../../data/clean/"
CM_FIG_PATH = "./cms/"
TRAINING_DATA_FILE = "training_data.csv"

if __name__ == "__main__":
    print()
    print("Starting model training...")

    # Read in the cleaned training data
    print(f" > Reading in the cleaned training data")
    training_data_df = pd.read_csv(os.path.join(CLEAN_DATA_PATH, TRAINING_DATA_FILE))

    # Define the feature columns and target variable
    print(f" > Extracting the feature columns and target variable")
    feature_columns = ["Round", "RoundsCompleted", "RoundsRemaining", "AvgGridPosition", "AvgPosition", "DNFRate", "AvgPointsPerRace", \
                        "TotalPointFinishes", "TotalPodiums", "TotalPoints", "hadPenaltyThisYear"]
    
    X = training_data_df[feature_columns]
    y = training_data_df["FinalRank"]

    # Split the data into training and testing sets
    print(f" > Splitting the data into training and testing sets")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24, shuffle=True, stratify=y)

    # Initialize and train the ordinal logistic regression model
    model = LogisticAT(alpha=1.0)

    # Traing the model
    print(f" > Training the ordinal logistic regression model")
    model.fit(X_train, y_train)

    # Make predictions on the test set
    print(f" > Making predictions on the test set")
    y_pred = model.predict(X_test)

    print(f" > Model training and prediction completed", end="\n\n")

    # Evaluate the model's performance
    print("Model Evaluation Metrics:")
    print(classification_report(y_test, y_pred))
    print()

    # Generate and display the confusion matrix
    print("Displaying Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=sorted(y.unique()),
                yticklabels=sorted(y.unique()))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(CM_FIG_PATH, f"mord_model_cm_{date.today().strftime('%Y-%m-%d %H:%M')}.png"))
    plt.show()