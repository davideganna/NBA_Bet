import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.ensemble import RandomForestClassifier

pd.options.mode.chained_assignment = None

# TODO write a class
def build_RF_classifier(df: DataFrame):
    """
    Builds a Random Forest Classifier.
    """
    # Split into train and test
    X_train, y_train = df[features], df[[target]]
    # Define a Random Forest Classifier
    rf_clf = RandomForestClassifier(
        n_estimators=500,
        n_jobs=-1,
        random_state=42,
    )
    rf_clf.fit(X_train, y_train.values.ravel())
    return rf_clf


# Define the target
target = "Winner"

# Only consider the features with abs(CorrelationValue) > 0.3
home_features = [
    "FG_home",
    "FG%_home",
    "3P%_home",
    "FT%_home",
    "ORB_home",
    "DRB_home",
    "TRB_home",
    "AST_home",
    "STL_home",
    "TOV_home",
    "PTS_home",
    "LogRatio_home",
    "RB_aggr_home",
    "eFG%_home",
    "TS%_home",
]

away_features = [
    "FG_away",
    "FG%_away",
    "3P%_away",
    "FT%_away",
    "ORB_away",
    "DRB_away",
    "TRB_away",
    "AST_away",
    "STL_away",
    "TOV_away",
    "PTS_away",
    "LogRatio_away",
    "RB_aggr_away",
    "eFG%_away",
    "TS%_away",
    "HomeTeamWon",
]

features = away_features + home_features
