import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

pd.options.mode.chained_assignment = None

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


############### Different functions for building different models. ###############


def build_AdaBoost_classifier(df: DataFrame):
    """
    Builds an AdaBoost Classifier.
    """
    X_train, y_train = df[features], df[[target]]
    ada_clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=1),
        n_estimators=125,
        algorithm="SAMME.R",
        learning_rate=0.5,
    )
    ada_clf.fit(X_train, y_train.values.ravel())
    return ada_clf


def build_DT_classifier(df: DataFrame):
    """
    Builds a Decision Tree Classifier.
    """
    # Split into train and test
    X_train, y_train = df[features], df[[target]]
    tree_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    tree_clf.fit(X_train, y_train)
    return tree_clf


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


def build_XGBoostClassifier(df: DataFrame):
    """
    Builds an Extreme Gradient Boosting Classifier.
    """
    # Split into train and test
    X_train, y_train = df[features], df[[target]]
    # Define a Random Forest Classifier
    xgbc = XGBClassifier(n_estimators=50)
    xgbc.fit(X_train, y_train.values.ravel())
    return xgbc


def build_SVM_classifier(df: DataFrame):
    """
    Builds a Support Vector Machine Classifier.
    """
    X_train, y_train = df[features], df[[target]]
    # Normalize the data
    svm_clf = make_pipeline(StandardScaler(), SVC())

    # Define a SVM Classifier
    svm_clf.fit(X_train, y_train.values.ravel())

    return svm_clf
