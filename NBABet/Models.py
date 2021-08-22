from sklearn import tree
import Helper
import pandas as pd
pd.options.mode.chained_assignment = None
from sklearn.preprocessing import OrdinalEncoder
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


# Create the df containing stats per single game on every row
df = pd.read_csv('past_data/average_seasons/average_N_3Seasons.csv')
#df = pd.read_csv('past_data/merged_seasons/2017_to_2020_Stats.csv')

# Define the target
target = 'Winner'

# Only consider the features with abs(CorrelationValue) > 0.3
away_features = [
    'FG_home',
    'FG%_home',
    '3P%_home',
    'FT%_home',
    #'ORB_home',
    #'DRB_home',
    #'TRB_home',
    #'AST_home',
    #'STL_home',
    #'TOV_home',
    'PTS_home',
    'LogRatio_home',
    #'RB_aggr_home',
    'eFG%_home',
    'TS%_home'
]
home_features = [
    'FG_away',
    'FG%_away',
    '3P%_away',
    'FT%_away',
    #'ORB_away',
    #'DRB_away',
    #'TRB_away',
    #'AST_away',
    #'STL_away',
    #'TOV_away',
    'PTS_away',
    'LogRatio_away',
    #'RB_aggr_away',
    'eFG%_away',
    'TS%_away',
    'HomeTeamWon'
]

features = away_features + home_features


############### Different functions for building different models. ###############

def build_AdaBoost_classifier():
    """
    Builds an AdaBoost Classifier.
    """
    X_train, y_train = df[features], df[[target]]
    ada_clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=1), n_estimators=125,
        algorithm='SAMME.R', learning_rate=0.5
    )
    ada_clf.fit(X_train, y_train.values.ravel())
    return ada_clf

def build_DT_classifier():
    """
    Builds a Decision Tree Classifier.
    """
    # Split into train and test
    X_train, y_train = df[features], df[[target]]
    tree_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    tree_clf.fit(X_train, y_train)
    return tree_clf 

def build_RF_classifier():
    """
    Builds a Random Forest Classifier.
    """
    # Split into train and test
    X_train, y_train = df[features], df[[target]]
    # Define a Random Forest Classifier
    rf_clf = RandomForestClassifier(
        n_estimators=200, max_leaf_nodes=16, n_jobs=-1, random_state=42
    )
    rf_clf.fit(X_train, y_train.values.ravel())
    return rf_clf

def build_SVM_classifier():
    """
    Builds a Support Vector Machine Classifier.
    """
    X_train, y_train = df[features], df[[target]]
    # Normalize the data
    svm_clf = make_pipeline(StandardScaler(), SVC())

    # Define a SVM Classifier
    svm_clf.fit(X_train, y_train.values.ravel())

    return svm_clf

