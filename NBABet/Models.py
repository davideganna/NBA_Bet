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

# Define the target
target = 'Winner'

# Only consider the features with abs(CorrelationValue) > 0.3
away_features = [
    'RB_aggr_away',
    'PTS_away',
    'FG%_away',    
    'FG_away',     
    'DRB_away',    
    '3P%_away',    
    'TRB_away',    
    'LogRatio_away',
    'eFG%_away',
    'TS%_away'
]
home_features = [ 
    'RB_aggr_home',
    '3P%_home',   
    'DRB_home',   
    'FG_home',    
    'FG%_home',   
    'PTS_home',
    'LogRatio_home',
    'eFG%_home',
    'TS%_home'
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

def get_odds_Elo(away_team, home_team):
    """
    Get probabilities based on the Team's Elo value.
    """
    elo_df = pd.read_csv('elo.csv')
    elo_away_team = float(elo_df.loc[elo_df['Team'] == away_team, 'Elo'].values[0])
    elo_home_team = float(elo_df.loc[elo_df['Team'] == home_team, 'Elo'].values[0])

    # Expected Win probability for away_team and home_team
    odds_away_team = 1+10**((elo_home_team - elo_away_team)/400)
    odds_home_team = 1+10**((elo_away_team - elo_home_team)/400)

    return odds_away_team, odds_home_team