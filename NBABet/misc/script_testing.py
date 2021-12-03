import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import Helper
import Elo
import dicts_and_lists as dal
import logging, coloredlogs
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pd.set_option('display.max_rows', 1000)

# ------ Logger ------- #
logger = logging.getLogger('script_testing.py')
coloredlogs.install(level='INFO', logger=logger)



# ---------- TESTS ---------- #
# Define the target
target = 'Winner'

# Only consider the features with abs(CorrelationValue) > 0.3
away_features = [
    'FG_home',
    'FG%_home',
    '3P%_home',
    'FT%_home',
    'ORB_home',
    'DRB_home',
    'TRB_home',
    'AST_home',
    'STL_home',
    'TOV_home',
    'PTS_home',
    'LogRatio_home',
    'RB_aggr_home',
    'eFG%_home',
    'TS%_home'
]
home_features = [
    'FG_away',
    'FG%_away',
    '3P%_away',
    'FT%_away',
    'ORB_away',
    'DRB_away',
    'TRB_away',
    'AST_away',
    'STL_away',
    'TOV_away',
    'PTS_away',
    'LogRatio_away',
    'RB_aggr_away',
    'eFG%_away',
    'TS%_away',
    'HomeTeamWon'
]

features = away_features + home_features
df = pd.read_csv('../past_data/average_seasons/average_N_4Seasons.csv')

x = df.loc[:, features].values
y = df.loc[:, [target]].values

x = StandardScaler().fit_transform(x)

std_df = pd.DataFrame(x, columns=features)
std_df = pd.concat([std_df, df['Winner'].reset_index(drop=True)], axis=1)

