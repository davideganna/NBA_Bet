# ----------------------------------- backtesting.py ----------------------------------- #
# Allows the user to backtest a model, by iterating over the 2020_2021 Season dataset.
# -------------------------------------------------------------------------------------- #
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import dicts_and_lists as dal
import pandas as pd

def get_first_N_games(df:pd.DataFrame, n:int, skip:int):
    """
    For each team, get the first N home games and away games they have played.
    Parameter "skip" represents an offset which is increased at each iteration. Example below:
    To get the first 5 games played: Set n = 5. 
    To skip the first 100 games played: Set skip = 100. 
    """
    home_list = []
    away_list = []
    for team in dal.teams:
        away_list.append(df.loc[df['Team_away'] == team][skip:n+skip])
        home_list.append(df.loc[df['Team_home'] == team][skip:n+skip])

    return away_list, home_list

