# Libraries
import pandas as pd
from pandas.core.frame import DataFrame
import dicts_and_lists as dal

# Functions
def update(row):
    """
    Updates the Elo rating for team_A and team_B.
    Returns the updated DataFrame.
    """
    away_team = row['Team_away']
    home_team = row['Team_home']
    winner = row['Winner']
    # Current Elo ratings for away_team and home_team
    elo_away_team = dal.current_team_Elo[away_team]
    elo_home_team = dal.current_team_Elo[home_team]
    
    # Expected Win probability for away_team and home_team
    exp_win_away_team = 1/(1+10**((elo_home_team - elo_away_team)/400))
    exp_win_home_team = 1/(1+10**((elo_away_team - elo_home_team)/400))
    
    # Define the K-Factor as K: the maximum possible adjustment per game.
    if (abs(row['PTS_home'] - row['PTS_away']) > 15):
        K = 30
    elif (abs(row['PTS_home'] - row['PTS_away']) > 9):
        K = 15
    elif (abs(row['PTS_home'] - row['PTS_away']) > 5):
        K = 7
    else:
        K = 0
    elo_away_team_updated = elo_away_team + K*(winner - exp_win_away_team)
    elo_home_team_updated = elo_home_team + K*((1-winner) - exp_win_home_team)
    
    # Update the Dictionary
    dal.current_team_Elo[away_team] = elo_away_team_updated
    dal.current_team_Elo[home_team] = elo_home_team_updated
    
    # Update the DataFrame
    row['Elo_away'] = elo_away_team_updated
    row['Elo_home'] = elo_home_team_updated
    
    return row

def get_probas(away_team, home_team):
    # Current Elo ratings for away_team and home_team
    elo_away_team = dal.current_team_Elo[away_team]
    elo_home_team = dal.current_team_Elo[home_team]
    # Expected Win probability for away_team and home_team
    prob_away_wins = 1/(1+10**((elo_home_team - elo_away_team)/400))
    prob_home_wins = 1/(1+10**((elo_away_team - elo_home_team)/400))
    
    return prob_away_wins, prob_home_wins

def get_odds(away_team, home_team):
    """
    Get probabilities based on the Team's Elo value.
    """
    prob_away_wins, prob_home_wins = get_probas(away_team, home_team)
    odds_away_wins, odds_home_wins = 1/prob_away_wins, 1/prob_home_wins

    return odds_away_wins, odds_home_wins