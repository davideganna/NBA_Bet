# Libraries
import pandas as pd

# Functions
def setup(df):
    """ 
    Elo.setup() must be called once, when no teams have their Elo set. 
    An Elo rating of 1500 is assigned for each team. 
    Updating the Elo rating is done through Elo.update(). 
    Returns the updated DataFrame.
    """
    df["Elo"] = 1500
    return df

def update(df, away_team, home_team, winner):
    """
    Updates the Elo rating for team_A and team_B.
    Returns the updated DataFrame.
    """
    # Current Elo ratings for away_team and home_team
    elo_away_team = float(df.loc[df['Team'] == away_team, 'Elo'].values[0])
    elo_home_team = float(df.loc[df['Team'] == home_team, 'Elo'].values[0])
    # Expected Win probability for away_team and home_team
    exp_win_away_team = 1/(1+10**((elo_home_team - elo_away_team)/400))
    exp_win_home_team = 1/(1+10**((elo_away_team - elo_home_team)/400))
    # Define the K-Factor as K: the maximum possible adjustment per game.
    K = 16
    w = 1 if winner==away_team else 0
    elo_away_team_updated = elo_away_team + K*(w - exp_win_away_team)
    elo_home_team_updated = elo_home_team + K*((1-w) - exp_win_home_team)
    # Update the DataFrame
    df.loc[df['Team'] == away_team, 'Elo'] = elo_away_team_updated
    df.loc[df['Team'] == home_team, 'Elo'] = elo_home_team_updated
    return df

