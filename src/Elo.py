# Libraries
import pandas as pd
from pandas.core.frame import DataFrame
import src.dicts_and_lists as dal

# Functions
def update_row(row, teams_seen: list):
    """
    Used in backtesting. Updates the Elo rating for team_A and team_B for a single row.
    Returns the updated row.
    """
    away_team = row["Team_away"]
    home_team = row["Team_home"]
    winner = 0 if row["PTS_home"] > row["PTS_away"] else 1
    
    # Current Elo ratings for away_team and home_team
    elo_away_team = dal.current_team_Elo[away_team]
    elo_home_team = dal.current_team_Elo[home_team]

    # Expected Win probability for away_team and home_team
    exp_win_away_team = 1 / (1 + 10 ** ((elo_home_team - elo_away_team) / 400))
    exp_win_home_team = 1 / (1 + 10 ** ((elo_away_team - elo_home_team) / 400))

    # Define the K-Factor as K: the maximum possible adjustment per game.
    if abs(row["PTS_home"] - row["PTS_away"]) > 15:
        K = 30
    elif abs(row["PTS_home"] - row["PTS_away"]) > 9:
        K = 15
    elif abs(row["PTS_home"] - row["PTS_away"]) > 5:
        K = 7
    else:
        K = 0
    elo_away_team_updated = elo_away_team + K * (winner - exp_win_away_team)
    elo_home_team_updated = elo_home_team + K * ((1 - winner) - exp_win_home_team)

    # If this is the first time the algorithm sees the team, fix its Elo to 1500 and return
    if away_team in teams_seen:
        dal.current_team_Elo[away_team] = elo_away_team_updated
        row["Elo_pregame_away"] = elo_away_team_updated
    else:
        row["Elo_pregame_away"] = 1500
        teams_seen.append(away_team)

    if home_team in teams_seen:
        dal.current_team_Elo[home_team] = elo_home_team_updated
        row["Elo_pregame_home"] = elo_home_team_updated
    else:
        row["Elo_pregame_home"] = 1500
        teams_seen.append(home_team)

    return row, teams_seen


def update_DataFrame(
    elo_df: DataFrame, away_team, home_team, away_pts, home_pts, winner
):
    """
    Updates the Elo rating for team_A and team_B.
    Returns the updated DataFrame.
    """
    # Current Elo ratings for away_team and home_team
    elo_away_team = float(elo_df.loc[elo_df["Team"] == away_team, "Elo"].values[0])
    elo_home_team = float(elo_df.loc[elo_df["Team"] == home_team, "Elo"].values[0])
    # Expected Win probability for away_team and home_team
    exp_win_away_team = 1 / (1 + 10 ** ((elo_home_team - elo_away_team) / 400))
    exp_win_home_team = 1 / (1 + 10 ** ((elo_away_team - elo_home_team) / 400))
    # Define the K-Factor as K: the maximum possible adjustment per game.
    if abs(home_pts - away_pts) > 15:
        K = 30
    elif abs(home_pts - away_pts) > 9:
        K = 15
    elif abs(home_pts - away_pts) > 5:
        K = 7
    else:
        K = 0
    elo_away_team_updated = elo_away_team + K * (winner - exp_win_away_team)
    elo_home_team_updated = elo_home_team + K * ((1 - winner) - exp_win_home_team)
    # Update the DataFrame
    elo_df.loc[elo_df["Team"] == away_team, "Elo"] = elo_away_team_updated
    elo_df.loc[elo_df["Team"] == home_team, "Elo"] = elo_home_team_updated
    return elo_df


def add_elo_to_df(folder):
    """
    Iteratively adds the Elo for each match.
    Saves the updated DataFrame as a .csv file.
    """
    df = pd.read_csv(f"{folder}split_stats_per_game.csv")
    df["Elo_pregame_away"] = None
    df["Elo_pregame_home"] = None
    # Define a list of teams already seen: on the first occurrence Elo doesn't need to be calculated 
    teams_seen = []
    for ix, row in df.iterrows():
        df.iloc[ix], teams_seen = update_row(row, teams_seen)

    df.to_csv(f"{folder}split_stats_per_game.csv", index=False)


def get_probas(away_team, home_team, years):
    # Current Elo ratings for away_team and home_team
    elo_df = pd.read_csv("src/past_data/" + years + "/elo.csv")
    elo_away_team = elo_df.loc[elo_df["Team"] == away_team, "Elo"].values[0]
    elo_home_team = elo_df.loc[elo_df["Team"] == home_team, "Elo"].values[0]
    # Expected Win probability for away_team and home_team
    prob_away_wins = 1 / (1 + 10 ** ((elo_home_team - elo_away_team) / 400))
    prob_home_wins = 1 / (1 + 10 ** ((elo_away_team - elo_home_team) / 400))

    return prob_away_wins, prob_home_wins


def get_odds(away_team, home_team, years):
    """
    Get probabilities based on the Team's Elo value.
    """
    prob_away_wins, prob_home_wins = get_probas(away_team, home_team, years)
    odds_away_wins, odds_home_wins = 1 / prob_away_wins, 1 / prob_home_wins

    return odds_away_wins, odds_home_wins
