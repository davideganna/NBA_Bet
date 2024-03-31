import pandas as pd
from src.elo import get_elo_probs


def predict_on_elo(avg_df: pd.DataFrame, next_games: dict) -> dict:
    avg_df_last_away = avg_df.groupby("Team_away", as_index=False).last()
    avg_df_last_home = avg_df.groupby("Team_home", as_index=False).last()

    elo_away = avg_df_last_away.loc[
        avg_df_last_away["Team_away"].isin(next_games.keys()), "Elo_postgame_away"
    ]

    elo_home = avg_df_last_home.loc[
        avg_df_last_home["Team_home"].isin(next_games.values()), "Elo_postgame_home"
    ]

    away_team_to_elo = dict(zip(next_games.keys(), elo_away))
    home_team_to_elo = dict(zip(next_games.values(), elo_home))

    team_to_prob = dict()
    for away, home in zip(away_team_to_elo.items(), home_team_to_elo.items()):
        exp_win_away_team, exp_win_home_team = get_elo_probs(away[1], home[1])
        team_to_prob[away[0]] = exp_win_away_team
        team_to_prob[home[0]] = exp_win_home_team

    return team_to_prob
