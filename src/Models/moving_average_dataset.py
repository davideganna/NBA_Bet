import sys
import os

from pandas.core.frame import DataFrame

from src.helper import add_features_to_df

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import pandas as pd
from models.models import away_features, home_features, features
import models.backtesting as backtesting
import src.dicts_and_lists as dal
import logging, coloredlogs

pd.set_option("display.max_rows", 1000)

# ------ Logger ------- #
logger = logging.getLogger("build_moving_average_model.py")
coloredlogs.install(level="INFO", logger=logger)

# To evaluate accuracy
dates_list = []
predictions = []
true_values = []
model_prob = []
model_odds = []
odds_winner = []
odds_loser = []
home_teams_list = []
away_teams_list = []
winners_list = []


def extract_and_insert(
    next_game,
    _df: DataFrame,
    average_N,
    evaluated_indexes,
    to_insert_list,
    winners_list,
):
    """
    Based on the next game, the function computes the average of N the previous games played by
        the same team and inserts the values in "averageN_season.csv".
    """
    # Extract away_team Name and home_team Name from last_N_games_away and last_N_games_home
    away_team = next_game["Team_away"].values[0]
    home_team = next_game["Team_home"].values[0]

    # Before predicting a game, check that it has not yet been predicted.
    # This is the case where e.g., TeamHome's next game at home against TeamAway has been evaluated ...
    # by both next home game and next away game. They are the same game, which are therefore predicted twice.
    if next_game.index[0] not in evaluated_indexes:
        # Track the inserted game based on its index
        evaluated_indexes.append(next_game.index[0])

        # Extract indexes for last N games
        next_games_away_indexes = _df.loc[_df["Team_away"] == away_team].index
        next_games_home_indexes = _df.loc[_df["Team_home"] == home_team].index
        next_away_indexes_reduced = [
            x for x in next_games_away_indexes if x < next_game.index[0]
        ][-average_N:]
        next_home_indexes_reduced = [
            x for x in next_games_home_indexes if x < next_game.index[0]
        ][-average_N:]

        # Extract last N games based on indexes
        last_N_games_away = _df.iloc[next_away_indexes_reduced]
        last_N_games_home = _df.iloc[next_home_indexes_reduced]

        # Concatenate the two teams with their average stats
        to_insert = pd.concat(
            [
                round(last_N_games_away[away_features].mean(), 5),
                round(last_N_games_home[home_features].mean(), 5),
            ],
            axis=0,
        )

        to_insert_list.append(to_insert)
        winners_list.append(
            _df["Winner"].loc[_df.index == next_game.index[0]].values[0]
        )


def build_moving_average_dataset(average_N, skip_n, leave_out=None):
    df_2017 = pd.read_csv("src/past_data/2017-2018/split_stats_per_game.csv")
    df_2018 = pd.read_csv("src/past_data/2018-2019/split_stats_per_game.csv")
    df_2019 = pd.read_csv("src/past_data/2019-2020/split_stats_per_game.csv")
    df_2020 = pd.read_csv("src/past_data/2020-2021/split_stats_per_game.csv")
    df_2021 = pd.read_csv("src/past_data/2021-2022/split_stats_per_game.csv")

    df_2021 = add_features_to_df(df_2021)

    logger.info(
        f"Averaging the datasets. MA: {average_N} games, first {skip_n} games are skipped."
    )

    # FIXME Improve using config
    if leave_out == "2019":
        df_list = [df_2017, df_2018, df_2020, df_2021]
    elif leave_out == "2020":
        df_list = [df_2017, df_2018, df_2019, df_2021]
    else:
        df_list = [df_2017, df_2018, df_2019, df_2020, df_2021]

    for _df in df_list:
        # Cleanup at every iteration
        evaluated_indexes = []
        to_insert_list = []
        winners_list = []
        for skip_n_games in range(skip_n, 50 - average_N):
            last_N_games_away, last_N_games_home = backtesting.get_first_N_games(
                _df, average_N, skip_n_games
            )
            # Get next game based on next_game_index
            for team in dal.teams:
                # Find all games where "team" plays away
                next_games_away_indexes = _df.loc[_df["Team_away"] == team].index
                last_away_game = last_N_games_away[dal.teams_to_int[team]][-1:]
                # Check if there are more games past the current index
                try:
                    dal.last_home_away_index_dict[team][0] = last_away_game.index[0]
                except:
                    pass
                if (
                    max(next_games_away_indexes)
                    != dal.last_home_away_index_dict[team][0]
                ):
                    next_game_index = min(
                        i
                        for i in next_games_away_indexes[skip_n + average_N :]
                        if i > last_away_game.index
                    )
                    next_game = _df.loc[_df.index == next_game_index]

                    next_games_home_indexes = _df.loc[
                        _df["Team_home"] == next_game["Team_home"].values[0]
                    ].index

                    if next_game_index in next_games_home_indexes[skip_n + average_N :]:
                        extract_and_insert(
                            next_game,
                            _df,
                            average_N,
                            evaluated_indexes,
                            to_insert_list,
                            winners_list,
                        )

                # Find all games where "team" plays home
                next_games_home_indexes = _df.loc[_df["Team_home"] == team].index
                last_home_game = last_N_games_home[dal.teams_to_int[team]][-1:]
                # Check if there are more games past the current index
                try:
                    dal.last_home_away_index_dict[team][1] = last_home_game.index[0]
                except:
                    pass
                if (
                    max(next_games_home_indexes)
                    != dal.last_home_away_index_dict[team][1]
                ):
                    next_game_index = min(
                        i
                        for i in next_games_home_indexes[skip_n + average_N :]
                        if i > last_home_game.index
                    )
                    next_game = _df.loc[_df.index == next_game_index]

                    next_games_away_indexes = _df.loc[
                        _df["Team_away"] == next_game["Team_away"].values[0]
                    ].index

                    if next_game_index in next_games_away_indexes[skip_n + average_N :]:
                        extract_and_insert(
                            next_game,
                            _df,
                            average_N,
                            evaluated_indexes,
                            to_insert_list,
                            winners_list,
                        )

        avg_df = pd.concat(to_insert_list, axis=1).transpose()
        avg_df["Winner"] = winners_list

        if _df is df_2017:
            _df.name = "2017/2018 Season DataFrame"
            avg_df.to_csv("src/past_data/average_seasons/average2017.csv", index=False)
            logger.info(f"Retrieved stats for {_df.name}")
        elif _df is df_2018:
            _df.name = "2018/2019 Season DataFrame"
            avg_df.to_csv("src/past_data/average_seasons/average2018.csv", index=False)
            logger.info(f"Retrieved stats for {_df.name}")
        if leave_out != "2019":
            if _df is df_2019:
                _df.name = "2019/2020 Season DataFrame"
                avg_df.to_csv(
                    "src/past_data/average_seasons/average2019.csv", index=False
                )
                logger.info(f"Retrieved stats for {_df.name}")
        if leave_out != "2020":
            if _df is df_2020:
                _df.name = "2020/2021 Season DataFrame"
                avg_df.to_csv(
                    "src/past_data/average_seasons/average2020.csv", index=False
                )
                logger.info(f"Retrieved stats for {_df.name}")
        elif _df is df_2021:
            _df.name = "2021/2022 Season DataFrame"
            avg_df.to_csv("src/past_data/average_seasons/average2021.csv", index=False)
            logger.info(f"Retrieved stats for {_df.name}")

    # Concatenate the 3 average season datasets
    avg_2017_df = pd.read_csv("src/past_data/average_seasons/average2017.csv")
    avg_2018_df = pd.read_csv("src/past_data/average_seasons/average2018.csv")
    if leave_out != "2019":
        avg_2019_df = pd.read_csv("src/past_data/average_seasons/average2019.csv")
    if leave_out != "2020":
        avg_2020_df = pd.read_csv("src/past_data/average_seasons/average2020.csv")
    avg_2021_df = pd.read_csv("src/past_data/average_seasons/average2021.csv")

    if leave_out == "2019":
        avg_total_df = pd.concat(
            [avg_2017_df, avg_2018_df, avg_2020_df, avg_2021_df], axis=0
        )
    elif leave_out == "2020":
        avg_total_df = pd.concat(
            [avg_2017_df, avg_2018_df, avg_2019_df, avg_2021_df], axis=0
        )
    else:
        avg_total_df = pd.concat(
            [avg_2017_df, avg_2018_df, avg_2019_df, avg_2020_df, avg_2021_df], axis=0
        )

    if leave_out == "2019" or leave_out == "2020":
        avg_total_df.to_csv(
            "src/past_data/average_seasons/average_N_4Seasons.csv", index=False
        )
        logger.info("Average N 4 Seasons has been updated.")
    else:
        avg_total_df.to_csv(
            "src/past_data/average_seasons/average_NSeasons_prod.csv", index=False
        )
        logger.info("Average N Seasons Prod has been updated.")
