import sys
import os
from typing import Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# External Libraries
import pandas as pd
import numpy as np
import logging, coloredlogs
from src.models.models import target, features
import src.dicts_and_lists as dal

import yaml

pd.options.mode.chained_assignment = None

# ------ Logger ------- #
logger = logging.getLogger("helper.py")
coloredlogs.install(level="DEBUG")

# Functions
def add_features_to_df(df: pd.DataFrame, season: str):
    # TODO put in pandas pipeline
    df["season"] = season
    # Log Ratio
    df["LogRatio_home"] = np.log2(df["PTS_home"] / df["PTS_away"])
    df["LogRatio_away"] = np.log2(df["PTS_away"] / df["PTS_home"])
    # Rebounds Ratio
    df["RB_aggr_home"] = df["ORB_home"] / df["DRB_away"]
    df["RB_aggr_away"] = df["ORB_away"] / df["DRB_home"]
    # eFG% - Effective Field Goal %
    df["eFG%_home"] = (df["FG_home"] + (0.5 * df["3P_home"])) / df["FGA_home"]
    df["eFG%_away"] = (df["FG_away"] + (0.5 * df["3P_away"])) / df["FGA_away"]
    # TS% - True Shooting %
    df["TS%_home"] = df["PTS_home"] / (2 * df["FGA_home"] + (0.44 * df["FTA_home"]))
    df["TS%_away"] = df["PTS_away"] / (2 * df["FGA_away"] + (0.44 * df["FTA_away"]))
    # Team at home won the match
    df = df.assign(HomeTeamWon=1)
    df.loc[(df["PTS_away"] > df["PTS_home"]), "HomeTeamWon"] = 0

    return df


def add_odds_to_split_df():
    with open("src/configs/main_conf.yaml") as f:
        config = yaml.safe_load(f)
    years = config["years"]
    odds_df = pd.read_csv(
        f"src/past_data/{years}/historical_odds-{years}.csv", sep=";", index_col=False
    )
    # Compute European Odds
    odds_df = odds_df.assign(
        Odds=1 + odds_df["ML"] / 100
    )  # Set the winner as the Home Team
    odds_df["Odds"].loc[odds_df["ML"] < 0] = 1 + 100 / (-odds_df["ML"])

    odds_df.drop(
        ["Rot", "1st", "2nd", "3rd", "4th", "Open", "Close", "ML", "2H"],
        axis=1,
        inplace=True,
    )

    odds_away = odds_df.loc[odds_df["VH"] == "V"]
    odds_home = odds_df.loc[odds_df["VH"] == "H"]

    dates_list = odds_away["Date"]
    dates_list = dates_list.values

    odds_away.drop(["Date", "VH"], axis=1, inplace=True)
    odds_home.drop(["Date", "VH"], axis=1, inplace=True)

    odds_away.rename(dal.historical_away, axis=1, inplace=True)
    odds_home.rename(dal.historical_home, axis=1, inplace=True)

    odds_away.reset_index(drop=True, inplace=True)
    odds_home.reset_index(drop=True, inplace=True)

    odds_df = pd.concat([odds_away, odds_home], axis=1)
    odds_df.insert(loc=0, column="Date", value=dates_list)

    # Read stats_per_game.csv
    split_stats_df = pd.read_csv(f"src/past_data/{years}/split_stats_per_game.csv")

    # Add column with odds
    split_stats_df = split_stats_df.assign(OddsAway=np.nan)
    split_stats_df = split_stats_df.assign(OddsHome=np.nan)
    split_copy = split_stats_df.copy()
    for n, row in split_copy.iterrows():
        # Extract day and month to compare both DataFrames
        date = row["Date"].partition(", ")[2]
        month = dal.months_dict[date[:3]]
        if month not in ["10", "11", "12"]:
            month = month[1]
        day = date.partition(",")[0][4:]
        if len(day) == 1:
            day = "0" + day

        # Home team won, Away team lost
        if row["Winner"] == 0:
            row["OddsHome"] = odds_df["Odds_home"].loc[
                (odds_df["Team_away"] == row["Team_away"])
                & (odds_df["Team_home"] == row["Team_home"])
                & (odds_df["Final_home"] == row["PTS_home"])
                & (odds_df["Final_away"] == row["PTS_away"])
                & (odds_df["Date"] == int(month + day))
            ]
            row["OddsAway"] = odds_df["Odds_away"].loc[
                (odds_df["Team_away"] == row["Team_away"])
                & (odds_df["Team_home"] == row["Team_home"])
                & (odds_df["Final_home"] == row["PTS_home"])
                & (odds_df["Final_away"] == row["PTS_away"])
                & (odds_df["Date"] == int(month + day))
            ]
        # Away team won, Home team lost
        else:
            row["OddsAway"] = odds_df["Odds_away"].loc[
                (odds_df["Team_away"] == row["Team_away"])
                & (odds_df["Team_home"] == row["Team_home"])
                & (odds_df["Final_home"] == row["PTS_home"])
                & (odds_df["Final_away"] == row["PTS_away"])
                & (odds_df["Date"] == int(month + day))
            ]
            row["OddsHome"] = odds_df["Odds_home"].loc[
                (odds_df["Team_away"] == row["Team_away"])
                & (odds_df["Team_home"] == row["Team_home"])
                & (odds_df["Final_home"] == row["PTS_home"])
                & (odds_df["Final_away"] == row["PTS_away"])
                & (odds_df["Date"] == int(month + day))
            ]
        logger.info(row["Date"])
        logger.info(row["Team_away"], row["Team_home"])
        split_stats_df["OddsHome"].iloc[n] = round(row["OddsHome"].values[0], 2)
        split_stats_df["OddsAway"].iloc[n] = round(row["OddsAway"].values[0], 2)

    split_stats_df.to_csv(
        f"src/past_data/{years}/split_stats_per_game.csv", index=False
    )


def build_merged_seasons():
    df_2017 = pd.read_csv("src/past_data/2017-2018/split_stats_per_game-2017.csv")
    df_2018 = pd.read_csv("src/past_data/2018-2019/split_stats_per_game-2018.csv")
    df_2019 = pd.read_csv("src/past_data/2019-2020/split_stats_per_game-2019.csv")
    df_2020 = pd.read_csv("src/past_data/2020-2021/split_stats_per_game.csv")

    df_2018 = df_2018.drop(["Date"], axis=1)
    merged_19 = pd.concat([df_2017, df_2018], axis=1)
    merged_20 = pd.concat([df_2017, df_2018, df_2019], axis=1)
    merged_21 = pd.concat([df_2017, df_2018, df_2019, df_2020], axis=1)

    merged_19.to_csv("src/past_data/merged_seasons/2017_to_2019_Stats.csv", index=False)
    merged_20.to_csv("src/past_data/merged_seasons/2017_to_2020_Stats.csv", index=False)
    merged_21.to_csv("src/past_data/merged_seasons/2017_to_2021_Stats.csv", index=False)


def build_season_df(folder, years):
    # TODO make it dynamic
    october_df = pd.read_csv(folder + "october_data.csv")
    november_df = pd.read_csv(folder + "november_data.csv")

    season_df = pd.concat([october_df, november_df])
    season_df.to_csv(f"{folder}{years}_season.csv", index=False)
