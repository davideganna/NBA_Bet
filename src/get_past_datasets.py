# python src/get_past_datasets.py

# External Libraries
from datetime import date
import pandas as pd

pd.options.mode.chained_assignment = None
import os
from pathlib import Path
import logging, coloredlogs

# Internal Libraries
import dicts_and_lists as dal
import Helper as Helper
import yaml
import time
import elo

from ETL import DataTransformer

# ------ Logger ------- #
logger = logging.getLogger("get_past_datasets.py")
coloredlogs.install(level="DEBUG")

with open("src/configs/main_conf.yaml") as f:
    config = yaml.safe_load(f)

years = config["years"]
folder = f"src/past_data/{years}/"
season = config["season"]

""" months = [
    "october",
    "november",
    "december",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
]

for month in months:
    url = (
        f"https://www.basketball-reference.com/leagues/NBA_{season}_games-{month}.html"
    )
    try:
        df_url = pd.read_html(url)[0]
        df_url = df_url.rename(
            columns={
                "Visitor/Neutral": "AwayTeam",
                "Home/Neutral": "HomeTeam",
                "PTS": "AwayPoints",
                "PTS.1": "HomePoints",
            }
        )
        df_url = df_url.drop(
            ["Unnamed: 6", "Unnamed: 7", "Attend.", "Arena", "Notes"], axis=1
        )  # Remove non interesting columns
        df_url = df_url.dropna(
            subset=["AwayPoints", "HomePoints"]
        )  # Remove rows containing games not yet played

        my_file = Path(os.getcwd() + "/" + folder + month + "_data.csv")

        if (
            not my_file.exists()
        ):  # If current data is not present in past_data folder, add it
            df_url.to_csv(my_file, index=False)
            logger.info(f"An update has been made: {month}_data.csv has been created.")

        logger.info(f"{month}_data.csv is up to date.")
        # Avoid multiple requests error
        time.sleep(5)
    except Exception as exc:
        logger.info(f"An exception occured while reading {month}.\nURL: {url}")

month_dfs = []
for month in months:
    try:
        month_df = pd.read_csv(f"{folder}{month}_data.csv")
        month_dfs.append(month_df)
    except Exception as exc:
        logger.info(f"An exception occured while reading {month}.")
        raise Exception

season_df = pd.concat(month_dfs, ignore_index=True)
season_df.to_csv(f"{folder}{config['years']}_season.csv", index=False)

df = pd.DataFrame(columns=dal.columns_data_dict)

trans = DataTransformer.Transformation(folder)

for _, row in season_df.iterrows():
    logger.info(row["HomeTeam"])
    logger.info(row["AwayTeam"])
    try:
        date = row["Date"].partition(", ")[2]
        month = dal.months_dict[date[:3]]
        day = date.partition(",")[0][4:]
        if len(day) == 1:
            day = "0" + day
        year = date[-4:]
    except Exception as e:
        logger.error(e)

    try:
        home_team_short = dal.teams_dict[row["HomeTeam"]]
        url = (
            "https://www.basketball-reference.com/boxscores/"
            + year
            + month
            + day
            + "0"
            + home_team_short
            + ".html"
        )
        logger.info(f"Fetching data from: {url}")
    except Exception as e:
        logger.error(e)

    try:
        tables = pd.read_html(url, match="Basic")
        # Avoid multiple requests error
        time.sleep(5)
        for table in tables:
            try:
                if (
                    int(table.loc[table.index[-1], ("Basic Box Score Stats", "MP")])
                ) >= 240:  # Get only the full game tables
                    if (
                        int(
                            table.loc[table.index[-1], ("Basic Box Score Stats", "PTS")]
                        )
                    ) == int(row["HomePoints"]):
                        data_dict = trans.append_stats_per_game(
                            df=table, team=row["HomeTeam"]
                        )
                    else:
                        data_dict = trans.append_stats_per_game(
                            df=table, team=row["AwayTeam"]
                        )
            except Exception as e:
                logger.error(e)
    except Exception as e:
        logger.error(e)

df["Team"] = data_dict["Team"]
df["MP"] = data_dict["MP"]
df["FG"] = data_dict["FG"]
df["FGA"] = data_dict["FGA"]
df["FG%"] = data_dict["FG%"]
df["3P"] = data_dict["3P"]
df["3PA"] = data_dict["3PA"]
df["3P%"] = data_dict["3P%"]
df["FT"] = data_dict["FT"]
df["FTA"] = data_dict["FTA"]
df["FT%"] = data_dict["FT%"]
df["ORB"] = data_dict["ORB"]
df["DRB"] = data_dict["DRB"]
df["TRB"] = data_dict["TRB"]
df["AST"] = data_dict["AST"]
df["STL"] = data_dict["STL"]
df["BLK"] = data_dict["BLK"]
df["TOV"] = data_dict["TOV"]
df["PF"] = data_dict["PF"]
df["PTS"] = data_dict["PTS"]
df["+/-"] = data_dict["+/-"]

df.to_csv(f"{folder}half_stats_per_game-{season}.csv", index=False)

trans.split_stats_per_game(f"half_stats_per_game-{season}.csv") """

split_df = pd.read_csv(f'{folder}split_stats_per_game.csv')

df = Helper.add_features_to_df(split_df)

df.to_csv(f'{folder}split_stats_per_game.csv', index=False)

elo.add_elo_to_df(folder)

# TODO add merged data up to season just processed
# TODO look at build_merged_seasons():

"""
df_new = pd.read_csv(f'{folder}split_stats_per_game-{season}.csv', index_col=False)
df_old = pd.read_csv(f"src/past_data/merged_seasons/{config['available_merged_data']}_Stats.csv", index_col=False)

df_new.drop('Date', axis=1, inplace=True)

df = pd.concat([df_new, df_old], axis=0)
df.to_csv(f'src/past_data/merged_seasons/2017_to_{season}_Stats.csv', index=False)
"""
