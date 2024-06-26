from typing import Tuple
import pandas as pd
import os
from pathlib import Path
from pandas.core.frame import DataFrame
from src.ETL import DataLoader
import logging, coloredlogs
import src.dicts_and_lists as dal
import yaml

pd.options.mode.chained_assignment = None

# ------ Logger ------- #
logger = logging.getLogger("DataTransformer.py")
coloredlogs.install(level="DEBUG")


class Transformation:
    """
    Transformation represents the second module in the ETL pipeline.
    Data passed in this method is polished, arranged and organized in specific columns.
    """

    def __init__(self, folder: str) -> None:
        self.folder = folder
        with open("src/configs/main_conf.yaml") as f:
            self.config = yaml.safe_load(f)
        self.years = self.config["years"]

    def polish_df_month(
        self, df_month: pd.DataFrame, current_month: str
    ) -> Tuple[pd.DataFrame, str]:
        """
        Cleans the DataFrame by
            1) Renaming the columns,
            2) Dropping non useful ones,
            3) Removing rows containing games not yet played
        """

        df_polished = (
            df_month.filter(
                items=[
                    "Visitor/Neutral",
                    "Home/Neutral",
                    "PTS",
                    "PTS.1"
                ]
            )
            .rename(
                columns={
                    "Visitor/Neutral": "AwayTeam",
                    "Home/Neutral": "HomeTeam",
                    "PTS": "AwayPoints",
                    "PTS.1": "HomePoints",
                }
            )
            .dropna(subset=["AwayPoints", "HomePoints"])
        )

        csv_path = Path(os.getcwd() + "/" + self.folder + current_month + "_data.csv")

        return df_polished, csv_path

    @staticmethod
    def append_stats_per_game(df: DataFrame, team: str):
        dal.data_dict["Team"].append(team)
        dal.data_dict["MP"].append(
            int(df.loc[df.index[-1], ("Basic Box Score Stats", "MP")])
        )
        dal.data_dict["FG"].append(
            int(df.loc[df.index[-1], ("Basic Box Score Stats", "FG")])
        )
        dal.data_dict["FGA"].append(
            int(df.loc[df.index[-1], ("Basic Box Score Stats", "FGA")])
        )
        dal.data_dict["FG%"].append(
            float(df.loc[df.index[-1], ("Basic Box Score Stats", "FG%")])
        )
        dal.data_dict["3P"].append(
            int(df.loc[df.index[-1], ("Basic Box Score Stats", "3P")])
        )
        dal.data_dict["3PA"].append(
            int(df.loc[df.index[-1], ("Basic Box Score Stats", "3PA")])
        )
        dal.data_dict["3P%"].append(
            float(df.loc[df.index[-1], ("Basic Box Score Stats", "3P%")])
        )
        dal.data_dict["FT"].append(
            int(df.loc[df.index[-1], ("Basic Box Score Stats", "FT")])
        )
        dal.data_dict["FTA"].append(
            int(df.loc[df.index[-1], ("Basic Box Score Stats", "FTA")])
        )
        dal.data_dict["FT%"].append(
            float(df.loc[df.index[-1], ("Basic Box Score Stats", "FT%")])
        )
        dal.data_dict["ORB"].append(
            int(df.loc[df.index[-1], ("Basic Box Score Stats", "ORB")])
        )
        dal.data_dict["DRB"].append(
            int(df.loc[df.index[-1], ("Basic Box Score Stats", "DRB")])
        )
        dal.data_dict["TRB"].append(
            int(df.loc[df.index[-1], ("Basic Box Score Stats", "TRB")])
        )
        dal.data_dict["AST"].append(
            int(df.loc[df.index[-1], ("Basic Box Score Stats", "AST")])
        )
        dal.data_dict["STL"].append(
            int(df.loc[df.index[-1], ("Basic Box Score Stats", "STL")])
        )
        dal.data_dict["BLK"].append(
            int(df.loc[df.index[-1], ("Basic Box Score Stats", "BLK")])
        )
        dal.data_dict["TOV"].append(
            int(df.loc[df.index[-1], ("Basic Box Score Stats", "TOV")])
        )
        dal.data_dict["PF"].append(
            int(df.loc[df.index[-1], ("Basic Box Score Stats", "PF")])
        )
        dal.data_dict["PTS"].append(
            int(df.loc[df.index[-1], ("Basic Box Score Stats", "PTS")])
        )
        dal.data_dict["+/-"].append(
            df.loc[df.index[-1], ("Basic Box Score Stats", "+/-")]
        )
        return dal.data_dict

    @staticmethod
    def assign_teams_data_to_df(df: DataFrame):
        df["Team"] = dal.data_dict["Team"]
        df["MP"] = dal.data_dict["MP"]
        df["FG"] = dal.data_dict["FG"]
        df["FGA"] = dal.data_dict["FGA"]
        df["FG%"] = dal.data_dict["FG%"]
        df["3P"] = dal.data_dict["3P"]
        df["3PA"] = dal.data_dict["3PA"]
        df["3P%"] = dal.data_dict["3P%"]
        df["FT"] = dal.data_dict["FT"]
        df["FTA"] = dal.data_dict["FTA"]
        df["FT%"] = dal.data_dict["FT%"]
        df["ORB"] = dal.data_dict["ORB"]
        df["DRB"] = dal.data_dict["DRB"]
        df["TRB"] = dal.data_dict["TRB"]
        df["AST"] = dal.data_dict["AST"]
        df["STL"] = dal.data_dict["STL"]
        df["BLK"] = dal.data_dict["BLK"]
        df["TOV"] = dal.data_dict["TOV"]
        df["PF"] = dal.data_dict["PF"]
        df["PTS"] = dal.data_dict["PTS"]
        df["+/-"] = dal.data_dict["+/-"]

    def split_stats_per_game(self, filename="stats_per_game.csv"):
        """
        Starting from stats_per_game.csv, create a file containing on each row the stats from the two teams.
        """
        df = pd.read_csv(self.folder + filename, index_col=False)

        spg_away: DataFrame = df.iloc[::2]
        spg_home: DataFrame = df.iloc[1::2]

        spg_away = spg_away.drop(["+/-"], axis=1)
        spg_home = spg_home.drop(["+/-"], axis=1)

        spg_away = spg_away.rename(dal.spg_away, axis=1)
        spg_home = spg_home.rename(dal.spg_home, axis=1)

        spg_away.reset_index(drop=True, inplace=True)
        spg_home.reset_index(drop=True, inplace=True)

        df = pd.concat([spg_away, spg_home], axis=1)

        # Assign a column containing the winner: 0 = Home, 1 = Away
        df = df.assign(Winner=0)  # Set the winner as the Home Team
        df.loc[
            df["PTS_away"] > df["PTS_home"], "Winner"
        ] = 1  # Change to Away if PTS_away > PTS_home

        # Assign the date per single game based on past season DataFrame
        season_df = pd.read_csv(
            self.folder + f"{self.years}_season.csv", index_col=False
        )
        df.insert(loc=0, column="Date", value=season_df["Date"])

        Loading = DataLoader.Loading(self.folder)

        Loading.save_split_stats_per_game(df)
