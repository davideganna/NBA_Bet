import pandas as pd
from datetime import datetime, date, timedelta
import numpy as np
import os
import src.elo as elo
import src.dicts_and_lists as dal
from pathlib import Path
from pandas.core.frame import DataFrame
from src.ETL import DataExtractor
import logging, coloredlogs
import yaml

pd.options.mode.chained_assignment = None

# ------ Logger ------- #
logger = logging.getLogger("DataLoader.py")
coloredlogs.install(level="DEBUG")


class Loading:
    """
    Loading represents the third and final module in the ETL pipeline.
    Data passed to this class is saved in dedicated .csv files.
    """

    def __init__(self, folder: str) -> None:
        with open("src/configs/main_conf.yaml") as f:
            self.config = yaml.safe_load(f)
        self.folder = folder
        self.years = self.config["years"]

    def save_df_month(self, df_month: DataFrame, current_month: str, csv_path: Path):
        # If current month is not saved as csv, create the file
        if not csv_path.exists():
            df_month.to_csv(
                self.folder + current_month + "_data.csv", index=False
            )  # Save the df as .csv
            season_df = pd.read_csv(self.folder + f"/{self.years}_season.csv")
            logger.info(
                f"An update has been made: {current_month}_data.csv has been created."
            )

            # Check if the intersection between new data and saved data is equal to new data.
            # If not, new rows have been added. Calculates the Elo and saves to season_df.
            if (
                not season_df.merge(df_month)
                .drop_duplicates()
                .reset_index(drop=True)
                .equals(df_month)
            ):
                self.update_elo_csv(df_month)
                logger.info(
                    f"An update has been made: elo.csv has been updated based on {current_month}_data.csv."
                )
                DataExtractor.Extraction(self.folder).get_stats_per_game(df_month)
                season_df = pd.concat([season_df, df_month])
                season_df = season_df.drop_duplicates().reset_index(drop=True)
                season_df.to_csv(self.folder + f"{self.years}_season.csv", index=False)
                logger.info(
                    f"An update has been made: rows from {current_month}_data.csv have been added to the {self.years}_season.csv file."
                )

        self.add_df_month_to_season_df(df_month, current_month)

    def add_df_month_to_season_df(self, df_month: DataFrame, current_month: str):
        """
        Checks if the current data already contains the retrieved data.
        If there is new data, this is added to the Season .csv file and saved.
        """
        # If the file did not exist, it has been created.
        df_old = pd.read_csv(
            self.folder + current_month + "_data.csv"
        )  # Extract the old DataFrame

        # Compare if there are new rows
        if df_month.shape[0] > df_old.shape[0]:
            # Get the new rows (Pandas right-excluding merge)
            diff = (
                df_old.merge(df_month, how="right", indicator=True)
                .query('_merge == "right_only"')
                .drop("_merge", 1)
            )

            # Compute the intersection between the old and the new DataFrame
            season_df = pd.read_csv(self.folder + f"{self.years}_season.csv")
            season_df = season_df.drop_duplicates().reset_index(drop=True)
            inner_merged = pd.merge(season_df, df_month, how="inner")

            # If the intersection between the two DataFrames is the original DataFrame, it already contains the diff rows
            # If not, add the diff rows to the month and the season DataFrame
            if not inner_merged.equals(df_month):
                # Update rows in the Season DataFrame
                season_df = pd.concat([season_df, diff])
                season_df = season_df.drop_duplicates().reset_index(drop=True)
                season_df.to_csv(self.folder + f"{self.years}_season.csv", index=False)
                logger.info(
                    f"An update has been made: new rows have been added to the {self.years}_season.csv file."
                )
                logger.info(f"Added rows:\n {diff}")

                # Following is a pipeline of actions to be performed every time new rows are added.
                self.update_elo_csv(diff)
                Extraction = DataExtractor.Extraction(self.folder)
                Extraction.get_stats_per_game(diff)

            # Update rows in the month DataFrame
            df_month.to_csv(
                self.folder + current_month + "_data.csv", index=False
            )  # Save the df as .csv
            logger.info(
                f"An update has been made: new rows have been added to {current_month}_data.csv file."
            )

        logger.info(f"\n----- Dataset {self.years}_season.csv is up to date. -----\n")

    def update_elo_csv(self, df: DataFrame):
        """
        Updates the elo.csv dataset based on the rows contained in df.
        """
        elo_df = pd.read_csv(f"src/past_data/{self.years}/elo.csv")

        for _, row in df.iterrows():
            away_team = row["AwayTeam"]
            home_team = row["HomeTeam"]
            away_pts = row["AwayPoints"]
            home_pts = row["HomePoints"]

            if away_pts > home_pts:
                winner = 1
            elif home_pts > away_pts:
                winner = 0
            elo_df = elo.update_DataFrame(
                elo_df, away_team, home_team, away_pts, home_pts, winner
            )

        elo_df.sort_values(by="Elo", ascending=False).to_csv(
            "src/past_data/" + self.folder + "/elo.csv", index=False
        )

        # Refactor
        elo_df = pd.read_csv("src/past_data/" + self.folder + "/elo.csv")

    def save_split_stats_per_game(self, df: DataFrame):
        """
        Saves the SSPG file to the corresponding location.
        """
        df.to_csv(self.folder + "split_stats_per_game.csv", index=False)
