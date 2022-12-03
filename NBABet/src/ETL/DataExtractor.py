from typing import Tuple
import pandas as pd
from datetime import datetime, date, timedelta
import numpy as np
import os
from ETL import DataTransformer
import src.dicts_and_lists as dal
import logging, coloredlogs
pd.options.mode.chained_assignment = None

# ------ Logger ------- #
logger = logging.getLogger('DataExtractor.py')
coloredlogs.install(level='DEBUG')

class Extraction():
    """
    Extraction represents the first module in the ETL pipeline.
    It involves the acquisition of data from basketball-reference.com.
    """

    def __init__(self, folder: str, season: str) -> None:
        self.folder = folder
        self.season = season


    def get_current_month_data(self) -> Tuple[pd.DataFrame, str]:
        """
        Checks if the 202*-202*_season.csv file is up to date.
        If not, new rows are added to the file.
        """
        # If it is the first day of the month, get last month's data. Otherwise, get this month's one.
        if date.today().day == 1:
            yesterday = datetime.now() - timedelta(1)
            current_month = yesterday.strftime("%B").lower()
        else:
            current_month = date.today().strftime("%B").lower()

        # Retrieve url based on current month
        url = 'https://www.basketball-reference.com/leagues/NBA_'+ self.season + '_games-' + current_month + '.html'
        df_month = pd.read_html(url)[0]

        return df_month, current_month

    
    def get_stats_per_game(self, diff: pd.DataFrame) -> None:
        """
        For each game in the diff DataFrame, get in-game stats (e.g., Steals, Assists, etc.).
        """

        Transformation = DataTransformer.Transformation(self.folder)

        df = pd.DataFrame(columns = dal.columns_data_dict)

        for _, row in diff.iterrows():
            date = row['Date'].partition(', ')[2]
            month = dal.months_dict[date[:3]]
            day = date.partition(',')[0][4:]
            if len(day) == 1:
                day = '0' + day
            year = date[-4:]
            home_team_short = dal.teams_dict[row['HomeTeam']]
            url = 'https://www.basketball-reference.com/boxscores/' + year + month + day + '0' + home_team_short + '.html'
            logger.info(f'Fetching data from: {url}')
            tables = pd.read_html(url, match='Basic')
            for table in tables:
                if (table.loc[table.index[-1], ('Basic Box Score Stats', 'MP')]) is np.nan:
                    continue
                if (int(table.loc[table.index[-1], ('Basic Box Score Stats', 'MP')])) >= 240: # Get only the full game tables
                    if (int(table.loc[table.index[-1], ('Basic Box Score Stats', 'PTS')])) == row['HomePoints']:
                        Transformation.append_stats_per_game(df=table, team=row['HomeTeam'])
                    else:
                        Transformation.append_stats_per_game(df=table, team=row['AwayTeam'])
        
        Transformation.assign_teams_data_to_df(df)

        # Concatenate new data
        stats_per_game_df = pd.read_csv(self.folder + 'stats_per_game.csv')
        stats_per_game_df = pd.concat([stats_per_game_df, df])
        stats_per_game_df = stats_per_game_df.drop_duplicates().reset_index(drop=True)
        stats_per_game_df.to_csv(self.folder + 'stats_per_game.csv', index=False)
        
        Transformation.split_stats_per_game()

        

