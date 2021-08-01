# External Libraries
from datetime import date
import pandas as pd
import os
from pathlib import Path
from pandas.core.frame import DataFrame
import logging, coloredlogs
# Internal Libraries
import Elo
import dicts_and_lists as dal

# ------ Logger ------- #
logger = logging.getLogger('Helper.py')
coloredlogs.install(level='DEBUG')

# Functions
def append_stats_per_game(df, team):
    dal.data_dict['Team'].append(team)
    dal.data_dict['MP'].append(int(df.loc[df.index[-1], ('Basic Box Score Stats', 'MP')]))
    dal.data_dict['FG'].append(int(df.loc[df.index[-1], ('Basic Box Score Stats', 'FG')]))
    dal.data_dict['FGA'].append(int(df.loc[df.index[-1], ('Basic Box Score Stats', 'FGA')]))
    dal.data_dict['FG%'].append(float(df.loc[df.index[-1], ('Basic Box Score Stats', 'FG%')]))
    dal.data_dict['3P'].append(int(df.loc[df.index[-1], ('Basic Box Score Stats', '3P')]))
    dal.data_dict['3PA'].append(int(df.loc[df.index[-1], ('Basic Box Score Stats', '3PA')]))
    dal.data_dict['3P%'].append(float(df.loc[df.index[-1], ('Basic Box Score Stats', '3P%')]))
    dal.data_dict['FT'].append(int(df.loc[df.index[-1], ('Basic Box Score Stats', 'FT')]))
    dal.data_dict['FTA'].append(int(df.loc[df.index[-1], ('Basic Box Score Stats', 'FTA')]))
    dal.data_dict['FT%'].append(float(df.loc[df.index[-1], ('Basic Box Score Stats', 'FT%')]))
    dal.data_dict['ORB'].append(int(df.loc[df.index[-1], ('Basic Box Score Stats', 'ORB')]))
    dal.data_dict['DRB'].append(int(df.loc[df.index[-1], ('Basic Box Score Stats', 'DRB')]))
    dal.data_dict['TRB'].append(int(df.loc[df.index[-1], ('Basic Box Score Stats', 'TRB')]))
    dal.data_dict['AST'].append(int(df.loc[df.index[-1], ('Basic Box Score Stats', 'AST')]))
    dal.data_dict['STL'].append(int(df.loc[df.index[-1], ('Basic Box Score Stats', 'STL')]))
    dal.data_dict['BLK'].append(int(df.loc[df.index[-1], ('Basic Box Score Stats', 'BLK')]))
    dal.data_dict['TOV'].append(int(df.loc[df.index[-1], ('Basic Box Score Stats', 'TOV')]))
    dal.data_dict['PF'].append(int(df.loc[df.index[-1], ('Basic Box Score Stats', 'PF')]))
    dal.data_dict['PTS'].append(int(df.loc[df.index[-1], ('Basic Box Score Stats', 'PTS')]))
    dal.data_dict['+/-'].append(df.loc[df.index[-1], ('Basic Box Score Stats', '+/-')])

def build_season_df(folder):
    december_df = pd.read_csv(folder + 'december_data.csv')
    january_df = pd.read_csv(folder + 'january_data.csv')
    february_df = pd.read_csv(folder + 'february_data.csv')
    march_df = pd.read_csv(folder + 'march_data.csv')
    april_df = pd.read_csv(folder + 'april_data.csv')
    may_df = pd.read_csv(folder + 'may_data.csv')
    june_df = pd.read_csv(folder + 'june_data.csv')

    season_df = pd.concat([december_df, january_df, february_df, march_df, april_df, may_df, june_df])
    season_df.to_csv(folder + '2020_2021_season.csv', index=False)

    return season_df

def build_stats_per_game_csv(folder:str):
    # Compute the URL based on the Date field
    season_df = pd.read_csv(folder + '2020_2021_season.csv') 

    df = pd.DataFrame(columns = dal.columns_data_dict)

    for _, row in season_df.iterrows():
        # Get day, month and year 
        try:
            date = row['Date'].partition(', ')[2]
            month = dal.months_dict[date[:3]]
            day = date.partition(',')[0][4:]
            if len(day) == 1:
                day = '0' + day
            year = date[-4:]
        except Exception as e:
            logger.error(e)

        # Use the day, month and year to scrape the NBA stats website    
        try:
            home_team_short = dal.teams_dict[row['HomeTeam']]
            url = 'https://www.basketball-reference.com/boxscores/' + year + month + day + '0' + home_team_short + '.html'
            logger.info(f'Fetching data from: {url}')
        except Exception as e:
            logger.error(e)

        # Extract the tables from the website
        try:
            tables = pd.read_html(url, match='Basic')
            for table in tables:
                try:
                    if (int(table.loc[table.index[-1], ('Basic Box Score Stats', 'MP')])) >= 240: # Get only the full game tables
                        if (int(table.loc[table.index[-1], ('Basic Box Score Stats', 'PTS')])) == row['HomePoints']:
                            append_stats_per_game(df=table, team=row['HomeTeam'])
                        else:
                            append_stats_per_game(df=table, team=row['AwayTeam'])
                except Exception as e:
                    logger.error(e)
        except Exception as e:
            logger.error(e)
    
    df['Team'] = dal.data_dict['Team']
    df['MP']   = dal.data_dict['MP']
    df['FG']   = dal.data_dict['FG']
    df['FGA']  = dal.data_dict['FGA']
    df['FG%']  = dal.data_dict['FG%']
    df['3P']   = dal.data_dict['3P']
    df['3PA']  = dal.data_dict['3PA']
    df['3P%']  = dal.data_dict['3P%']
    df['FT']   = dal.data_dict['FT']
    df['FTA']  = dal.data_dict['FTA']
    df['FT%']  = dal.data_dict['FT%']
    df['ORB']  = dal.data_dict['ORB']
    df['DRB']  = dal.data_dict['DRB']
    df['TRB']  = dal.data_dict['TRB']
    df['AST']  = dal.data_dict['AST']
    df['STL']  = dal.data_dict['STL']
    df['BLK']  = dal.data_dict['BLK']
    df['TOV']  = dal.data_dict['TOV']
    df['PF']   = dal.data_dict['PF']
    df['PTS']  = dal.data_dict['PTS']
    df['+/-']  = dal.data_dict['+/-']

    df.to_csv(folder + 'stats_per_game.csv', index=False)
    split_stats_per_game(folder)
    
def check_df(folder:str):
    """
    Checks if 2020_2021_season.csv file is up to date.
    If not, new rows are added to the file.
    """
    current_month = date.today().strftime("%B").lower()
    current_month = 'july'
    # Retrieve url based on current month
    url = 'https://www.basketball-reference.com/leagues/NBA_2021_games-'+ current_month + '.html'
    df_url = pd.read_html(url)[0]
    df_url = df_url.rename(columns=
        {
            'Visitor/Neutral' : 'AwayTeam',
            'Home/Neutral' : 'HomeTeam',
            'PTS' : 'AwayPoints',
            'PTS.1' : 'HomePoints'
        }
    )
    df_url = df_url.drop(['Unnamed: 6', 'Unnamed: 7', 'Attend.', 'Notes'], axis=1) # Remove non interesting columns
    df_url = df_url.dropna(subset=['AwayPoints', 'HomePoints']) # Remove rows containing games not yet played

    my_file = Path(os.getcwd() + '/' + folder + current_month + '_data.csv')

    if not my_file.exists(): # If current data is not present in past_data folder, add it
        df_url.to_csv(folder + current_month + '_data.csv', index=False) # Save the df as .csv
        season_df = pd.read_csv(folder + '/2020_2021_season.csv')
        
        logger.info(f'An update has been made: {current_month}_data.csv has been created.')
        update_elo_csv(df_url)
        
        logger.info(f'An update has been made: elo.csv updated based on {current_month}_data.csv.')
        update_stats_per_game_csv(folder, df_url)

        # Drop duplicates in season_df if it already contains data from current month
        if not ((season_df.merge(df_url)).drop_duplicates().reset_index(drop=True).equals(df_url)): 
            season_df = pd.concat([season_df, df_url])
            season_df = season_df.drop_duplicates().reset_index(drop=True)
            season_df.to_csv(folder + '2020_2021_season.csv', index=False)
            logger.info(f'An update has been made: rows from {current_month}_data.csv have been added to the 2020_2021_season.csv file.')
            
    # If the file did not exist, it has been created.
    df_old = pd.read_csv(folder + current_month + '_data.csv') # Extract the old DataFrame
    
    rows_df_old = df_old.shape[0]
    rows_df_url = df_url.shape[0]
    
    # Compare if there are new rows
    if rows_df_url > rows_df_old:
        # Get the new rows (Pandas right-excluding merge)
        diff = df_old.merge(df_url, how='right', indicator=True).query('_merge == "right_only"').drop('_merge', 1)

        # Compute the intersection between the old and the new DataFrame
        season_df = pd.read_csv(folder + '2020_2021_season.csv')
        season_df = season_df.drop_duplicates().reset_index(drop=True)
        inner_merged = pd.merge(season_df, diff, how='inner')

        # If the intersection between the two DataFrames is the original DataFrame, it already contains the diff rows
        # If not, add the diff rows to the month and the season DataFrame
        if not season_df.equals(inner_merged): 
            # Update rows in the month DataFrame
            df_url.to_csv(folder + current_month + '_data.csv', index=False) # Save the df as .csv
            logger.info(f'An update has been made: new rows have been added to {current_month}_data.csv file.')
            # Update rows in the Season DataFrame
            season_df = pd.concat([season_df, diff])
            season_df = season_df.drop_duplicates().reset_index(drop=True)
            season_df.to_csv(folder + '2020_2021_season.csv', index=False)
            logger.info(f'An update has been made: new rows have been added to the 2020_2021_season.csv file.')
            logger.info(f'Added rows:\n {diff}')
            
        # Following is a pipeline of actions to be performed every time new rows are added.
        update_elo_csv(diff)
        update_stats_per_game_csv(folder, diff)
    
    logger.info(f'\n----- Dataset 2020_2021_season.csv is up to date. -----\n')


def elo_setup():
    # Initial Elo setup
    df = pd.DataFrame(dal.teams, columns=['Team'])
    df = Elo.setup(df)
    df.to_csv('elo.csv', index=False)
    return df

def split_stats_per_game(folder:str):
    df = pd.read_csv(folder + 'stats_per_game.csv', index_col=False)
    spg_away =  df.iloc[::2]
    spg_home =  df.iloc[1::2]

    spg_away = spg_away.drop(['+/-'], axis=1)
    spg_home = spg_home.drop(['+/-'], axis=1)

    spg_away = spg_away.rename(dal.spg_away, axis=1)
    spg_home = spg_home.rename(dal.spg_home, axis=1)

    spg_away.reset_index(drop=True, inplace=True)
    spg_home.reset_index(drop=True, inplace=True)

    df = pd.concat([spg_away, spg_home], axis=1)
    
    # Assign a column containing the winner: 0 = Home, 1 = Away
    df = df.assign(Winner = 0) # Set the winner as the Home Team
    df['Winner'].loc[df['PTS_away'] > df['PTS_home']] = 1 # Change to Away if PTS_away > PTS_home
    
    df.to_csv(folder + 'split_stats_per_game.csv', index=False)

    return df

def update_elo_csv(df):
    """
    Updates the elo.csv dataset based on the rows contained in df. 
    """
    elo_df = pd.read_csv('elo.csv')

    for _, row in df.iterrows():
        away_team = row['AwayTeam']
        home_team = row['HomeTeam']
        away_pts = row['AwayPoints']
        home_pts = row['HomePoints']
        
        if(away_pts > home_pts):
            winner = away_team
        elif(home_pts > away_pts):
            winner = home_team
        elo_df = Elo.update(elo_df, away_team, home_team, winner)
    
    elo_df.to_csv('elo.csv', index=False)


def update_stats_per_game_csv(folder:str, diff:DataFrame):
    """
    Updates the stats_per_game.csv dataset based on the rows contained in diff. 
    """
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
            if (int(table.loc[table.index[-1], ('Basic Box Score Stats', 'MP')])) >= 240: # Get only the full game tables
                if (int(table.loc[table.index[-1], ('Basic Box Score Stats', 'PTS')])) == row['HomePoints']:
                    append_stats_per_game(df=table, team=row['HomeTeam'])
                else:
                    append_stats_per_game(df=table, team=row['AwayTeam'])
    
    df['Team'] = dal.data_dict['Team']
    df['MP']   = dal.data_dict['MP']
    df['FG']   = dal.data_dict['FG']
    df['FGA']  = dal.data_dict['FGA']
    df['FG%']  = dal.data_dict['FG%']
    df['3P']   = dal.data_dict['3P']
    df['3PA']  = dal.data_dict['3PA']
    df['3P%']  = dal.data_dict['3P%']
    df['FT']   = dal.data_dict['FT']
    df['FTA']  = dal.data_dict['FTA']
    df['FT%']  = dal.data_dict['FT%']
    df['ORB']  = dal.data_dict['ORB']
    df['DRB']  = dal.data_dict['DRB']
    df['TRB']  = dal.data_dict['TRB']
    df['AST']  = dal.data_dict['AST']
    df['STL']  = dal.data_dict['STL']
    df['BLK']  = dal.data_dict['BLK']
    df['TOV']  = dal.data_dict['TOV']
    df['PF']   = dal.data_dict['PF']
    df['PTS']  = dal.data_dict['PTS']
    df['+/-']  = dal.data_dict['+/-']

    # Concatenate new data
    stats_per_game_df = pd.read_csv(folder + 'stats_per_game.csv')
    stats_per_game_df = pd.concat([stats_per_game_df, df])
    stats_per_game_df = stats_per_game_df.drop_duplicates().reset_index(drop=True)
    stats_per_game_df.to_csv(folder + 'stats_per_game.csv', index=False)
    split_stats_per_game(folder)
