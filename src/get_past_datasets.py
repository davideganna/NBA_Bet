# Inside src: python get_past_datasets.py

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

# ------ Logger ------- #
logger = logging.getLogger('get_past_datasets.py')
coloredlogs.install(level='DEBUG')

year = '2023'
folder = 'past_data/2022-2023/'

months = ['october', 'november', 'december'] #'january', 'february', 'march', 'april', 'may', 'june']

for month in months:
    url = f'https://www.basketball-reference.com/leagues/NBA_{year}_games-{month}.html'
    df_url = pd.read_html(url)[0]
    df_url = df_url.rename(columns=
        {
            'Visitor/Neutral' : 'AwayTeam',
            'Home/Neutral' : 'HomeTeam',
            'PTS' : 'AwayPoints',
            'PTS.1' : 'HomePoints'
        }
    )
    df_url = df_url.drop(['Unnamed: 6', 'Unnamed: 7', 'Attend.', 'Arena', 'Notes'], axis=1) # Remove non interesting columns
    df_url = df_url.dropna(subset=['AwayPoints', 'HomePoints']) # Remove rows containing games not yet played

    my_file = Path(os.getcwd() + '/' + folder + month + '_data.csv')

    if not my_file.exists(): # If current data is not present in past_data folder, add it
        df_url.to_csv(my_file, index=False) # Save the df as .csv
        logger.info(f'An update has been made: {month}_data.csv has been created.')
    
    logger.info(f'{month}_data.csv is up to date.')

# Create a big dataset
october_df = pd.read_csv(folder + 'october_data.csv')
november_df = pd.read_csv(folder + 'november_data.csv')
december_df = pd.read_csv(folder + 'december_data.csv')
#january_df = pd.read_csv(folder + 'january_data.csv')
#february_df = pd.read_csv(folder + 'february_data.csv')
#march_df = pd.read_csv(folder + 'march_data.csv')
#april_df = pd.read_csv(folder + 'april_data.csv')
#may_df = pd.read_csv(folder + 'may_data.csv')
#june_df = pd.read_csv(folder + 'june_data.csv')

season_df = pd.concat([october_df, november_df, december_df])
season_df.to_csv(folder + '2022-2023_season.csv', index=False)

"""
df = pd.DataFrame(columns = dal.columns_data_dict)

for _, row in season_df.iterrows():
    print(row['HomeTeam'])
    print(row['AwayTeam'])
    try:
        date = row['Date'].partition(', ')[2]
        month = dal.months_dict[date[:3]]
        day = date.partition(',')[0][4:]
        if len(day) == 1:
            day = '0' + day
        year = date[-4:]
    except Exception as e:
        logger.error(e)

    try:
        home_team_short = dal.teams_dict[row['HomeTeam']]
        url = 'https://www.basketball-reference.com/boxscores/' + year + month + day + '0' + home_team_short + '.html'
        logger.info(f'Fetching data from: {url}')
    except Exception as e:
        logger.error(e)

    try:
        tables = pd.read_html(url, match='Basic')
        for table in tables:
            try:
                if (int(table.loc[table.index[-1], ('Basic Box Score Stats', 'MP')])) >= 240: # Get only the full game tables
                    if (int(table.loc[table.index[-1], ('Basic Box Score Stats', 'PTS')])) == int(row['HomePoints']):
                        Helper.append_stats_per_game(df=table, team=row['HomeTeam'])
                    else:
                        Helper.append_stats_per_game(df=table, team=row['AwayTeam'])
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

df.to_csv(folder + 'half_stats_per_game-2017.csv', index=False)

Helper.split_stats_per_game(folder)

df_new = pd.read_csv(folder + 'half_split_stats_per_game-2017.csv', index_col=False)
df_old = pd.read_csv('past_data/merged_seasons/2018_to-2020_Stats.csv', index_col=False)

df_new.drop('Date', axis=1, inplace=True)

df = pd.concat([df_new, df_old], axis=0)
df.to_csv('src/past_data/merged_seasons/2017_to-2020_Stats.csv', index=False)
"""

