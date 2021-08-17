import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Models
import backtesting
import dicts_and_lists as dal
import logging, coloredlogs

pd.set_option('display.max_rows', 1000)

# ------ Logger ------- #
logger = logging.getLogger('build_moving_average_model.py')
coloredlogs.install(level='INFO', logger=logger)


def extract_and_insert(next_game):
    """
    Based on the next game, the function computes the average of N the previous games played by 
        the same team and inserts the values in "averageN_season.csv".
    """
    # Extract away_team Name and home_team Name from last_N_games_away and last_N_games_home
    away_team = next_game['Team_away'].values[0]
    home_team = next_game['Team_home'].values[0]

    # Before predicting a game, check that it has not yet been predicted.
    # This is the case where e.g., TeamHome's next game at home against TeamAway has been evaluated ...
    # by both next home game and next away game. They are the same game, which are therefore predicted twice. 
    if next_game.index[0] not in evaluated_indexes:
        # Track the inserted game based on its index
        evaluated_indexes.append(next_game.index[0])

        # Extract indexes for last N games
        next_games_away_indexes = _df.loc[_df['Team_away'] == away_team].index
        next_games_home_indexes = _df.loc[_df['Team_home'] == home_team].index
        next_away_indexes_reduced = [x for x in next_games_away_indexes if x < next_game.index[0]][-average_N:]
        next_home_indexes_reduced = [x for x in next_games_home_indexes if x < next_game.index[0]][-average_N:]

        # Extract last N games based on indexes
        last_N_games_away = _df.iloc[next_away_indexes_reduced]
        last_N_games_home = _df.iloc[next_home_indexes_reduced]

        # Concatenate the two teams with their average stats
        to_insert = pd.concat(
            [
                last_N_games_away[away_features].mean(), 
                last_N_games_home[home_features].mean()
            ],
            axis=0)[features]

        # Print here for debugging purposes
        print(f'Next game index: {next_game.index[0]}')
        print(f'Away team: {away_team}')
        print(last_N_games_away[away_features])
        print(f'Home team: {home_team}')
        print(last_N_games_home[home_features])
        print(f'Mean to insert: {to_insert}')
        to_insert_list.append(to_insert)
        winners_list.append(_df['Winner'].loc[_df.index == next_game.index[0]].values[0])

# Only the most significant features will be considered
away_features = Models.away_features
home_features = Models.home_features
features = Models.features

# To evaluate accuracy
dates_list  = []
predictions = []
true_values = []
model_prob  = []
model_odds  = []
odds_winner = []
odds_loser  = []
home_teams_list   = []
away_teams_list   = []

df_2017 = pd.read_csv('past_data/2017_2018/split_stats_per_game_2017.csv')
df_2018 = pd.read_csv('past_data/2018_2019/split_stats_per_game_2018.csv')
df_2019 = pd.read_csv('past_data/2019_2020/split_stats_per_game_2019.csv')

# Maximum allowed average_N: 35
average_N = 10
skip_n = 0
print(f'Stats averaged from {average_N} games, first {skip_n} games are skipped.')

for _df in [df_2017, df_2018, df_2019]:
    # Cleanup at every iteration
    evaluated_indexes = []
    to_insert_list = []
    winners_list = []
    for skip_n_games in range(skip_n, 50-average_N):
        last_N_games_away, last_N_games_home = backtesting.get_first_N_games(_df, average_N, skip_n_games)
        # Get next game based on next_game_index
        for team in dal.teams:
            # Find all games where "team" plays away
            next_games_away_indexes = _df.loc[_df['Team_away'] == team].index
            last_away_game = last_N_games_away[dal.teams_to_int[team]][-1:]
            # Check if there are more games past the current index 
            try:
                dal.last_home_away_index_dict[team][0] = last_away_game.index[0]
            except: 
                pass
            if max(next_games_away_indexes) != dal.last_home_away_index_dict[team][0]:
                next_game_index = min(i for i in next_games_away_indexes[skip_n+average_N:] if i > last_away_game.index)
                next_game = _df.loc[_df.index == next_game_index]

                next_games_home_indexes = _df.loc[_df['Team_home'] == next_game['Team_home'].values[0]].index

                if next_game_index in next_games_home_indexes[skip_n+average_N:]:
                    extract_and_insert(next_game)
                    
            # Find all games where "team" plays home
            next_games_home_indexes = _df.loc[_df['Team_home'] == team].index
            last_home_game = last_N_games_home[dal.teams_to_int[team]][-1:]
            # Check if there are more games past the current index 
            try:
                dal.last_home_away_index_dict[team][1] = last_home_game.index[0]
            except: 
                pass
            if max(next_games_home_indexes) != dal.last_home_away_index_dict[team][1]:
                next_game_index = min(i for i in next_games_home_indexes[skip_n+average_N:] if i > last_home_game.index)
                next_game = _df.loc[_df.index == next_game_index]
                
                next_games_away_indexes = _df.loc[_df['Team_away'] == next_game['Team_away'].values[0]].index
                
                if next_game_index in next_games_away_indexes[skip_n+average_N:]:
                    extract_and_insert(next_game)

    avg_df = pd.concat(to_insert_list, axis=1).transpose()
    avg_df['Winner'] = winners_list
    
    if _df is df_2017:
        _df.name = '2017/2018 Season DataFrame' 
        avg_df.to_csv('past_data/average_seasons/average2017.csv', index=False)
    elif _df is df_2018:
        _df.name = '2018/2019 Season DataFrame' 
        avg_df.to_csv('past_data/average_seasons/average2018.csv', index=False)
    elif _df is df_2019:
        _df.name = '2019/2020 Season DataFrame' 
        avg_df.to_csv('past_data/average_seasons/average2019.csv', index=False)

    logger.info(f'Retrieved stats for {_df.name}')

# Concatenate the 3 average season datasets
avg_2017_df = pd.read_csv('past_data/average_seasons/average2017.csv')
avg_2018_df = pd.read_csv('past_data/average_seasons/average2018.csv')
avg_2019_df = pd.read_csv('past_data/average_seasons/average2019.csv')

avg_total_df = pd.concat([avg_2017_df, avg_2018_df, avg_2019_df], axis=0)
avg_total_df.to_csv('past_data/average_seasons/average_3seasons.csv', index=False)


