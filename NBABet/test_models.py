import numpy as np
import pandas as pd
import Models
import backtesting
import dicts_and_lists as dal
import logging, coloredlogs

# ------ Logger ------- #
logger = logging.getLogger('test_models.py')
coloredlogs.install(level='DEBUG')

# Only the most significant features will be considered
away_features = Models.away_features
home_features = Models.home_features
features = Models.features

### Test the Bagging Classification model based on the mean of the last average_N games ###
logger.info('\nSelect the type of model you want to backtest:\n\
    [1]: AdaBoost\n\
    [2]: Decision Tree\n\
    [3]: Random Forest\n\
    [4]: Support Vector Machine'
    )
inp = input()
if inp == '1':
    logger.info('Building an AdaBoost Classifier...')
    clf = Models.build_AdaBoost_classifier()
elif inp == '2':
    logger.info('Building a Decision Tree Classifier...')
    clf = Models.build_DT_classifier()
elif inp == '3':
    logger.info('Building a Random Forest Classifier...')
    clf = Models.build_RF_classifier()
elif inp == '4':
    logger.info('Building a Support Vector Machine Classifier...')
    clf = Models.build_SVM_classifier()

##### Try some backtesting

df = pd.read_csv('past_data/2020_2021/split_stats_per_game.csv')

# To evaluate accuracy
predictions = []
true_values = []

# Maximum allowed average_N: 35
average_N = 10
print(f'Stats averaged from {average_N} games.')

for skip_n_games in range(36-average_N):
    last_N_games_home, last_N_games_away = backtesting.get_first_N_games(df, average_N, skip_n_games)
    # Get next game based on next_game_index
    for team in dal.teams:
        # Find next games where "team" plays away
        next_games_indexes = df.loc[df['Team_away'] == team].index
        last_away_game = last_N_games_away[dal.teams_to_int[team]][-1:]
        next_game_index = min(i for i in next_games_indexes if i > last_away_game.index)
        next_game = df.loc[df.index == next_game_index]

        # Extract away_team Name and home_team Name from last_N_games_away and last_N_games_home
        away_team = next_game['Team_away'].values[0]
        home_team = next_game['Team_home'].values[0]

        # Concatenate the two teams with their average stats
        to_predict = pd.concat(
            [
                last_N_games_away[dal.teams_to_int[away_team]][away_features].mean(), 
                last_N_games_home[dal.teams_to_int[home_team]][home_features].mean()
            ],
            axis=0)[features]

        pred = int(clf.predict(to_predict.values.reshape(1,-1)))
        true_value = next_game['Winner'].values[0]
        predictions.append(pred)
        true_values.append(true_value)

        # Find next games where "team" plays home
        next_games_indexes = df.loc[df['Team_home'] == team].index
        last_home_game = last_N_games_home[dal.teams_to_int[team]][-1:]
        next_game_index = min(i for i in next_games_indexes if i > last_home_game.index)
        next_game = df.loc[df.index == next_game_index]

        # Extract away_team Name and home_team Name from last_N_games_away and last_N_games_home
        away_team = next_game['Team_away'].values[0]
        home_team = next_game['Team_home'].values[0]

        # Concatenate the two teams with their average stats
        to_predict = pd.concat(
            [
                last_N_games_away[dal.teams_to_int[away_team]][away_features].mean(), 
                last_N_games_home[dal.teams_to_int[home_team]][home_features].mean()
            ],
            axis=0)[features]

        pred = int(clf.predict(to_predict.values.reshape(1,-1)))
        true_value = next_game['Winner'].values[0]
        predictions.append(pred)
        true_values.append(true_value)

    difference = np.array(predictions) - np.array(true_values)
    print(f'Accuracy after {len(difference)} samples: {np.count_nonzero(difference==0)/len(difference):.3f}')