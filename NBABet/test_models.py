import numpy as np
import pandas as pd
import Models
import backtesting
import dicts_and_lists as dal
import logging, coloredlogs

pd.set_option('display.max_rows', 500)

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
odds_winner = []
odds_loser  = []

# Maximum allowed average_N: 35
average_N = 5
skip_n = 20
print(f'Stats averaged from {average_N} games, first {skip_n} games are skipped.')

for skip_n_games in range(skip_n, 36-average_N):
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
        odds_winner.append(next_game['OddsWinner'].values[0])
        odds_loser.append(next_game['OddsLoser'].values[0])

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
        odds_winner.append(next_game['OddsWinner'].values[0])
        odds_loser.append(next_game['OddsLoser'].values[0])

    difference = np.array(predictions) - np.array(true_values)
    accuracy = np.count_nonzero(difference==0)/len(difference)
    print(f'Accuracy after {len(difference)} samples: {accuracy:.3f}')

# Evaluate the predictions
data = {
    'Predictions' : predictions,
    'TrueValues'  : true_values,
    'OddsWinner'  : odds_winner,
    'OddsLoser'   : odds_loser
}
ev_df = pd.DataFrame(data)

# Calculate accuracy of predicted teams, when they were the favorite by a margin
margin = 0.1

correctly_predicted = ev_df.loc[
    (ev_df['Predictions'] == ev_df['TrueValues']) &
    (ev_df['OddsLoser'] >= ev_df['OddsWinner'] + margin)
    ].count()

total = ev_df.loc[
    (ev_df['OddsLoser'] >= ev_df['OddsWinner'] + margin)
    ].count()

accuracy_favorite = correctly_predicted[0]/total[0]
print(f'Accuracy of predicted teams when they were the favorite: {accuracy_favorite:.2f}')

# Drop the rows under the given accuracy
ev_df = ev_df[ev_df['OddsWinner'] > (1/accuracy_favorite)]

# Compare Predictions and TrueValues
comparison_column = np.where(ev_df['Predictions'] == ev_df['TrueValues'], True, False)

# Bet a different amount depending on odds
ev_df = ev_df.assign(BetAmount = (-2.5*ev_df['OddsWinner'] + 15))
ev_df['BetAmount'].loc[(ev_df['BetAmount'] < 2)] = 2

net_worth = comparison_column * ev_df['OddsWinner'] * ev_df['BetAmount'] - ev_df['BetAmount']

# Extract the rows where the model predicted the lowest odds between the two teams,
# i.e., where the team is the favorite to win
ev_df = ev_df.loc[
    (ev_df['OddsLoser'] >= ev_df['OddsWinner'] + margin)
    ]

# Assign new Net Worth row
ev_df['NetWorth'] = net_worth
print(ev_df)
print(f'Net worth: {net_worth.sum():.2f} €')
