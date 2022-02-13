import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import dicts_and_lists as dal
import Helper
import Models.moving_average_dataset
import Models.backtesting as backtesting
import Models.elo_model as elo_model
import dicts_and_lists as dal
from sklearn.metrics import confusion_matrix
import logging, coloredlogs

pd.set_option('display.max_rows', 2000)

# ------ Logger ------- #
logger = logging.getLogger('neural_networks.py')
coloredlogs.install(level='INFO', logger=logger)

# Define the features

home_features = [
    'FG%_home',
    '3P%_home',
    'FT%_home',
    'LogRatio_home',
    'RB_aggr_home',
    'eFG%_home',
    'TS%_home'
]

away_features = [
    'FG%_away',
    '3P%_away',
    'FT%_away',
    'LogRatio_away',
    'RB_aggr_away',
    'eFG%_away',
    'TS%_away'
]

features = home_features + away_features

def standardize_DataFrame(test_df:DataFrame):
    # Standardize the DataFrame
    x = test_df.loc[:, features].values

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    std_df = pd.DataFrame(x, columns=features)
    std_df = pd.concat([std_df, test_df['Winner'].reset_index(drop=True)], axis=1)
    return std_df, scaler

train_df = pd.read_csv('past_data/average_seasons/average_N_4Seasons.csv')
train_df, scaler = standardize_DataFrame(train_df)

# Validate on the 2020 dataset
test_df = pd.read_csv('past_data/average_seasons/average2020.csv')
x = test_df.loc[:, features].values
x = scaler.transform(x)
std_df = pd.DataFrame(x, columns=features)
valid_df = pd.concat([std_df, test_df['Winner'].reset_index(drop=True)], axis=1)



# ------------ Neural Networks ------------ #

# Define the Training dataset
X_train = train_df[features]
y_train = train_df['Winner']

# Define the Validation dataset
X_valid = valid_df[features]
y_valid = valid_df['Winner']

# Define the model
model = keras.Sequential([
    layers.BatchNormalization(input_shape=[14]),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),  
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid'),
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)

early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True,
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=1000,
    callbacks=[early_stopping],
    verbose=0,
)

history_df = pd.DataFrame(history.history)
# Start the plot at epoch 5
#history_df.loc[5:, ['loss', 'val_loss']].plot()
#history_df.loc[5:, ['binary_accuracy', 'val_binary_accuracy']].plot()
#plt.show()

print(("Best Validation Loss: {:0.4f}" +\
      "\nBest Validation Accuracy: {:0.4f}")\
      .format(history_df['val_loss'].min(), 
              history_df['val_binary_accuracy'].max()))


# ------------ Hyperparameters ------------ #
leave_out = '2020'
margin = 0
betting_limiter = True
betting_limit = 0.125
prob_threshold = 0.65
prob_2x_bet = 0.7
offset = 0.0 # Added probability
average_N = 3
skip_n = 0

# To evaluate accuracy
dates_list  = []
predictions = []
winners = []
model_prob  = []
model_odds  = []
odds_away = []
odds_home  = []
home_teams_list   = []
away_teams_list   = []
evaluated_indexes = []

def extract_and_predict(next_game):
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
        next_games_away_indexes = test_df.loc[test_df['Team_away'] == away_team].index
        next_games_home_indexes = test_df.loc[test_df['Team_home'] == home_team].index
        next_away_indexes_reduced = [x for x in next_games_away_indexes if x < next_game.index[0]][-average_N:]
        next_home_indexes_reduced = [x for x in next_games_home_indexes if x < next_game.index[0]][-average_N:]

        # Extract last N games based on indexes
        last_N_games_away = test_df.iloc[next_away_indexes_reduced]
        last_N_games_home = test_df.iloc[next_home_indexes_reduced]

        # Concatenate the two teams with their average stats
        to_predict = pd.concat(
            [
                last_N_games_away[away_features].mean(), 
                last_N_games_home[home_features].mean()
            ],
            axis=0)[features]
        
        # Standardize the input
        to_predict = scaler.transform(to_predict.values.reshape(1,-1))
    
        pred = model.predict(to_predict)
        if pred[0][0] > 0.5:
            predictions.append(1)
            model_prob.append(pred[0][0])
            model_odds.append(1/(pred[0][0]))
        else:
            predictions.append(0)
            model_prob.append(1 - pred[0][0])
            model_odds.append(1/(1 - pred[0][0]))
        true_winner = next_game['Winner'].values[0]
        winners.append(true_winner)
        odds_away.append(next_game['OddsAway'].values[0])
        odds_home.append(next_game['OddsHome'].values[0])
        dates_list.append(next_game['Date'].values[0])
        home_teams_list.append(home_team)
        away_teams_list.append(away_team)

# Create the test_df containing stats per single game on every row
train_df = pd.read_csv('past_data/average_seasons/average_N_4Seasons.csv')
test_df = pd.read_csv('past_data/2020_2021/split_stats_per_game.csv')

# Standardize the DataFrame
std_df, scaler = standardize_DataFrame(train_df)

for skip_n_games in range(skip_n, 50-average_N):
    last_N_games_away, last_N_games_home = backtesting.get_first_N_games(test_df, average_N, skip_n_games)
    # Get next game based on next_game_index
    for team in dal.teams:
        # Find all games where "team" plays away
        next_games_away_indexes = test_df.loc[test_df['Team_away'] == team].index
        last_away_game = last_N_games_away[dal.teams_to_int[team]][-1:]
        # Check if there are more games past the current index 
        try:
            dal.last_home_away_index_dict[team][0] = last_away_game.index[0]
        except: 
            pass
        if max(next_games_away_indexes) != dal.last_home_away_index_dict[team][0]:
            next_game_index = min(i for i in next_games_away_indexes[skip_n+average_N:] if i > last_away_game.index)
            next_game = test_df.loc[test_df.index == next_game_index]

            next_games_home_indexes = test_df.loc[test_df['Team_home'] == next_game['Team_home'].values[0]].index

            if next_game_index in next_games_home_indexes[skip_n+average_N:]:
                extract_and_predict(next_game)

        # Find all games where "team" plays home
        next_games_home_indexes = test_df.loc[test_df['Team_home'] == team].index
        last_home_game = last_N_games_home[dal.teams_to_int[team]][-1:]
        # Check if there are more games past the current index 
        try:
            dal.last_home_away_index_dict[team][1] = last_home_game.index[0]
        except: 
            pass
        if max(next_games_home_indexes) != dal.last_home_away_index_dict[team][1]:
            next_game_index = min(i for i in next_games_home_indexes[skip_n+average_N:] if i > last_home_game.index)
            next_game = test_df.loc[test_df.index == next_game_index]
            
            next_games_away_indexes = test_df.loc[test_df['Team_away'] == next_game['Team_away'].values[0]].index
            
            if next_game_index in next_games_away_indexes[skip_n+average_N:]:
                extract_and_predict(next_game)

    print(f'Evaluated samples: {len(predictions)}')

# Evaluate the predictions
data = {
    'index'             : evaluated_indexes,
    'Date'              : dates_list,
    'Team_away'         : away_teams_list,
    'Team_home'         : home_teams_list,
    'Predictions'       : predictions,
    'Winner'            : winners,
    'ModelProbability'  : model_prob,
    'ModelOdds'         : model_odds,
    'OddsHome'          : odds_home,
    'OddsAway'          : odds_away
}
nn_df = pd.DataFrame(data).sort_values('index')

print(nn_df)

# ----------------------------------- Kelly ------------------------------------------ #

merged_df = nn_df.assign(CombinedProb = (nn_df['ModelProbability']))
merged_df['CombinedProb'].loc[(merged_df['Predictions'] == 1)] = merged_df['ModelProbability']
merged_df['CombinedOdds'] = 1/merged_df['CombinedProb']

# Extract the rows where the model predicted the lowest odds between the two teams,
# i.e., where the team is the favorite to win. (Plus some margin)
correctly_pred_df = merged_df.loc[
        (merged_df['Predictions'] == merged_df['Winner']) &
        (
            ((merged_df['OddsHome'] > merged_df['OddsAway'] + margin) & (merged_df['Predictions'] == 1)) |
            ((merged_df['OddsAway'] > merged_df['OddsHome'] + margin) & (merged_df['Predictions'] == 0))
        )
    ]

wrongly_pred_df = merged_df.loc[
        (merged_df['Predictions'] != merged_df['Winner']) &
        (
            ((merged_df['OddsHome'] > merged_df['OddsAway'] + margin) & (merged_df['Predictions'] == 1)) |
            ((merged_df['OddsAway'] > merged_df['OddsHome'] + margin) & (merged_df['Predictions'] == 0))
        )
    ]

merged_df = pd.concat([correctly_pred_df, wrongly_pred_df], axis=0).sort_values('index').reset_index(drop=True)

# Compare Predictions and TrueValues
comparison_column = np.where(merged_df['Predictions'] == merged_df['Winner'], True, False)

# ---------- Kelly's Criterion ---------- #
# Bet a different fraction of the bankroll depending on odds
starting_bankroll = 100 # €
current_bankroll = starting_bankroll
bet_amount  = []
frac_bet    = [] # Percentage of bankroll bet
net_won     = []
bankroll    = []
for n, row in merged_df.iterrows():
    if row['CombinedProb'] < prob_threshold:
        frac_amount = 0
    elif row['Predictions'] == 0:
        frac_amount = ((row['CombinedProb'])*row['OddsHome']-1)/(row['OddsHome']-1)
    elif row['Predictions'] == 1:
        frac_amount = ((row['CombinedProb'])*row['OddsAway']-1)/(row['OddsAway']-1)
    
    if frac_amount > 0:
        # Bet a larger amount if probabilities exceed a threshold
        if (
            (row['CombinedProb'] >= prob_2x_bet and row['Predictions'] == 1) or
            (row['CombinedProb'] >= prob_2x_bet and row['Predictions'] == 0)
            ):
            if 2*betting_limit < current_bankroll:
                frac_amount = 2*betting_limit

            elif frac_amount > betting_limit and betting_limiter == True:
                frac_amount = betting_limit
        
        elif frac_amount > betting_limit and betting_limiter == True:
            frac_amount = betting_limit

        frac_bet.append(round(frac_amount, 2))
        
        # Max win is capped at 10000
        if ((current_bankroll * frac_amount * row['OddsHome']) and (row['Winner'] == 0)) > 10000:
            bet_amount.append(int(10000/row['OddsHome']))
        elif ((current_bankroll * frac_amount * row['OddsAway']) and (row['Winner'] == 1)) > 10000:
            bet_amount.append(int(10000/row['OddsAway']))
        # Min bet is 2€
        elif int(current_bankroll * frac_amount) >= 2:
            bet_amount.append(int(current_bankroll * frac_amount))
        elif int(current_bankroll * frac_amount) < 2:
            bet_amount.append(0)

        if row['Winner'] == 0:
            net_won.append(bet_amount[n] * row['OddsHome'] * (row['Predictions'] == row['Winner']) - bet_amount[n])
        else:
            net_won.append(bet_amount[n] * row['OddsAway'] * (row['Predictions'] == row['Winner']) - bet_amount[n])

        current_bankroll = current_bankroll + net_won[n]
        bankroll.append(current_bankroll)
    else:
        frac_bet.append(0)
        bet_amount.append(0)
        net_won.append(0)
        bankroll.append(current_bankroll)


merged_df['FractionBet'] = frac_bet
merged_df['BetAmount']   = bet_amount
merged_df['NetWon']      = net_won
merged_df['Bankroll']    = bankroll

# Evaluate the bankroll and the ROI
bet_df = merged_df.copy()
bet_df = bet_df.loc[bet_df['BetAmount'] > 0]
bet_df.reset_index(drop=True, inplace=True)
print(
    bet_df[
        [
            'index', 
            'Predictions', 
            'Winner', 
            'ModelProbability', 
            'CombinedProb', 
            'CombinedOdds', 
            'OddsHome', 
            'OddsAway', 
            'FractionBet',  
            'BetAmount',  
            'NetWon',  
            'Bankroll'
        ]
    ]
    )
print(f'Net return: {current_bankroll-starting_bankroll:.2f} €')
print(f'Net return per €: {(current_bankroll/starting_bankroll):.2f}')


# Calculate accuracy of predicted teams, when they were the favorite by a margin
correctly_amount = bet_df.loc[bet_df['Predictions'] == bet_df['Winner']].count()[0]
total_predicted = len(bet_df)

if correctly_pred_df.count()[0] != 0 and total_predicted != 0:
    accuracy = correctly_amount/total_predicted
    logger.info(f'Accuracy when team is favorite, loser odds are greater than winner ones + margin ({margin}): {accuracy:.3f}')
else:
    logger.info('Accuracy could not be computed. You may want to relax the conditions.')

# Confusion Matrix
conf_matrix = confusion_matrix(bet_df['Predictions'], bet_df['Winner'])
print(conf_matrix)

# Plot the results
ax = bet_df['Bankroll'].plot(grid=True)
ax.set_title('Bankroll versus number of games played')
ax.set_xlabel('Games played')
ax.set_ylabel('Bankroll (€)')
plt.show()


