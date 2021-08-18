import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Models
import backtesting
import dicts_and_lists as dal
import logging, coloredlogs

pd.set_option('display.max_rows', 1000)

# ------ Logger ------- #
logger = logging.getLogger('test_models.py')
coloredlogs.install(level='INFO', logger=logger)

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
        next_games_away_indexes = df.loc[df['Team_away'] == away_team].index
        next_games_home_indexes = df.loc[df['Team_home'] == home_team].index
        next_away_indexes_reduced = [x for x in next_games_away_indexes if x < next_game.index[0]][-average_N:]
        next_home_indexes_reduced = [x for x in next_games_home_indexes if x < next_game.index[0]][-average_N:]

        # Extract last N games based on indexes
        last_N_games_away = df.iloc[next_away_indexes_reduced]
        last_N_games_home = df.iloc[next_home_indexes_reduced]

        # Concatenate the two teams with their average stats
        to_predict = pd.concat(
            [
                last_N_games_away[away_features].mean(), 
                last_N_games_home[home_features].mean()
            ],
            axis=0)[features]
        
        pred = int(clf.predict(to_predict.values.reshape(1,-1)))
        true_value = next_game['Winner'].values[0]
        predictions.append(pred)
        true_values.append(true_value)
        prob = clf.predict_proba(to_predict.values.reshape(1,-1))
        model_prob.append(max(prob[0]))
        model_odds.append(1/max(prob[0]))
        odds_winner.append(next_game['OddsWinner'].values[0])
        odds_loser.append(next_game['OddsLoser'].values[0])
        dates_list.append(next_game['Date'].values[0])
        home_teams_list.append(dal.teams_dict[home_team])
        away_teams_list.append(dal.teams_dict[away_team])

# Only the most significant features will be considered
away_features = Models.away_features
home_features = Models.home_features
features = Models.features

### Test the Classification model based on the mean of the last average_N games ###
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
evaluated_indexes = []

# Backtest on the 2020/2021 Season
df = pd.read_csv('past_data/2020_2021/split_stats_per_game.csv')

# Maximum allowed average_N: 35
average_N = 5
skip_n = 0
print(f'Stats averaged from {average_N} games, first {skip_n} games are skipped.')

for skip_n_games in range(skip_n, 50-average_N):
    last_N_games_away, last_N_games_home = backtesting.get_first_N_games(df, average_N, skip_n_games)
    # Get next game based on next_game_index
    for team in dal.teams:
        # Find all games where "team" plays away
        next_games_away_indexes = df.loc[df['Team_away'] == team].index
        last_away_game = last_N_games_away[dal.teams_to_int[team]][-1:]
        # Check if there are more games past the current index 
        try:
            dal.last_home_away_index_dict[team][0] = last_away_game.index[0]
        except: 
            pass
        if max(next_games_away_indexes) != dal.last_home_away_index_dict[team][0]:
            next_game_index = min(i for i in next_games_away_indexes[skip_n+average_N:] if i > last_away_game.index)
            next_game = df.loc[df.index == next_game_index]

            next_games_home_indexes = df.loc[df['Team_home'] == next_game['Team_home'].values[0]].index

            if next_game_index in next_games_home_indexes[skip_n+average_N:]:
                extract_and_predict(next_game)

        # Find all games where "team" plays home
        next_games_home_indexes = df.loc[df['Team_home'] == team].index
        last_home_game = last_N_games_home[dal.teams_to_int[team]][-1:]
        # Check if there are more games past the current index 
        try:
            dal.last_home_away_index_dict[team][1] = last_home_game.index[0]
        except: 
            pass
        if max(next_games_home_indexes) != dal.last_home_away_index_dict[team][1]:
            next_game_index = min(i for i in next_games_home_indexes[skip_n+average_N:] if i > last_home_game.index)
            next_game = df.loc[df.index == next_game_index]
            
            next_games_away_indexes = df.loc[df['Team_away'] == next_game['Team_away'].values[0]].index
            
            if next_game_index in next_games_away_indexes[skip_n+average_N:]:
                extract_and_predict(next_game)

    print(f'Evaluated samples: {len(predictions)}')

# Evaluate the predictions
data = {
    'index'             : evaluated_indexes,
    'Date'              : dates_list,
    'AwayTeam'          : away_teams_list,
    'HomeTeam'          : home_teams_list,
    'Predictions'       : predictions,
    'TrueValues'        : true_values,
    'ModelProbability'  : model_prob,
    'ModelOdds'         : model_odds,
    'OddsWinner'        : odds_winner,
    'OddsLoser'         : odds_loser
}
ev_df = pd.DataFrame(data).sort_values('index')

# Calculate accuracy of predicted teams, when they were the favorite by a margin
margin = 0
prob_limit = 0.7
correctly_predicted_amount = ev_df.loc[
    (ev_df['Predictions'] == ev_df['TrueValues']) &
    ((ev_df['OddsLoser'] > ev_df['OddsWinner'] + margin)) &
    (ev_df['OddsWinner'] >= ev_df['ModelOdds']) &
    (ev_df['ModelProbability'] >= prob_limit)
    ].count()

wrongly_predicted_amount = ev_df.loc[
    (ev_df['Predictions'] != ev_df['TrueValues']) &
    ((ev_df['OddsWinner'] > ev_df['OddsLoser'] + margin)) &
    (ev_df['OddsLoser'] >= ev_df['ModelOdds']) &
    (ev_df['ModelProbability'] >= prob_limit)
    ].count()

total_predicted = correctly_predicted_amount[0] + wrongly_predicted_amount[0]

if correctly_predicted_amount[0] != 0 and total_predicted != 0:
    accuracy = correctly_predicted_amount[0]/total_predicted
    logger.info(f'Accuracy when team is favorite, loser odds are greater than winner ones + margin ({margin}) and model probability is > {prob_limit}: {accuracy:.3f}')
else:
    logger.info('Accuracy could not be computed. You may try to relax the conditions (margin and/or prob_limit).')

# Extract the rows where the model predicted the lowest odds between the two teams,
# i.e., where the team is the favorite to win. (Plus some margin)
correctly_pred_df = ev_df.loc[
    (ev_df['Predictions'] == ev_df['TrueValues']) &
    ((ev_df['OddsLoser'] > ev_df['OddsWinner'] + margin)) &
    (ev_df['OddsWinner'] >= ev_df['ModelOdds']) &
    (ev_df['ModelProbability'] >= prob_limit) 
]
wrongly_pred_df = ev_df.loc[
    (ev_df['Predictions'] != ev_df['TrueValues']) &
    ((ev_df['OddsWinner'] > ev_df['OddsLoser'] + margin)) &
    (ev_df['OddsLoser'] >= ev_df['ModelOdds']) &
    (ev_df['ModelProbability'] >= prob_limit)
]

ev_df = pd.concat([correctly_pred_df, wrongly_pred_df], axis=0).sort_values('index').reset_index(drop=True)

# Compare Predictions and TrueValues
comparison_column = np.where(ev_df['Predictions'] == ev_df['TrueValues'], True, False)

# Kelly's criterion: bet a different fraction of the bankroll depending on odds
starting_bankroll = 100 # €
current_bankroll = starting_bankroll
bet_amount  = []
frac_bet    = [] # Percentage of bankroll bet
net_won     = []
bankroll    = []
for n, row in ev_df.iterrows():
    frac_amount = (row['ModelProbability']*row['OddsWinner']-1)/(row['OddsWinner']-1)
    
    if frac_amount > 0:
        # Limit the portion of bankroll to bet
        if frac_amount > 0.2 and current_bankroll < 2*starting_bankroll:
            frac_amount = 0.2

        frac_bet.append(round(frac_amount, 2))
        
        # Max win is capped at 10000
        if (current_bankroll * frac_amount * row['OddsWinner']) > 10000:
            bet_amount.append(int(10000/row['OddsWinner']))
        else:
            bet_amount.append(int(current_bankroll * frac_amount))

        net_won.append(bet_amount[n] * row['OddsWinner'] * (row['Predictions'] == row['TrueValues']) - bet_amount[n])
        current_bankroll = current_bankroll + net_won[n]
        bankroll.append(current_bankroll)
    else:
        frac_bet.append(0)
        bet_amount.append(0)
        net_won.append(0)
        bankroll.append(current_bankroll)


ev_df['FractionBet'] = frac_bet
ev_df['BetAmount']   = bet_amount
ev_df['NetWon']      = net_won
ev_df['Bankroll']    = bankroll

# Evaluate the bankroll and the ROI
print(ev_df)
print(f'Net return: {current_bankroll-starting_bankroll:.2f} €')
print(f'Net return per €: {(current_bankroll/starting_bankroll)-1:.2f}')

# Plot the results
ax = ev_df['Bankroll'].plot(grid=True)
ax.set_title('Bankroll versus number of games played')
ax.set_xlabel('Games played')
ax.set_ylabel('Bankroll (€)')
plt.show()