import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Helper
import Elo
import dicts_and_lists as dal
import logging, coloredlogs
from sklearn.metrics import confusion_matrix

pd.set_option('display.max_rows', 1000)

# ------ Logger ------- #
logger = logging.getLogger('test_models.py')
coloredlogs.install(level='INFO', logger=logger)

df_2017 = pd.read_csv('past_data/2017_2018/split_stats_per_game_2017.csv')
df_2018 = pd.read_csv('past_data/2018_2019/split_stats_per_game_2018.csv')
df_2019 = pd.read_csv('past_data/2019_2020/split_stats_per_game_2019.csv')
df_2020 = pd.read_csv('past_data/2020_2021/split_stats_per_game.csv')
merged_19 = pd.read_csv('past_data/merged_seasons/2017_to_2019_Stats.csv')
merged_20 = pd.read_csv('past_data/merged_seasons/2017_to_2020_Stats.csv')
merged_21 = pd.read_csv('past_data/merged_seasons/2017_to_2021_Stats.csv')

df_list = [
    df_2017,
    df_2018,
    df_2019,
    df_2020,
    merged_19,
    merged_20,
    merged_21
]

""" for df in df_list:
    df = Helper.add_features_to_df(df)
    if df is df_2017:
        df.to_csv('past_data/2017_2018/split_stats_per_game_2017.csv', index=False)
    elif df is df_2018:
        df.to_csv('past_data/2018_2019/split_stats_per_game_2018.csv', index=False)
    elif df is df_2019:
        df.to_csv('past_data/2019_2020/split_stats_per_game_2019.csv', index=False)
    elif df is df_2020:
        df.to_csv('past_data/2020_2021/split_stats_per_game.csv', index=False)
    elif df is merged_19:
        df.to_csv('past_data/merged_seasons/2017_to_2019_Stats.csv', index=False)
    elif df is merged_20:
        df.to_csv('past_data/merged_seasons/2017_to_2020_Stats.csv', index=False)
    elif df is merged_21:
        df.to_csv('past_data/merged_seasons/2017_to_2021_Stats.csv', index=False) """


#

df = df_2020
df[['Elo_home', 'Elo_away']] = np.nan

for team in dal.teams:
    dal.current_team_Elo[team] = 1500

rows = []
prob_away = []
prob_home = []
predictions = []

for _n, _row in df.iterrows():
    probas = Elo.get_probas(_row['Team_away'], _row['Team_home'])
    prob_away.append(probas[0])
    prob_home.append(probas[1])
    rows.append(Elo.update(_row, _row['Team_away'], _row['Team_home'], _row['Winner']))

# Create the DataFrame
df = pd.DataFrame(rows)

df['ModelProb_Away'] = prob_away
df['ModelProb_Home'] = prob_home 

df['OddsAway_Elo'] = 1/df['ModelProb_Away']
df['OddsHome_Elo'] = 1/df['ModelProb_Home']

for _n, _row in df.iterrows():
    # Predict new results based on Elo rating
    if (_row['ModelProb_Away'] > _row['ModelProb_Home']):
        predictions.append(1)
    else:
        predictions.append(0)

df['Predictions'] = predictions

print(df.head(30))

ev_df = df.copy()

# Hyperparameters
margin = 0
prob_limit = 0.7
betting_limiter = True

# Calculate accuracy of predicted teams, when they were the favorite by a margin
correctly_predicted_amount = ev_df.loc[
    (ev_df['Predictions'] == ev_df['Winner']) &
    (
        ((ev_df['OddsHome'] > ev_df['OddsAway'] + margin) & (ev_df['Predictions'] == 1)) |
        ((ev_df['OddsAway'] > ev_df['OddsHome'] + margin) & (ev_df['Predictions'] == 0))
    ) &
    (
        ((ev_df['OddsHome'] >= ev_df['OddsHome_Elo']) & (ev_df['Predictions'] == 0)) |
        ((ev_df['OddsAway'] >= ev_df['OddsAway_Elo']) & (ev_df['Predictions'] == 1))
    ) 
    ].count()

wrongly_predicted_amount = ev_df.loc[
    (ev_df['Predictions'] != ev_df['Winner']) &
    (
        ((ev_df['OddsHome'] > ev_df['OddsAway'] + margin) & (ev_df['Predictions'] == 1)) |
        ((ev_df['OddsAway'] > ev_df['OddsHome'] + margin) & (ev_df['Predictions'] == 0))
    ) &
    (
        ((ev_df['OddsHome'] >= ev_df['OddsHome_Elo']) & (ev_df['Predictions'] == 0)) |
        ((ev_df['OddsAway'] >= ev_df['OddsAway_Elo']) & (ev_df['Predictions'] == 1))
    ) 
    ].count()

total_predicted = correctly_predicted_amount[0] + wrongly_predicted_amount[0]

if correctly_predicted_amount[0] != 0 and total_predicted != 0:
    accuracy = correctly_predicted_amount[0]/total_predicted
    logger.info(f'Accuracy when team is favorite, loser odds are greater than winner ones + margin ({margin}) and model probability is > {prob_limit}: {accuracy:.3f}')
else:
    logger.info('Accuracy could not be computed. You may try to relax the conditions (margin and/or prob_limit).')


correctly_pred_df = ev_df.loc[
    (ev_df['Predictions'] == ev_df['Winner']) &
    (
        ((ev_df['OddsHome'] > ev_df['OddsAway'] + margin) & (ev_df['Predictions'] == 1)) |
        ((ev_df['OddsAway'] > ev_df['OddsHome'] + margin) & (ev_df['Predictions'] == 0))
    ) &
    (
        ((ev_df['OddsHome'] >= ev_df['OddsHome_Elo']) & (ev_df['Predictions'] == 0)) |
        ((ev_df['OddsAway'] >= ev_df['OddsAway_Elo']) & (ev_df['Predictions'] == 1))
    ) 
    ]

wrongly_pred_df = ev_df.loc[
    (ev_df['Predictions'] != ev_df['Winner']) &
    (
        ((ev_df['OddsHome'] > ev_df['OddsAway'] + margin) & (ev_df['Predictions'] == 1)) |
        ((ev_df['OddsAway'] > ev_df['OddsHome'] + margin) & (ev_df['Predictions'] == 0))
    ) &
    (
        ((ev_df['OddsHome'] >= ev_df['OddsHome_Elo']) & (ev_df['Predictions'] == 0)) |
        ((ev_df['OddsAway'] >= ev_df['OddsAway_Elo']) & (ev_df['Predictions'] == 1))
    ) 
    ]

ev_df = pd.concat([correctly_pred_df, wrongly_pred_df], axis=0).reset_index(drop=True)

# Compare Predictions and TrueValues
comparison_column = np.where(ev_df['Predictions'] == ev_df['Winner'], True, False)

# Kelly's criterion: bet a different fraction of the bankroll depending on odds
starting_bankroll = 100 # €
current_bankroll = starting_bankroll
bet_amount  = []
frac_bet    = [] # Percentage of bankroll bet
net_won     = []
bankroll    = []

for n, row in ev_df.iterrows():
    if row['Winner'] == 0:
        frac_amount = (row['ModelProb_Home']*row['OddsHome']-1)/(row['OddsHome']-1)
    else:
        frac_amount = (row['ModelProb_Away']*row['OddsAway']-1)/(row['OddsAway']-1)
    
    if frac_amount > 0:
        # Limit the portion of bankroll to bet
        if frac_amount > 0.2 and current_bankroll < 2*starting_bankroll and betting_limiter == True:
            frac_amount = 0.2

        frac_bet.append(round(frac_amount, 2))
        
        # Max win is capped at 10000
        if ((current_bankroll * frac_amount * row['OddsHome']) and (row['Winner'] == 0)) > 10000:
            bet_amount.append(int(10000/row['OddsHome']))
        elif ((current_bankroll * frac_amount * row['OddsAway']) and (row['Winner'] == 1)) > 10000:
            bet_amount.append(int(10000/row['OddsAway']))
        else:
            bet_amount.append(int(current_bankroll * frac_amount))

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

# Confusion Matrix
conf_matrix = confusion_matrix(ev_df['Predictions'], ev_df['Winner'])[0]
print(conf_matrix)
