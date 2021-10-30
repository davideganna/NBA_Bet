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
logger = logging.getLogger('script_testing.py')
coloredlogs.install(level='INFO', logger=logger)

# ---------- TESTS ---------- #
Helper.build_elo_csv()

""" 
merged_21 = pd.read_csv('past_data/merged_seasons/2017_to_2021_Stats.csv')
corr_matrix = merged_21.corr()
print(corr_matrix['Winner']) """




""" df_2017 = pd.read_csv('past_data/2017_2018/split_stats_per_game_2017.csv')
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

for df in df_list:
    df_x = Helper.add_features_to_df(df)
    if df is df_2017:
        df_x.to_csv('past_data/2017_2018/split_stats_per_game_2017.csv', index=False)
    elif df is df_2018:
        df_x.to_csv('past_data/2018_2019/split_stats_per_game_2018.csv', index=False)
    elif df is df_2019:
        df_x.to_csv('past_data/2019_2020/split_stats_per_game_2019.csv', index=False)
    elif df is df_2020:
        df_x.to_csv('past_data/2020_2021/split_stats_per_game.csv', index=False)
    elif df is merged_19:
        df_x.to_csv('past_data/merged_seasons/2017_to_2019_Stats.csv', index=False)
    elif df is merged_20:
        df_x.to_csv('past_data/merged_seasons/2017_to_2020_Stats.csv', index=False)
    elif df is merged_21:
        df_x.to_csv('past_data/merged_seasons/2017_to_2021_Stats.csv', index=False) """





""" # Evaluate odds per range
df_1_11 = df.loc[(df['OddsHome'] <= 1.1) | (df['OddsAway'] <= 1.1)].count()

correctly_1_11 = df.loc[
    (df['Winner'] == 0) & (df['OddsHome'] <= 1.1) |
    (df['Winner'] == 1) & (df['OddsAway'] <= 1.1)
].count()

print(f'Accuracy 1.0 to 1.1: {(correctly_1_11[0]/df_1_11[0]):.3f}')

df_1_11 = df.loc[
    (1.1 < df['OddsHome']) & (df['OddsHome'] <= 1.2) | 
    (1.1 < df['OddsAway']) & (df['OddsAway'] <= 1.2)
    ].count()

correctly_1_11 = df.loc[
    (df['Winner'] == 0) & (1.1 < df['OddsHome']) & (df['OddsHome'] <= 1.2) |
    (df['Winner'] == 1) & (1.1 < df['OddsAway']) & (df['OddsAway'] <= 1.2)
].count()

print(f'Accuracy 1.1 to 1.2: {(correctly_1_11[0]/df_1_11[0]):.3f}')

df_1_11 = df.loc[
    (1.2 < df['OddsHome']) & (df['OddsHome'] <= 1.3) | 
    (1.2 < df['OddsAway']) & (df['OddsAway'] <= 1.3)
    ].count()

correctly_1_11 = df.loc[
    (df['Winner'] == 0) & (1.2 < df['OddsHome']) & (df['OddsHome'] <= 1.3) |
    (df['Winner'] == 1) & (1.2 < df['OddsAway']) & (df['OddsAway'] <= 1.3)
].count()

print(f'Accuracy 1.2 to 1.3: {(correctly_1_11[0]/df_1_11[0]):.3f}')

df_1_11 = df.loc[
    (1.3 < df['OddsHome']) & (df['OddsHome'] <= 1.4) | 
    (1.3 < df['OddsAway']) & (df['OddsAway'] <= 1.4)
    ].count()

correctly_1_11 = df.loc[
    (df['Winner'] == 0) & (1.3 < df['OddsHome']) & (df['OddsHome'] <= 1.4) |
    (df['Winner'] == 1) & (1.3 < df['OddsAway']) & (df['OddsAway'] <= 1.4)
].count()

print(f'Accuracy 1.3 to 1.4: {(correctly_1_11[0]/df_1_11[0]):.3f}')

df_1_11 = df.loc[
    (1.4 < df['OddsHome']) & (df['OddsHome'] <= 1.5) | 
    (1.4 < df['OddsAway']) & (df['OddsAway'] <= 1.5)
    ].count()

correctly_1_11 = df.loc[
    (df['Winner'] == 0) & (1.4 < df['OddsHome']) & (df['OddsHome'] <= 1.5) |
    (df['Winner'] == 1) & (1.4 < df['OddsAway']) & (df['OddsAway'] <= 1.5)
].count()

print(f'Accuracy 1.4 to 1.5: {(correctly_1_11[0]/df_1_11[0]):.3f}')

df_1_11 = df.loc[
    (1.5 < df['OddsHome']) & (df['OddsHome'] <= 1.6) | 
    (1.5 < df['OddsAway']) & (df['OddsAway'] <= 1.6)
    ].count()

correctly_1_11 = df.loc[
    (df['Winner'] == 0) & (1.5 < df['OddsHome']) & (df['OddsHome'] <= 1.6) |
    (df['Winner'] == 1) & (1.5 < df['OddsAway']) & (df['OddsAway'] <= 1.6)
].count()

print(f'Accuracy 1.5 to 1.6: {(correctly_1_11[0]/df_1_11[0]):.3f}')

df_1_11 = df.loc[
    (1.6 < df['OddsHome']) & (df['OddsHome'] <= 1.7) | 
    (1.6 < df['OddsAway']) & (df['OddsAway'] <= 1.7)
    ].count()

correctly_1_11 = df.loc[
    (df['Winner'] == 0) & (1.6 < df['OddsHome']) & (df['OddsHome'] <= 1.7) |
    (df['Winner'] == 1) & (1.6 < df['OddsAway']) & (df['OddsAway'] <= 1.7)
].count()

print(f'Accuracy 1.6 to 1.7: {(correctly_1_11[0]/df_1_11[0]):.3f}')

df_1_11 = df.loc[
    (1.7 < df['OddsHome']) & (df['OddsHome'] <= 1.8) | 
    (1.7 < df['OddsAway']) & (df['OddsAway'] <= 1.8)
    ].count()

correctly_1_11 = df.loc[
    (df['Winner'] == 0) & (1.7 < df['OddsHome']) & (df['OddsHome'] <= 1.8) |
    (df['Winner'] == 1) & (1.7 < df['OddsAway']) & (df['OddsAway'] <= 1.8)
].count()

print(f'Accuracy 1.7 to 1.8: {(correctly_1_11[0]/df_1_11[0]):.3f}')

df_1_11 = df.loc[
    (1.8 < df['OddsHome']) & (df['OddsHome'] <= 1.9) | 
    (1.8 < df['OddsAway']) & (df['OddsAway'] <= 1.9)
    ].count()

correctly_1_11 = df.loc[
    (df['Winner'] == 0) & (1.8 < df['OddsHome']) & (df['OddsHome'] <= 1.9) |
    (df['Winner'] == 1) & (1.8 < df['OddsAway']) & (df['OddsAway'] <= 1.9)
].count()

print(f'Accuracy 1.8 to 1.9: {(correctly_1_11[0]/df_1_11[0]):.3f}')

df_1_11 = df.loc[
    (1.9 < df['OddsHome']) & (df['OddsHome'] <= 2) | 
    (1.9 < df['OddsAway']) & (df['OddsAway'] <= 2)
    ].count()

correctly_1_11 = df.loc[
    (df['Winner'] == 0) & (1.9 < df['OddsHome']) & (df['OddsHome'] <= 2) |
    (df['Winner'] == 1) & (1.9 < df['OddsAway']) & (df['OddsAway'] <= 2)
].count()

print(f'Accuracy 1.9 to 2: {(correctly_1_11[0]/df_1_11[0]):.3f}')


print(df.loc[
    (1.9 < df['OddsHome']) & (df['OddsHome'] <= 2) | 
    (1.9 < df['OddsAway']) & (df['OddsAway'] <= 2)
    ][
            [
                'Date', 
                'Team_away', 
                'Team_home', 
                'Predictions', 
                'Winner', 
                'OddsAway_Elo', 
                'OddsHome_Elo', 
                'ModelProb_Away',
                'ModelProb_Home',
                'OddsAway', 
                'OddsHome', 
            ]
        ]) """