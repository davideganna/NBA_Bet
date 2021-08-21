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