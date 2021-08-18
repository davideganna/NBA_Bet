import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Helper

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

for df in df_list:
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
        df.to_csv('past_data/merged_seasons/2017_to_2021_Stats.csv', index=False)

corr_matrix = df.corr()
print((corr_matrix['Winner'].sort_values(ascending=False)))
