import Helper
import pandas as pd

folder = 'past_data/2017_2018/'

df_new = pd.read_csv(folder + 'split_stats_per_game_2017.csv', index_col=False)
df_old = pd.read_csv('past_data/merged_seasons/2018_to_2021_Stats.csv', index_col=False)

df_new.drop('Date', axis=1, inplace=True)

df = pd.concat([df_new, df_old], axis=0)
df.to_csv('past_data/merged_seasons/2017_to_2021_Stats.csv', index=False)

