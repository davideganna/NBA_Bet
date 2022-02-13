import pandas as pd

df_2017 = pd.read_csv('past_data/2017_2018/stats_per_game_2017.csv')
df_2018 = pd.read_csv('past_data/2018_2019/stats_per_game_2018.csv')
df_2019 = pd.read_csv('past_data/2019_2020/stats_per_game_2019.csv')
df_2020 = pd.read_csv('past_data/2020_2021/stats_per_game.csv')
df_2021 = pd.read_csv('past_data/2021_2022/stats_per_game.csv')

df = pd.concat([df_2017, df_2018, df_2019, df_2020, df_2021], ignore_index=True)

tied_games_amount = len(df.loc[df['MP']>240])

print(f'Percentage of tied games: {tied_games_amount/len(df):.3f}')
