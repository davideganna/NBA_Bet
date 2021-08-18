import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Helper

csv_name = 'past_data/2020_2021/split_stats_per_game.csv'

df = pd.read_csv(csv_name)

df = Helper.add_features_to_df(df)

df.to_csv(csv_name, index=False)

corr_matrix = df.corr()
print((corr_matrix['Winner'].sort_values(ascending=False)))
