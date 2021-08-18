import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Helper

csv_name = 'past_data/merged_seasons/2017_to_2021_Stats.csv'


df = pd.read_csv(csv_name)

df = Helper.add_features_to_df(df)

df.to_csv(csv_name, index=False)

corr_matrix = df.corr()
print((corr_matrix['Winner'].sort_values(ascending=False)))
