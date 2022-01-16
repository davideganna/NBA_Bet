import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
from sklearn.preprocessing import StandardScaler

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

def standardize_DataFrame(df:DataFrame):
    # Standardize the DataFrame
    x = df.loc[:, features].values
    y = df.loc[:, ['Winner']].values

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    std_df = pd.DataFrame(x, columns=features)
    std_df = pd.concat([std_df, df['Winner'].reset_index(drop=True)], axis=1)
    return std_df, scaler

train_df = pd.read_csv('../past_data/average_seasons/average_N_4Seasons.csv')
train_df, scaler = standardize_DataFrame(train_df)

df = pd.read_csv('../past_data/average2020.csv')
x = df.loc[:, features].values
x = scaler.transform(x)
std_df = pd.DataFrame(x, columns=features)
valid_df = pd.concat([std_df, df['Winner'].reset_index(drop=True)], axis=1)

print(valid_df)

# Define the Training dataset
X_train = std_df[features]
y_train = std_df['Winner']

# Define the model
model = keras.Sequential([
    layers.Dense(4, activation='relu', input_shape=[14]),
    layers.Dense(4, activation='relu'),    
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
    verbose=0, # hide the output because we have so many epochs
)