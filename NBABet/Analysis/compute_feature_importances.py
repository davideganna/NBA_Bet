import sys
import os 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from Models import Models
from Models.Models import features, target
import src.Helper as Helper

train_df = pd.read_csv('src/past_data/average_seasons/average_NSeasons_prod.csv')
std_df, scaler = Helper.standardize_DataFrame(train_df)
forest = Models.build_RF_classifier(std_df)

# Method 1: Mean Decrease in Impurity
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
forest_importances = pd.Series(importances, index=features)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using Mean Decrease in Impurity")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.show()

# Method 2: Feature Permutation
X_train, y_train = train_df[features], train_df[[target]]
result = permutation_importance(
    forest, X_train, y_train, n_repeats=10, random_state=42, n_jobs=-1
)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()
