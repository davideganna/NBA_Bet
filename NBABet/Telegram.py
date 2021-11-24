# --------------------- Telegram.py --------------------------------- #
# Allows the integration with Telegram Bot.
# ------------------------------------------------------------------- #
from numpy.core.fromnumeric import around
import requests
import Elo
from Models import Models
import Helper
import pandas as pd
import numpy as np

class TelegramBot():
    """
    Allows integration with Telegram Bot.
    """
    def __init__(self):
        self.url = 'https://api.telegram.org/'
        with open('secrets/telegram_secrets') as f:
            lines = f.readlines()
            self.bot_token = lines[0].strip()
            self.chat_id = lines[1].strip()
    

    def send_message(self, d:dict):
        df = pd.read_csv('past_data/2021_2022/split_stats_per_game.csv')
        df = Helper.add_features_to_df(df)

        n = 3
        
        train_df = pd.read_csv('past_data/average_seasons/average_N_4Seasons.csv')
        clf = Models.build_RF_classifier(train_df)

        text = "🏀 Tonight's Games: Home vs. Away 🏀\n\n"
        for home, away in d.items():
            last_N_games_away = df.loc[df['Team_away'] == away].tail(n)
            last_N_games_home = df.loc[df['Team_home'] == home].tail(n)

            to_predict = pd.concat(
                [
                    last_N_games_away[Models.away_features].mean(), 
                    last_N_games_home[Models.home_features].mean()
                ],
                axis=0)[Models.features]

            prob_home_rf, prob_away_rf = clf.predict_proba(to_predict.values.reshape(1,-1))[0]

            prob_away_elo, prob_home_elo = Elo.get_probas(away, home)

            if ((prob_home_rf > 0.5) and (prob_home_elo > 0.5)):
                prob_home = str(around(np.maximum(prob_home_rf, prob_home_elo), decimals=3))
                odds_home = str(around(1/float(prob_home), decimals=2))
                text = text + home + '(' + prob_home + ' --> ' + odds_home + ') vs. ' + away + '\n\n'

            if ((prob_away_rf > 0.5) and (prob_away_elo > 0.5)):
                prob_away = str(around(np.maximum(prob_away_rf, prob_away_elo), decimals=3))
                odds_away = str(around(1/float(prob_away), decimals=2))
                text = text + home + ' vs. ' + away + '(' + prob_away + ' --> ' + odds_away + ')\n\n'

        query = self.url + self.bot_token + '/sendMessage?' + self.chat_id + '&text=' + text
        requests.request("POST", query)

        