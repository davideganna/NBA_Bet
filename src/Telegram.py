# --------------------- Telegram.py --------------------------------- #
# Allows the integration with Telegram Bot.
# ------------------------------------------------------------------- #
from numpy.core.fromnumeric import around, std
import requests
import src.elo as elo
from Models import Models
import src.Helper as Helper
import pandas as pd
import numpy as np
import yaml


class TelegramBot:
    """
    Allows integration with the Telegram Bot.
    """

    def __init__(self):
        self.url = "https://api.telegram.org/"
        with open("secrets/telegram_secrets") as f:
            lines = f.readlines()
            self.bot_token = lines[0].strip()
            self.chat_id = lines[1].strip()
        with open("src/configs/main_conf.yaml") as f:
            self.config = yaml.safe_load(f)

    def send_message(self, d: dict):
        # Read up-to-date data
        _path = f"src/past_data/{self.config['years']}/split_stats_per_game.csv"
        df = pd.read_csv(_path)
        df = Helper.add_features_to_df(df)

        n = 3

        train_df = pd.read_csv(
            "src/past_data/average_seasons/average_NSeasons_prod.csv"
        )
        # Standardize the DataFrame
        std_df, scaler = Helper.standardize_DataFrame(train_df)

        clf = Models.build_RF_classifier(std_df)

        text = "🏀 Tonight's Games: Home vs. Away 🏀\n\n"
        for home, away in d.items():
            last_N_games_away = df.loc[df["Team_away"] == away].tail(n)
            last_N_games_home = df.loc[df["Team_home"] == home].tail(n)

            to_predict = pd.concat(
                [
                    last_N_games_away[Models.away_features].mean(),
                    last_N_games_home[Models.home_features].mean(),
                ],
                axis=0,
            )[Models.features]

            prob_home_rf, prob_away_rf = clf.predict_proba(
                scaler.transform(to_predict.values.reshape(1, -1))
            )[0]

            prob_away_elo, prob_home_elo = elo.get_probas(away, home)

            if (prob_home_rf > 0.5) and (prob_home_elo > 0.5):
                prob_home = str(around((prob_home_rf + prob_home_elo) / 2, decimals=3))
                odds_home = str(around(1 / float(prob_home), decimals=2))
                if float(prob_home) >= 0.6:
                    text = (
                        text
                        + home
                        + "("
                        + prob_home
                        + " --> "
                        + odds_home
                        + ") vs. "
                        + away
                        + "\n\
                        RF Prob.: "
                        + str(around(prob_home_rf, decimals=3))
                        + "\n\
                        Elo Prob.: "
                        + str(around(prob_home_elo, decimals=3))
                        + "\n\n"
                    )

            if (prob_away_rf > 0.5) and (prob_away_elo > 0.5):
                prob_away = str(around((prob_away_rf + prob_away_elo) / 2, decimals=3))
                odds_away = str(around(1 / float(prob_away), decimals=2))
                if float(prob_away) >= 0.6:
                    text = (
                        text
                        + home
                        + " vs. "
                        + away
                        + "("
                        + prob_away
                        + " --> "
                        + odds_away
                        + ")"
                        + "\n\
                        RF Prob.: "
                        + str(around(prob_away_rf, decimals=3))
                        + "\n\
                        Elo Prob.: "
                        + str(around(prob_away_elo, decimals=3))
                        + "\n\n"
                    )

        query = (
            self.url + self.bot_token + "/sendMessage?" + self.chat_id + "&text=" + text
        )
        requests.request("POST", query)
