# --------------------- telegram.py --------------------------------- #
# Allows the integration with telegram Bot.
# ------------------------------------------------------------------- #
from numpy.core.fromnumeric import around, std
import requests
import src.elo as elo
from src.models import models
import src.helper as helper
import pandas as pd
import numpy as np
import yaml


class telegramBot:
    """
    Allows integration with the telegram Bot.
    """

    def __init__(self):
        self.url = "https://api.telegram.org/"
        with open("secrets/telegram_secrets") as f:
            lines = f.readlines()
            self.bot_token = lines[0].strip()
            self.chat_id = lines[1].strip()
        with open("src/configs/main_conf.yaml") as f:
            self.config = yaml.safe_load(f)

    def send_message(self, next_games: dict, team_to_prob: dict):
        text = "🏀 Tonight's Games: Away vs. Home 🏀\n\n"
        for away_team, home_team in next_games.items():
            prob_home = str(around(team_to_prob[home_team], decimals=3))
            odds_home = str(around(1 / float(prob_home), decimals=2))
            prob_away = str(around(team_to_prob[away_team], decimals=3))
            odds_away = str(around(1 / float(prob_away), decimals=2))

            text = (
                text
                + away_team
                + "("
                + prob_away
                + " --> "
                + odds_away
                + ") vs. "
                + home_team
                + "("
                + prob_home
                + " --> "
                + odds_home
                + ").\n\n"
            )

        query = (
            self.url + self.bot_token + "/sendMessage?" + self.chat_id + "&text=" + text
        )
        requests.request("POST", query)