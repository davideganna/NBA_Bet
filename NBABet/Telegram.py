# --------------------- Telegram.py --------------------------------- #
# Allows the integration with Telegram Bot.
# ------------------------------------------------------------------- #
import requests
import Elo
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
        text = "🏀 Tomorrow's Games: Home vs. Away 🏀\n\n"
        for home, away in d.items():
            prob_away, prob_home = Elo.get_probas(away, home)
            prob_away, prob_home = np.around(prob_away, decimals=3), np.around(prob_home, decimals=3), 
            text = text + home + '(' + str(prob_home) + ') vs. ' + away + '(' + str(prob_away) + ')\n\n'
        query = self.url + self.bot_token + '/sendMessage?' + self.chat_id + '&text=' + text
        requests.request("POST", query)

        