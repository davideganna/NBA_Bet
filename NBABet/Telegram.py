# --------------------- Telegram.py --------------------------------- #
# Allows the integration with Telegram Bot.
# ------------------------------------------------------------------- #
import requests

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
        text = "🏀 Tomorrow's Games: 🏀\n\n"
        for key, value in d.items():
            text = text + key + ' vs. ' + value + '\n\n'
        query = self.url + self.bot_token + '/sendMessage?' + self.chat_id + '&text=' + text
        requests.request("POST", query)

        