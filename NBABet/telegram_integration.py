# --------------------- telegram_integration.py --------------------- #
# Allows the integration with Telegram Bot.
# ------------------------------------------------------------------- #
import sys
import platform

plt = platform.system()

# Get Bot's directory based on the OS where the script is run
with open('secrets/os_paths', 'r') as f:
    if plt == "Windows":
        os_path = f.readlines()[0].partition(': ')[2].partition('\n')[0]
    elif plt == "Linux":
        os_path = f.readlines()[1].partition(': ')[2].partition('\n')[0]

sys.path.insert(0, os_path)

import telegram_bot # type: ignore "Do not remove this comment"

# Functions

def set_bankroll():
    telegram_bot.set_bankroll()

def send_predictions():
    telegram_bot.send_predictions()

def test_br():
    telegram_bot.test_br()
