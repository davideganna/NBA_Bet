#    888b    888 888888b.          d8888     888888b.            888   
#    8888b   888 888  "88b        d88888     888  "88b           888   
#    88888b  888 888  .88P       d88P888     888  .88P           888   
#    888Y88b 888 8888888K.      d88P 888     8888888K.   .d88b.  888888
#    888 Y88b888 888  "Y88b    d88P  888     888  "Y88b d8P  Y8b 888   
#    888  Y88888 888    888   d88P   888     888    888 88888888 888   
#    888   Y8888 888   d88P  d8888888888     888   d88P Y8b.     Y88b. 
#    888    Y888 8888888P"  d88P     888     8888888P"   "Y8888   "Y888

from apscheduler.schedulers.background import BackgroundScheduler
import Helper
import logging, coloredlogs
import pandas as pd
from Api import Api
from Telegram import TelegramBot

# ----- Scheduler ----- #
sched = BackgroundScheduler(daemon=True)
sched.add_job(lambda: Helper.check_df(),'interval', hours=24)
sched.add_job(lambda: TelegramBot().send_message(next_games),'interval', hours=24)

# ------ Logger ------- #
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO')

#-------- Main -------- #
folder = 'past_data/2021_2022/' # Specify the current NBA season to save the .csv datasets.

try:
    season_df = pd.read_csv(folder + '2021_2022_season.csv')
except:
    logger.error('Program could not access the .csv datasets. Be sure to run "Setup.py" before "main.py".\n'\
    'Alternatively, check that the path to the folder where the .csv files are located is correct.')
else:
    Helper.check_df(folder)
    next_games = Api().get_tomorrows_games()
    
    # 1. Get next matches --> API.get_next_games()
    # 2. Iteratively extract Home_Team and Away_Team
    # 3. Get Average stats for Home_Team and Away_Team
    # 4. Predict the results for the retrieved matches
    # 5. Send the predictions to the Telegram Bot

    # ----- If you want the Telegram Integration ----- #
    TelegramBot().send_message(next_games)
    
    # ----- If you want to run the program at fixed times, uncomment the lines below. ----- #
    sched.start()
    logger.warning('Type "exit" to stop the program.\n')
    inp = input()
    while inp != 'exit':
        logger.warning('Program is running. Type "exit" to stop the program.\n')
        inp = input()
